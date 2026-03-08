//! MIDI file loader: parses SMF via `midly`, streams note events to the
//! renderer at real-time pace, and synthesises simple sine-wave audio for
//! playback so the user can hear the notes.

use std::path::Path;

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use tracing::info;

use crate::events::{NoteEvent, NoteEventKind};
use crate::transport::{TransportCommand, TransportState};

// ═══════════════════════════════════════════════════════════════════════════
// Tempo map (tick → seconds conversion with mid-song tempo changes)
// ═══════════════════════════════════════════════════════════════════════════

struct TempoChange {
    tick: u64,
    us_per_beat: u32,
}

enum TimingMode {
    /// Standard MIDI: ticks per quarter-note.
    Metrical(u32),
    /// SMPTE time-code: ticks per second.
    Timecode(f64),
}

struct TempoMap {
    mode: TimingMode,
    changes: Vec<TempoChange>,
}

impl TempoMap {
    fn tick_to_seconds(&self, tick: u64) -> f64 {
        match self.mode {
            TimingMode::Timecode(tps) => tick as f64 / tps,
            TimingMode::Metrical(tpb) => {
                let mut secs = 0.0f64;
                let mut last_tick = 0u64;
                let mut tempo = 500_000u32; // 120 BPM default

                for tc in &self.changes {
                    if tc.tick > tick {
                        break;
                    }
                    let dt = (tc.tick - last_tick) as f64;
                    secs += dt * tempo as f64 / (tpb as f64 * 1_000_000.0);
                    last_tick = tc.tick;
                    tempo = tc.us_per_beat;
                }

                let dt = (tick - last_tick) as f64;
                secs += dt * tempo as f64 / (tpb as f64 * 1_000_000.0);
                secs
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Simple multi-voice sine synthesiser
// ═══════════════════════════════════════════════════════════════════════════

const MAX_VOICES: usize = 128;

struct Voice {
    pitch: u8,
    velocity: f32,
    phase: f32,
    freq: f32,
    active: bool,
    releasing: bool,
    env: f32,
}

struct Synth {
    sample_rate: f32,
    voices: Vec<Voice>,
}

impl Synth {
    fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate: sample_rate as f32,
            voices: Vec::with_capacity(MAX_VOICES),
        }
    }

    fn note_on(&mut self, pitch: u8, velocity: u8) {
        // Reuse existing voice for the same pitch
        for v in &mut self.voices {
            if v.pitch == pitch {
                v.velocity = velocity as f32 / 127.0;
                v.active = true;
                v.releasing = false;
                v.env = 1.0;
                return;
            }
        }
        if self.voices.len() < MAX_VOICES {
            let freq = 440.0 * 2.0f32.powf((pitch as f32 - 69.0) / 12.0);
            self.voices.push(Voice {
                pitch,
                velocity: velocity as f32 / 127.0,
                phase: 0.0,
                freq,
                active: true,
                releasing: false,
                env: 1.0,
            });
        }
    }

    fn note_off(&mut self, pitch: u8) {
        for v in &mut self.voices {
            if v.pitch == pitch && v.active && !v.releasing {
                v.releasing = true;
            }
        }
    }

    /// Render `out.len()` mono samples, mixing all active voices.
    fn render(&mut self, out: &mut [f32]) {
        let inv_sr = 1.0 / self.sample_rate;
        // Release → fade to 0 in ~60 ms
        let release_rate = inv_sr / 0.06;

        for s in out.iter_mut() {
            *s = 0.0;
        }

        for v in &mut self.voices {
            if !v.active {
                continue;
            }
            let dp = v.freq * inv_sr;
            for s in out.iter_mut() {
                // Fundamental + a touch of 2nd harmonic for warmth
                let wave = (v.phase * std::f32::consts::TAU).sin() * 0.80
                    + (v.phase * std::f32::consts::TAU * 2.0).sin() * 0.15
                    + (v.phase * std::f32::consts::TAU * 3.0).sin() * 0.05;
                *s += wave * v.velocity * v.env * 0.12;

                v.phase += dp;
                if v.phase >= 1.0 {
                    v.phase -= 1.0;
                }

                if v.releasing {
                    v.env -= release_rate;
                    if v.env <= 0.0 {
                        v.env = 0.0;
                        v.active = false;
                        break;
                    }
                }
            }
        }

        // Cull dead voices
        self.voices.retain(|v| v.active);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// A timed note event parsed from MIDI.
struct TimedEvent {
    sample_time: u64,
    event: NoteEvent,
}

/// Parse a MIDI file and stream its note events at real-time pace.
///
/// * Events are sent to `render_tx` for visualisation.
/// * Simple sine audio is synthesised and pushed to `playback_producer`.
/// * `clock` is advanced in lockstep so the renderer stays synchronised.
pub fn stream_midi_file(
    path: &Path,
    render_tx: Sender<NoteEvent>,
    mut playback_producer: Option<rtrb::Producer<f32>>,
    clock: std::sync::Arc<crate::timing::AudioClock>,
    sample_rate: u32,
    chunk_size: usize,
    transport_rx: Receiver<TransportCommand>,
    transport_state: std::sync::Arc<TransportState>,
) -> Result<()> {
    let data = std::fs::read(path)
        .with_context(|| format!("Cannot read MIDI file: {}", path.display()))?;
    let smf = midly::Smf::parse(&data).map_err(|e| anyhow::anyhow!("MIDI parse error: {e}"))?;

    // ---- Determine timing mode ----
    let timing_mode = match smf.header.timing {
        midly::Timing::Metrical(tpb) => TimingMode::Metrical(tpb.as_int() as u32),
        midly::Timing::Timecode(fps, sub) => TimingMode::Timecode(fps.as_int() as f64 * sub as f64),
    };

    // ---- Build tempo map from all tracks ----
    let mut tempo_changes: Vec<TempoChange> = Vec::new();
    for track in &smf.tracks {
        let mut abs_tick: u64 = 0;
        for event in track {
            abs_tick += event.delta.as_int() as u64;
            if let midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(t)) = event.kind {
                tempo_changes.push(TempoChange {
                    tick: abs_tick,
                    us_per_beat: t.as_int(),
                });
            }
        }
    }
    tempo_changes.sort_by_key(|tc| tc.tick);
    tempo_changes.dedup_by_key(|tc| tc.tick);

    let tempo_map = TempoMap {
        mode: timing_mode,
        changes: tempo_changes,
    };

    // ---- Collect all note events with absolute sample times ----
    let mut events: Vec<TimedEvent> = Vec::new();

    for track in &smf.tracks {
        let mut abs_tick: u64 = 0;
        for event in track {
            abs_tick += event.delta.as_int() as u64;
            let (kind, pitch, velocity) = match event.kind {
                midly::TrackEventKind::Midi { message, .. } => match message {
                    midly::MidiMessage::NoteOn { key, vel } => {
                        if vel.as_int() == 0 {
                            (NoteEventKind::NoteOff, key.as_int(), 0u8)
                        } else {
                            (NoteEventKind::NoteOn, key.as_int(), vel.as_int())
                        }
                    }
                    midly::MidiMessage::NoteOff { key, .. } => {
                        (NoteEventKind::NoteOff, key.as_int(), 0u8)
                    }
                    midly::MidiMessage::Controller { controller, value } => {
                        // CC64 = sustain pedal
                        if controller.as_int() == 64 {
                            if value.as_int() >= 64 {
                                (NoteEventKind::PedalOn, 0, 0)
                            } else {
                                (NoteEventKind::PedalOff, 0, 0)
                            }
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                },
                _ => continue,
            };

            let seconds = tempo_map.tick_to_seconds(abs_tick);
            let sample_time = (seconds * sample_rate as f64) as u64;

            events.push(TimedEvent {
                sample_time,
                event: NoteEvent {
                    kind,
                    pitch,
                    velocity,
                    sample_time,
                },
            });
        }
    }

    // Stable sort preserves per-track ordering for same-tick events
    events.sort_by_key(|e| e.sample_time);

    if events.is_empty() {
        info!("MIDI file contains no note events");
        return Ok(());
    }

    let last_sample = events.last().map(|e| e.sample_time).unwrap_or(0);
    let total_seconds = last_sample as f64 / sample_rate as f64;
    info!(
        "MIDI loaded: {} note events, {:.1}s, {} tracks",
        events.len(),
        total_seconds,
        smf.tracks.len()
    );

    // ---- Real-time streaming with synthesis ----
    let mut synth = Synth::new(sample_rate);
    let mut event_idx = 0;
    let mut current_sample: u64 = 0;
    let sleep_per_chunk =
        std::time::Duration::from_secs_f64(chunk_size as f64 / sample_rate as f64);
    // Extra 1s after last note for release tails
    let end_sample = last_sample + sample_rate as u64;
    transport_state.set_total_samples(end_sample);

    let mut audio_buf = vec![0.0f32; chunk_size];
    let mut paused = transport_state.is_paused();

    let mut seek_event_index = |target: u64| -> usize {
        let mut lo = 0usize;
        let mut hi = events.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if events[mid].sample_time < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    };

    while current_sample < end_sample {
        while let Ok(cmd) = transport_rx.try_recv() {
            match cmd {
                TransportCommand::Play => {
                    paused = false;
                    transport_state.set_paused(false);
                }
                TransportCommand::Pause => {
                    paused = true;
                    transport_state.set_paused(true);
                }
                TransportCommand::SeekSamples(sample) => {
                    let target = sample.min(end_sample);
                    current_sample = target;
                    clock.set_samples(target);
                    event_idx = seek_event_index(target);
                    synth = Synth::new(sample_rate);
                    audio_buf.fill(0.0);
                }
            }
        }

        if paused {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }

        let chunk_end = current_sample + chunk_size as u64;

        // Emit note events whose time falls within this chunk
        while event_idx < events.len() && events[event_idx].sample_time < chunk_end {
            let te = &events[event_idx];

            // Send to renderer
            let _ = render_tx.try_send(te.event);

            // Drive synth
            match te.event.kind {
                NoteEventKind::NoteOn => synth.note_on(te.event.pitch, te.event.velocity),
                NoteEventKind::NoteOff => synth.note_off(te.event.pitch),
                NoteEventKind::PedalOn | NoteEventKind::PedalOff => {} // visual only
            }
            event_idx += 1;
        }

        // Render synthesised audio for this chunk
        synth.render(&mut audio_buf);

        // Push to playback ring
        if let Some(ref mut pb) = playback_producer {
            for &s in &audio_buf {
                while pb.push(s).is_err() {
                    std::hint::spin_loop();
                }
            }
        }

        clock.advance(chunk_size as u64);
        current_sample += chunk_size as u64;
        std::thread::sleep(sleep_per_chunk);
    }

    info!("MIDI playback complete");
    Ok(())
}
