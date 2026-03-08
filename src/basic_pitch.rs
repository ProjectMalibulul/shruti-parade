//! Integration with spotify/basic-pitch for audio-to-MIDI conversion.
//!
//! basic-pitch is a Python package that uses a neural network to transcribe
//! audio to MIDI.  We invoke it as a subprocess and read the resulting MIDI file.

use std::path::{Path, PathBuf};
use std::process::Command;
use anyhow::{Context, Result};
use tracing::info;

/// Transcribe an audio file to MIDI using the basic-pitch CLI.
///
/// Requires `basic-pitch` to be installed (`pip install basic-pitch`).
/// Returns the path to the generated MIDI file.
pub fn transcribe_to_midi(audio_path: &Path, output_dir: &Path) -> Result<PathBuf> {
    let audio_path = audio_path
        .canonicalize()
        .with_context(|| format!("Cannot resolve audio path: {}", audio_path.display()))?;

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Cannot create output dir: {}", output_dir.display()))?;

    info!(
        "Running basic-pitch: {} → {}",
        audio_path.display(),
        output_dir.display()
    );

    let status = Command::new("basic-pitch")
        .arg(output_dir)
        .arg(&audio_path)
        .arg("--save-midi")
        .status()
        .context("Failed to run basic-pitch — is it installed? (pip install basic-pitch)")?;

    if !status.success() {
        anyhow::bail!(
            "basic-pitch exited with status {} for {}",
            status,
            audio_path.display()
        );
    }

    // basic-pitch outputs <stem>_basic_pitch.mid in the output directory
    let stem = audio_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let midi_path = output_dir.join(format!("{stem}_basic_pitch.mid"));

    if !midi_path.exists() {
        anyhow::bail!(
            "Expected MIDI output not found: {}",
            midi_path.display()
        );
    }

    info!("basic-pitch transcription complete: {}", midi_path.display());
    Ok(midi_path)
}
