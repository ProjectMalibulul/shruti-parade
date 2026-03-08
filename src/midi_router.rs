use crossbeam_channel::{Receiver, Sender};
use tracing::info;

use crate::events::NoteEvent;

/// Routes note events from the inference engine to:
///   - The render thread (for visualisation)
///   - (TODO) A `midir` output port (for external synths)
///   - (TODO) An SMF writer (for recording)
pub struct MidiRouter {
    event_rx: Receiver<NoteEvent>,
    render_tx: Sender<NoteEvent>,
}

impl MidiRouter {
    pub fn new(event_rx: Receiver<NoteEvent>, render_tx: Sender<NoteEvent>) -> Self {
        Self {
            event_rx,
            render_tx,
        }
    }

    /// Blocking router loop — call from a dedicated thread.
    pub fn run(&self) {
        info!("MIDI router started");
        let mut routed: u64 = 0;

        for event in self.event_rx.iter() {
            // Fan-out to render (drop if back-pressured — visual is non-critical).
            let _ = self.render_tx.try_send(event);
            routed += 1;
            if routed == 1 {
                info!("MIDI router: first event forwarded to renderer");
            }

            // TODO: forward to midir output port
            // TODO: append to SMF recorder
        }

        info!("MIDI router stopped — {routed} events routed");
    }
}
