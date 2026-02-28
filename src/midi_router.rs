use crossbeam_channel::{Receiver, Sender};
use tracing::{debug, info};

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
        Self { event_rx, render_tx }
    }

    /// Blocking router loop — call from a dedicated thread.
    pub fn run(&self) {
        info!("MIDI router started");

        for event in self.event_rx.iter() {
            // Fan-out to render (drop if back-pressured — visual is non-critical).
            let _ = self.render_tx.try_send(event);

            // TODO: forward to midir output port
            // TODO: append to SMF recorder
        }

        debug!("MIDI router stopped (channel closed)");
    }
}
