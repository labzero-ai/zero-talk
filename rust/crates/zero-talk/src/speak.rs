//! The join of the audio of successive sentences. Every sentence is rendered on its own with the seed `seed + i`, loses the first
//! samples of its start, and the kept parts are concatenated. A sentence starts `PRE_ROLL_FRAMES` 80 Hz frames before its start
//! token ends, which makes the pause between sentences match the model's own.
//!
//! The join is a pull-based iterator: each item is one push of the generator, so the caller can play or write the audio while
//! the rest of the text is still being rendered.
use crate::archive::Result;
use crate::kernels::Backend;
use crate::pipeline::{Pipeline, Sentence};
use crate::source::FRAME;

/// The pre-roll of a sentence, in 80 Hz frames.
pub const PRE_ROLL_FRAMES: usize = 11;

/// The samples dropped from the start of a sentence whose start token lasts `start_frames` frames at 40 Hz.
pub fn trimmed_samples(start_frames: usize) -> usize {
    (2 * start_frames).saturating_sub(PRE_ROLL_FRAMES) * FRAME
}

/// The joined audio of the sentences (token ids with their pad ids, and a voice row each), as pieces in order. A piece is never empty.
pub struct Joiner<'a> {
    pipeline: &'a Pipeline,
    backend: Backend,
    seed: u32,
    sentences: std::vec::IntoIter<(Vec<usize>, usize)>,
    /// How many sentences have been started; sentence `i` uses the seed `seed + i`.
    started: usize,
    current: Option<Current<'a>>,
    failed: bool,
}

struct Current<'a> {
    sentence: Sentence<'a>,
    /// Samples still to drop from the start of this sentence.
    skip: usize,
    finished: bool,
}

impl<'a> Joiner<'a> {
    pub fn new(pipeline: &'a Pipeline, backend: Backend, sentences: Vec<(Vec<usize>, usize)>, seed: u32) -> Self {
        Joiner { pipeline, backend, seed, sentences: sentences.into_iter(), started: 0, current: None, failed: false }
    }
}

impl Iterator for Joiner<'_> {
    type Item = Result<Vec<f32>>;

    fn next(&mut self) -> Option<Result<Vec<f32>>> {
        loop {
            if self.failed {
                return None;
            }
            if self.current.is_none() {
                let (ids, voice) = self.sentences.next()?;
                let seed = self.seed.wrapping_add(self.started as u32);
                self.started += 1;
                match self.pipeline.begin(self.backend, &ids, voice, seed) {
                    Ok(sentence) => {
                        let skip = trimmed_samples(sentence.start_frames);
                        self.current = Some(Current { sentence, skip, finished: false });
                    }
                    Err(e) => {
                        self.failed = true;
                        return Some(Err(e));
                    }
                }
            }
            let cur = self.current.as_mut()?;
            let mut audio = match cur.sentence.push_next() {
                Some(audio) => audio,
                None if !cur.finished => {
                    cur.finished = true;
                    cur.sentence.finish_audio()
                }
                None => {
                    self.current = None;
                    continue;
                }
            };
            let cut = cur.skip.min(audio.len());
            cur.skip -= cut;
            if cut < audio.len() {
                audio.drain(..cut);
                return Some(Ok(audio));
            }
        }
    }
}

/// The samples by which `speak` grows its array (1 MiB).
const GROW: usize = 1 << 18;

/// The joined audio of the sentences, whole.
pub fn speak(p: &Pipeline, be: Backend, sentences: Vec<(Vec<usize>, usize)>, seed: u32) -> Result<Vec<f32>> {
    let mut audio: Vec<f32> = Vec::new();
    for piece in Joiner::new(p, be, sentences, seed) {
        let piece = piece?;
        // Room is made in steps of GROW samples at most, not by doubling: a doubled array would be up to twice the audio, and copied at
        // each doubling beside the old one.
        if audio.capacity() - audio.len() < piece.len() {
            audio.reserve_exact(piece.len().max(GROW));
        }
        audio.extend_from_slice(&piece);
    }
    Ok(audio)
}
