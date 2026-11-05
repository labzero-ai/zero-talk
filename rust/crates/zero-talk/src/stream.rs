//! The audio of a text as it is made: [`Voice::stream`](crate::Voice::stream) gives a [`Stream`] of [`Chunk`]s.
use crate::error::{Error, Result};
use crate::speak::Joiner;

/// The audio of a text, one [`Chunk`] at a time, in order. Each chunk is made when the iterator is pulled, so the first audio
/// is ready long before the last. Concatenated, the chunks are the samples that [`Voice::speak`](crate::Voice::speak) gives for
/// the same text.
pub struct Stream<'a> {
    inner: Joiner<'a>,
}

impl<'a> Stream<'a> {
    pub(crate) fn new(inner: Joiner<'a>) -> Self {
        Stream { inner }
    }
}

impl Iterator for Stream<'_> {
    type Item = Result<Chunk>;

    fn next(&mut self) -> Option<Result<Chunk>> {
        self.inner.next().map(|piece| piece.map(|samples| Chunk { samples }).map_err(Error::from))
    }
}

/// A piece of the audio of a text: mono samples at 24 kHz.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    samples: Vec<f32>,
}

impl Chunk {
    /// The samples of this piece: mono, from -1 to 1, at 24 kHz.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}
