//! The voice: a model loaded and ready to speak.
use std::path::Path;

use crate::archive::Archive;
use crate::audio::Audio;
use crate::error::{Error, Result};
use crate::kernels::Backend;
use crate::pipeline::Pipeline;
use crate::source::SEED;
use crate::speak::Joiner;
use crate::stream::Stream;
use crate::text::frontend::Frontend;

/// The identifier of Zero Talk 11M, the only voice of this version.
const ZERO_TALK_11M: &str = "labzero-ai/zero-talk-11m";

/// The sentences of a text: each with its phoneme ids (pad ids at both ends) and its voice row.
pub(crate) type Sentences = Vec<(Vec<usize>, usize)>;

/// The identifiers of the voices that [`Voice::load`] accepts.
///
/// ```
/// assert_eq!(zero_talk::voices(), ["labzero-ai/zero-talk-11m"]);
/// ```
pub fn voices() -> &'static [&'static str] {
    &[ZERO_TALK_11M]
}

/// A voice: a loaded model that turns text into speech.
///
/// ```no_run
/// use zero_talk::Voice;
///
/// let voice = Voice::load("labzero-ai/zero-talk-11m")?;
/// let audio = voice.speak("Hello from a small voice.")?;
/// audio.save("hello.wav")?;
/// for chunk in voice.stream("A longer text, read piece by piece.")? {
///     let chunk = chunk?;
///     println!("{} samples", chunk.samples().len());
/// }
/// # Ok::<(), zero_talk::Error>(())
/// ```
pub struct Voice {
    pipeline: Pipeline,
    frontend: Frontend,
    backend: Backend,
    seed: u32,
}

impl Voice {
    /// Loads a voice by its identifier. The only identifier is `"labzero-ai/zero-talk-11m"` (see [`voices`]). The model is built
    /// into the crate by the default feature `embed`, so this needs no files on disk. An unknown identifier is an error that lists
    /// the valid ones.
    ///
    /// ```
    /// let voice = zero_talk::Voice::load("labzero-ai/zero-talk-11m")?;
    /// # Ok::<(), zero_talk::Error>(())
    /// ```
    pub fn load(id: &str) -> Result<Voice> {
        if id != ZERO_TALK_11M {
            return Err(Error::new(format!("no voice named '{id}'; the voices are: {}", voices().join(", "))));
        }
        Voice::open(None, None)
    }

    /// Loads a voice from model files on disk: `model` is the model file and `frontend` the text front-end file. Both are the
    /// files of Zero Talk 11M. This works without the `embed` feature.
    pub fn from_files(model: impl AsRef<Path>, frontend: impl AsRef<Path>) -> Result<Voice> {
        Voice::open(Some(model.as_ref()), Some(frontend.as_ref()))
    }

    /// Speaks a text: English sentences, with numbers, dates and prices written as digits read out in words. Words that cannot be
    /// read and characters that the model has no sound for are left out, and named in [`Audio::warnings`]. A text with nothing to
    /// say is an error.
    ///
    /// ```
    /// let voice = zero_talk::Voice::load("labzero-ai/zero-talk-11m")?;
    /// let audio = voice.speak("Hello from a small voice.")?;
    /// assert_eq!(audio.sample_rate(), 24_000);
    /// assert!(!audio.samples().is_empty());
    /// # Ok::<(), zero_talk::Error>(())
    /// ```
    pub fn speak(&self, text: &str) -> Result<Audio> {
        let (sentences, warnings) = self.prepare(text)?;
        let samples = crate::speak::speak(&self.pipeline, self.backend, sentences, self.seed)?;
        Ok(Audio::new(samples, warnings))
    }

    /// Speaks a text piece by piece: the iterator makes each chunk when it is pulled, so audio is available before the whole text
    /// is made. The chunks, concatenated, equal the samples of [`speak`](Self::speak) for the same text. A text with nothing to
    /// say is an error.
    ///
    /// ```
    /// let voice = zero_talk::Voice::load("labzero-ai/zero-talk-11m")?;
    /// for chunk in voice.stream("A longer text, read piece by piece.")? {
    ///     let chunk = chunk?;
    ///     let samples: &[f32] = chunk.samples();
    ///     println!("{} samples", samples.len());
    /// }
    /// # Ok::<(), zero_talk::Error>(())
    /// ```
    pub fn stream(&self, text: &str) -> Result<Stream<'_>> {
        let (sentences, _warnings) = self.prepare(text)?;
        Ok(Stream::new(self.joiner(sentences)))
    }

    /// Loads the model and the front end: from the files given, or else the ones built into the crate.
    pub(crate) fn open(model: Option<&Path>, frontend: Option<&Path>) -> Result<Voice> {
        let frontend = Frontend::load(&match frontend {
            Some(path) => checked_frontend(Archive::open(path)?, &path.display().to_string())?,
            None => embedded(Embedded::Frontend)?,
        })?;
        let pipeline = Pipeline::load(&match model {
            Some(path) => checked_model(Archive::open(path)?, &path.display().to_string())?,
            None => embedded(Embedded::Model)?,
        })?;
        Ok(Voice { pipeline, frontend, backend: Backend::Portable, seed: SEED })
    }

    pub(crate) fn set_backend(&mut self, backend: Backend) {
        self.backend = backend;
    }

    pub(crate) fn set_seed(&mut self, seed: u32) {
        self.seed = seed;
    }

    /// The sentences of a text, and the warnings about what was left out.
    pub(crate) fn prepare(&self, text: &str) -> Result<(Sentences, Vec<String>)> {
        let plan = self.frontend.plan(text);
        let mut warnings = Vec::new();
        if !plan.unread.is_empty() {
            warnings.push(format!("{} word(s) could not be read and are left out: {}", plan.unread.len(), plan.unread.join(" ")));
        }
        let mut sentences = Vec::new();
        let mut dropped: Vec<char> = Vec::new();
        for chunk in &plan.chunks {
            let (ids, lost) = self.pipeline.vocab().encode(&chunk.phonemes);
            dropped.extend(lost);
            if ids.len() >= 3 {
                let voice = self.pipeline.voice_index(&ids);
                sentences.push((ids, voice));
            }
        }
        let mut unique: Vec<char> = Vec::new();
        for c in dropped {
            if !unique.contains(&c) {
                unique.push(c);
            }
        }
        if !unique.is_empty() {
            let shown: Vec<String> = unique.iter().map(|c| format!("'{c}'")).collect();
            warnings.push(format!("{} character(s) have no sound in the model and are left out: {}", unique.len(), shown.join(" ")));
        }
        if sentences.is_empty() {
            return Err(Error::new("nothing to say: the text has no words the model can read; pass a sentence made of letters, for example \"Hello.\""));
        }
        Ok((sentences, warnings))
    }

    pub(crate) fn joiner(&self, sentences: Sentences) -> Joiner<'_> {
        Joiner::new(&self.pipeline, self.backend, sentences, self.seed)
    }
}

enum Embedded {
    Model,
    Frontend,
}

#[cfg(feature = "embed")]
fn embedded(which: Embedded) -> Result<Archive> {
    let (bytes, what) = match which {
        Embedded::Model => (zero_talk_model_11m::MODEL, "the model built into this crate"),
        Embedded::Frontend => (zero_talk_model_11m::FRONTEND, "the front-end file built into this crate"),
    };
    let archive = Archive::parse(bytes.to_vec())?;
    match which {
        Embedded::Model => checked_model(archive, what),
        Embedded::Frontend => checked_frontend(archive, what),
    }
}

#[cfg(not(feature = "embed"))]
fn embedded(_which: Embedded) -> Result<Archive> {
    Err(Error::new(
        "this build has no model inside (the default feature `embed` is off): give the model and front-end files with Voice::from_files",
    ))
}

fn checked_model(ar: Archive, what: &str) -> Result<Archive> {
    match ar.meta("kind") {
        Ok("tts") => Ok(ar),
        Ok(kind) => Err(Error::new(format!("{what}: this is a '{kind}' file, not a model file"))),
        Err(_) => Err(Error::new(format!("{what}: not a Zero Talk file"))),
    }
}

fn checked_frontend(ar: Archive, what: &str) -> Result<Archive> {
    if ar.meta("kind") != Ok("frontend") {
        return Err(Error::new(format!("{what}: not a front-end file of Zero Talk 11M")));
    }
    Ok(ar)
}
