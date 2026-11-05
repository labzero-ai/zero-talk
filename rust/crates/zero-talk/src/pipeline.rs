//! The pipeline from phoneme ids to audio: voice row, text encoder, durations, the neutral plan, conditioner, the student's
//! latents, harmonic source, generator and soft limiter. The text front end (phonemes from text) is not part of it.
use crate::archive::{Archive, Result};
use crate::conditioner::{latent_statistics, Conditioner};
use crate::generator::{Generator, Stream};
use crate::kernels::{Act, Backend};
use crate::plan::{bresenham, plan_neutral, LengthCurve, Plan, Vocab};
use crate::post::soft_limit;
use crate::source::{Source, Stft, COLUMNS_PER_FRAME};
use crate::student::TextToAcoustic;

pub const SAMPLE_RATE: u32 = 24_000;
/// The streaming schedule: frames per push to the generator, growing from 8 to 60.
const SCHEDULE: [usize; 4] = [8, 16, 32, 60];

/// The pushes of `n` frames in the streaming schedule, as (first frame, frames).
pub fn ramp(n: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut a = 0;
    SCHEDULE.into_iter().chain(std::iter::repeat(SCHEDULE[SCHEDULE.len() - 1])).map_while(move |size| {
        let c = size.min(n.checked_sub(a).filter(|&left| left > 0)?);
        a += c;
        Some((a - c, c))
    })
}

/// The student's latents for a whole chunk, made by the schedule's own pushes back to back from a fresh state, their outputs stored
/// (the conditioner needs their statistics before the first sample). They equal, bit for bit, what the stream's pushes
/// give the generator, so storing them changes no sample; one push over the whole chunk would differ in the last bits and hold the
/// activations of the whole chunk at once.
pub fn prepass(student: &TextToAcoustic, be: Backend, frames: &Act, plan: &Plan) -> Act {
    let (mut state, mut out) = (student.acoustic_state(), None::<Act>);
    for (a, k) in ramp(frames.t) {
        let l = student.push(be, &mut state, &frames.columns(a, a + k), &plan.scaled_f0[a..a + k], &plan.energy[a..a + k]);
        let out = out.get_or_insert_with(|| Act::zeros(l.c, frames.t));
        for c in 0..l.c {
            out.row_mut(c)[a..a + k].copy_from_slice(l.row(c));
        }
    }
    out.unwrap_or_default()
}

const MAX_TOKENS: usize = 512; // the voice table has 510 rows

pub struct Pipeline {
    gen: Generator,
    student: TextToAcoustic,
    cond: Conditioner,
    vocab: Vocab,
    voices: LengthCurve,
    source: Source,
    stft: Stft,
    /// The generator's norm sites, named as the conditioner names them (`{site}.norm`).
    site_names: Vec<String>,
}

impl Pipeline {
    pub fn load(ar: &Archive) -> Result<Self> {
        let g = ar.part("generator");
        let gen = Generator::load(&g)?;
        let site_names = gen.sites().map(|s| format!("{s}.norm")).collect();
        Ok(Pipeline {
            gen,
            site_names,
            student: TextToAcoustic::load(&ar.part("student"))?,
            cond: Conditioner::load(&ar.part("conditioner"))?,
            vocab: Vocab::load(&g)?,
            voices: LengthCurve::load(&ar.part("voice"))?,
            source: Source::load(&g)?,
            stft: Stft::new(&g.f32("stft.window")?, 5),
        })
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// The voice row for a sentence: its length coordinates' index, one per phoneme count (the ids include a pad at each end).
    pub fn voice_index(&self, ids: &[usize]) -> usize {
        ids.len().saturating_sub(3).min(self.voices.rows() - 1)
    }

    /// Everything a sentence needs before its first push to the generator (the text encoder, durations, plan, conditioner, harmonic
    /// source and its spectrum): the time to the first audio is spent here and in the first push.
    pub fn begin(&self, be: Backend, ids: &[usize], voice: usize, seed: u32) -> Result<Sentence<'_>> {
        if ids.len() < 3 || ids.len() > MAX_TOKENS {
            return Err(format!("{} tokens: a sentence needs 1 to {} phonemes (and a pad id at each end)", ids.len(), MAX_TOKENS - 2));
        }
        if let Some(&bad) = ids.iter().find(|&&i| i >= self.vocab.len()) {
            return Err(format!("token id {bad} is not in the vocabulary (ids 0 to {})", self.vocab.len() - 1));
        }
        if voice >= self.voices.rows() {
            return Err(format!("voice row {voice}: the table has {} rows", self.voices.rows()));
        }
        let coords = self.voices.coords(voice);
        let (feats, dur) = self.student.encode(be, ids, &coords);
        let counts = bresenham(&dur, ids);
        let start_frames = counts[0];
        let (frames, raw_f0, raw_energy) = self.student.frames(be, &feats, &counts);
        let plan = plan_neutral(counts, &raw_f0, &raw_energy, ids, &self.vocab);
        // The conditioner needs the statistics of the whole chunk's latents first, so the student is pushed once, ahead, and what it
        // made is kept for the generator.
        let n = frames.t;
        let latents = prepass(&self.student, be, &frames, &plan);
        drop(frames); // the frame features are not needed again
        let moments = self.cond.moments(&coords, ids, &latent_statistics(&latents))?;
        let sites = self.cond.sites(&moments, self.site_names.iter().map(String::as_str))?;
        let aff = self.gen.condition(&coords, &sites)?;

        let mut source = self.source.stream(seed);
        let mut wave = source.push(&plan.f0_hz);
        wave.extend(source.finish());
        let har = self.stft.transform(&wave);
        Ok(Sentence { latents, n, har, stream: self.gen.stream(be, aff), start_frames, pushes: Box::new(ramp(n)) })
    }
}

/// A sentence that is ready to run (`Pipeline::begin`), pulled through the generator one push at a time.
pub struct Sentence<'a> {
    /// The pre-pass's latents (`prepass`), of which the generator takes the next columns.
    latents: Act,
    /// Frames (80 Hz).
    n: usize,
    har: Act,
    stream: Stream<'a>,
    /// The start token's length in 40 Hz frames.
    pub start_frames: usize,
    pushes: Box<dyn Iterator<Item = (usize, usize)>>,
}

impl Sentence<'_> {
    /// The audio of the next push, after the soft limiter; `None` when the sentence's pushes are all made (then `finish_audio` gives the
    /// rest). The limiter acts on one sample at a time, so the pieces equal the limited whole.
    pub fn push_next(&mut self) -> Option<Vec<f32>> {
        let (a, c) = self.pushes.next()?;
        let latents = self.latents.columns(a, a + c);
        let last = usize::from(a + c == self.n); // the spectrum's final column comes with the last push
        let har = self.har.columns(COLUMNS_PER_FRAME * a, COLUMNS_PER_FRAME * (a + c) + last);
        let mut audio = self.stream.push(&latents, &har);
        soft_limit(&mut audio);
        Some(audio)
    }

    /// The audio that the generator still holds after the last push, after the soft limiter.
    pub fn finish_audio(&mut self) -> Vec<f32> {
        let mut audio = self.stream.finish();
        soft_limit(&mut audio);
        audio
    }
}
