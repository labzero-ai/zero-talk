//! WebAssembly build of the engine (the `zero-talk` crate): the path from text to audio of the `zero-talk` command, with the portable
//! matrix-multiply backend, for the npm package (js/dist/engine.wasm, built by build.sh). The model must be a model file, and the front
//! end's file is checked as `zero-talk` checks it.
//!
//! The C ABI, with no bindings. The caller writes its inputs into memory from zt_alloc. The module keeps one loaded engine and the
//! result of the last call, which the caller reads through the pointer and length functions below.
//! - zt_load(model, model_len, frontend, frontend_len) -> 0, or -1 (zt_error_*).
//! - zt_speak(text, text_len, seed, format) -> 0, or -1 (zt_error_*). Format 0 is WAV (16-bit, 24 kHz), 1 is Ogg Opus at
//!   24 kbit/s, 2 is float32 samples (little endian, for the checks). The bytes are at zt_out_ptr / zt_out_len, the audio's
//!   length in seconds at zt_duration.
//! - zt_info_ptr / zt_info_len: the last text's phonemes, one line per chunk as "ROW<TAB>PHONEMES" (the lines of
//!   `zero-talk dev phonemize`), then warnings, each starting "warning: ", as `zero-talk` prints them.
//! - zt_error_ptr / zt_error_len: the message of the last failed call (empty after a success).
//!
//! The GPU path (js/dist/webgpu): the generator and the acoustic blocks' pre-pass run in JavaScript (the pre-pass on WebGPU), so this part loads
//! without the generator's weights and gives each sentence's inputs to them (the front half of zero_talk::pipeline::Pipeline::begin).
//! - zt_load_front(model, model_len, frontend, frontend_len) -> 0, or -1: as zt_load, without the generator.
//! - zt_gpu_begin(text, text_len) -> the sentences' count, or -1: the text's info (as zt_info_*) and its sentences, kept.
//! - zt_gpu_sentence(i, seed) -> 0, or -1: the whole front half of sentence i in WebAssembly, with the base seed (its source is
//!   seeded seed + i). Then zt_gpu_x_ptr / zt_gpu_x_len (the acoustic blocks' latents, [512][frames] float32), zt_gpu_har_ptr /
//!   zt_gpu_har_len (the source's spectrum, [22][60 frames + 1]), zt_gpu_frames, zt_gpu_start_frames, zt_gpu_trim (samples
//!   dropped at the start), zt_gpu_coords_ptr (4 length coordinates), zt_gpu_moments_ptr / zt_gpu_moments_len (per site, its mean
//!   then its variance), zt_gpu_site_channels_ptr (channels per site), zt_gpu_site_count, zt_gpu_site_names_ptr /
//!   zt_gpu_site_names_len (the names without ".norm", joined by newlines, in the order of the moments).
//! - The front's three calls, zt_gpu_front, zt_gpu_source and zt_gpu_moments, with the pre-pass on the GPU between the first two.
//!   zt_gpu_front(i, seed) -> 0, or -1: sentence i's text
//!   encoder, durations, frames and plan. Its pre-pass inputs are zt_gpu_front_frames_ptr (the frame features, [384][frames]),
//!   zt_gpu_front_f0_ptr and zt_gpu_front_energy_ptr (the scaled F0 and the energy, [frames]). zt_gpu_source(frames) -> 0, or -1:
//!   the source's spectrum (zt_gpu_har_*) as far as `frames` frames of the sentence need it: columns [0, 60 min(frames, F) + 1) are
//!   final on return, and a larger frames computes more (frames >= F is the whole spectrum). zt_gpu_moments(stats_ptr, stats_len)
//!   -> 0, or -1: the moments (zt_gpu_moments_*, and the channels) of the latents' 1024 statistics (latent_statistics: 512 channel
//!   means, then 512 values ln(sqrt(var) + 1e-4)).
//!   zt_gpu_frames, zt_gpu_start_frames and zt_gpu_trim hold from zt_gpu_front on, and from zt_gpu_counts on the prepare path
//!   below; zt_gpu_coords_ptr from zt_gpu_front or zt_gpu_prepare on.
//! - The prepare path, zt_gpu_prepare, zt_gpu_counts and zt_gpu_plan: the text encoder and the frame stage on the GPU too (the
//!   worker's path: js/dist/webgpu/latents.js). Its calls, in this order:
//!   zt_gpu_prepare(i, seed) -> 0, or -1: sentence i's token ids (zt_gpu_ids_ptr / zt_gpu_ids_len, u32), its coordinates, and the
//!   source's seed (seed + i). zt_gpu_counts(dur_ptr, n) -> 0, or -1: the GPU's durations (n = the tokens, softplus applied) give the
//!   40 Hz counts (zt_gpu_counts_ptr, u32), and with them the frames, start and trim, as zt_gpu_front gives them. zt_gpu_plan(f0_ptr,
//!   energy_ptr, count) -> 0, or -1: the GPU's raw scaled F0 and energy (count = zt_gpu_frames) give the plan, which
//!   zt_gpu_front_f0_ptr and zt_gpu_front_energy_ptr read back. Then zt_gpu_source and zt_gpu_moments, as after zt_gpu_front.
//!   The frame features stay on the GPU on this path: zt_gpu_front_frames_ptr and zt_gpu_front_frames_len give none (length 0).
//!   zt_gpu_counts and zt_gpu_plan reject a NaN, an infinite, a negative or an over-large duration, and a NaN or infinite raw F0 or
//!   energy: -1 with the message, and nothing stays prepared, so the worker falls back to zt_gpu_front, which stays the reference.
//! - zt_post_limit(ptr, count): the soft limiter in place on float32 samples at ptr.
//! - zt_encode(ptr, count, format) -> 0, or -1: float32 samples at ptr encoded as zt_speak encodes its audio.
use std::cell::RefCell;
use std::fmt::Write as _;

use zero_talk::archive::Archive;
use zero_talk::conditioner::{latent_statistics, Conditioner};
use zero_talk::kernels::{Act, Backend};
use zero_talk::opus;
use zero_talk::pipeline::{prepass, Pipeline, SAMPLE_RATE};
use zero_talk::plan::{bresenham, plan_neutral, LengthCurve, Plan, Vocab, COORDS};
use zero_talk::post::soft_limit;
use zero_talk::source::{Source, SourceStream, Stft};
use zero_talk::speak::{speak, trimmed_samples};
use zero_talk::student::TextToAcoustic;
use zero_talk::text::frontend::Frontend;
use zero_talk::wav;

/// Pipeline::begin's limit: 510 phonemes and a pad id at each end (the voice table has 510 rows).
const MAX_TOKENS: usize = 512;

/// latent_statistics' length: the 512 channel means, then the 512 log standard deviations.
const STATISTICS: usize = 1024;

/// zt_gpu_counts' bound on a duration (40 Hz units): f32 holds whole numbers exactly up to 2^24, so the counts are exact and fit u32.
const MAX_DURATION: f32 = 16_777_216.0;

/// What zt_load_front keeps for the GPU path: the generator's inputs, without the generator's weights.
struct GpuFront {
    /// Loaded by zt_load_student, only when a sentence runs its front half in WebAssembly (the fallback of the GPU path).
    student: Option<TextToAcoustic>,
    cond: Conditioner,
    vocab: Vocab,
    voices: LengthCurve,
    /// Leaked by zt_load_front, so that a SourceRun's stream can borrow it for 'static (40 bytes per zt_load_front call).
    source: &'static Source,
    stft: Stft,
    /// The conditioner's site names with ".norm", in the model's order (the names Conditioner::sites looks up).
    sites: Vec<String>,
    /// The same names without ".norm", joined by newlines (zt_gpu_site_names_*).
    names: String,
}

/// How far the sentence last prepared has got. zt_gpu_source and zt_gpu_moments need Planned.
#[derive(Default, PartialEq)]
enum Stage {
    /// Nothing prepared: after zt_gpu_begin, after a failed call, and after a successful zt_gpu_sentence (the calls after that
    /// refuse, as they do after nothing).
    #[default]
    Empty,
    /// zt_gpu_prepare: the token ids, the coordinates and the source's seed.
    Prepared,
    /// zt_gpu_counts: and the frames, start and trim.
    Counted,
    /// zt_gpu_front or zt_gpu_plan: and the plan.
    Planned,
}

/// The generator's inputs for the sentence last prepared: zt_gpu_sentence's, zt_gpu_front's, or the prepare path's (zt_gpu_prepare,
/// zt_gpu_counts, zt_gpu_plan), with the stage it has reached.
#[derive(Default)]
struct Prepared {
    stage: Stage,
    frames: usize,
    start_frames: usize,
    /// zt_gpu_sentence's acoustic latents, [512][frames] row-major.
    x: Vec<f32>,
    /// The harmonic source's spectrum, [22][60 frames + 1] row-major: zt_gpu_sentence fills all of it, zt_gpu_source as far as asked
    /// (allocated zeros by its first call, then filled in place, so the pointer holds).
    har: Vec<f32>,
    coords: [f32; COORDS],
    /// Per site in order: its mean, then its variance.
    moments: Vec<f32>,
    /// The channels of each site.
    channels: Vec<u32>,
    /// zt_gpu_front's frame features, [384][frames] row-major, and its scaled F0 and energy, [frames] each: the pre-pass's inputs.
    /// The features are empty after zt_gpu_plan (they stay on the GPU).
    feats: Vec<f32>,
    f0: Vec<f32>,
    energy: Vec<f32>,
    /// What zt_gpu_source and zt_gpu_moments need from the plan: the source's F0 in Hz and its seed (the base seed + i). The token ids
    /// are the sentence's inputs (plan_neutral, the moments).
    f0_hz: Vec<f32>,
    source_seed: u32,
    ids: Vec<usize>,
    /// The token ids as u32 (zt_gpu_ids_*), and zt_gpu_counts' 40 Hz counts (zt_gpu_counts_*), one per token.
    token_ids: Vec<u32>,
    counts: Vec<u32>,
    /// zt_gpu_source's run: none until its first call. Each zt_gpu_prepare, zt_gpu_front or zt_gpu_sentence call drops it.
    run: Option<SourceRun>,
}

#[derive(Default)]
struct State {
    frontend: Option<Frontend>,
    pipeline: Option<Pipeline>,
    source: Option<(Source, Stft)>,
    gpu: Option<GpuFront>,
    /// The sentences of the last text (zt_gpu_begin): token ids and voice row.
    sentences: Vec<(Vec<usize>, usize)>,
    prepared: Prepared,
    out: Vec<u8>,
    duration: f64,
    info: String,
    error: String,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn with<T>(f: impl FnOnce(&mut State) -> T) -> T {
    STATE.with(|s| f(&mut s.borrow_mut()))
}

/// Records the outcome of a call for zt_error and returns its value (the count, or 0), or -1 on failure.
fn outcome_value(r: Result<i32, String>) -> i32 {
    with(|st| match r {
        Ok(v) => {
            st.error.clear();
            v
        }
        Err(e) => {
            st.error = e;
            -1
        }
    })
}

/// Records the outcome of a call for zt_error: 0 for success, -1 for failure.
fn outcome(r: Result<(), String>) -> i32 {
    outcome_value(r.map(|()| 0))
}

/// The bytes the caller wrote at `ptr`.
///
/// # Safety
/// `ptr` must address `len` readable bytes (any pointer when `len` is 0).
unsafe fn input<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

/// The float32 samples the caller wrote at `ptr`.
///
/// # Safety
/// `ptr` must address `count` float32 values (zt_alloc's blocks are aligned for them), or `count` must be 0.
unsafe fn floats<'a>(ptr: *const f32, count: usize) -> &'a [f32] {
    if count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, count)
    }
}

/// Memory for the caller to write into; free it with zt_free(ptr, len).
#[no_mangle]
pub extern "C" fn zt_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Frees a block from zt_alloc(len).
///
/// # Safety
/// `ptr` and `len` must be those of one block from zt_alloc, freed once.
#[no_mangle]
pub unsafe extern "C" fn zt_free(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len));
}

/// Loads the model and the front end from their bytes.
///
/// # Safety
/// Each pointer must address its length in readable bytes.
#[no_mangle]
pub unsafe extern "C" fn zt_load(model: *const u8, model_len: usize, frontend: *const u8, frontend_len: usize) -> i32 {
    outcome(load(input(model, model_len), input(frontend, frontend_len)))
}

/// The checks of the `zero-talk` command (checked_model, checked_frontend in the crate's cli module): the two files' kinds.
fn checked(model: &[u8], frontend: &[u8]) -> Result<(Archive, Archive), String> {
    let fe = Archive::parse(frontend.to_vec()).map_err(|e| format!("front-end file: {e}"))?;
    if fe.meta("kind") != Ok("frontend") {
        return Err("front-end file: not a front-end data file; use the front-end file that ships with this version".into());
    }
    let md = Archive::parse(model.to_vec()).map_err(|e| format!("model file: {e}"))?;
    match md.meta("kind") {
        Ok("tts") => {}
        Ok(kind) => return Err(format!("model file: this is a {kind} file, not a model file")),
        Err(e) => return Err(format!("model file: {e}")),
    }
    Ok((fe, md))
}

fn load(model: &[u8], frontend: &[u8]) -> Result<(), String> {
    let (fe, md) = checked(model, frontend)?;
    let frontend = Frontend::load(&fe)?;
    let pipeline = Pipeline::load(&md)?;
    let generator = md.part("generator");
    let source = (Source::load(&generator)?, Stft::new(&generator.f32("stft.window")?, 5));
    with(|st| {
        st.frontend = Some(frontend);
        st.pipeline = Some(pipeline);
        st.source = Some(source);
    });
    Ok(())
}

/// Loads the GPU path's inputs (zt_gpu_begin, zt_gpu_sentence) from the model and the front end, without the generator's weights.
///
/// # Safety
/// Each pointer must address its length in readable bytes.
#[no_mangle]
pub unsafe extern "C" fn zt_load_front(model: *const u8, model_len: usize, frontend: *const u8, frontend_len: usize) -> i32 {
    outcome(load_front(input(model, model_len), input(frontend, frontend_len)))
}

fn load_front(model: &[u8], frontend: &[u8]) -> Result<(), String> {
    let (fe, md) = checked(model, frontend)?;
    let frontend = Frontend::load(&fe)?;
    let generator = md.part("generator");
    let sites = site_names(md.meta("conditioner_sites")?)?;
    let names = sites.iter().map(|s| s.as_str().strip_suffix(".norm").unwrap_or(s.as_str())).collect::<Vec<_>>().join("\n");
    let gpu = GpuFront {
        student: None,
        cond: Conditioner::load(&md.part("conditioner"))?,
        vocab: Vocab::load(&generator)?,
        voices: LengthCurve::load(&md.part("voice"))?,
        source: Box::leak(Box::new(Source::load(&generator)?)),
        stft: Stft::new(&generator.f32("stft.window")?, 5),
        sites,
        names,
    };
    with(|st| {
        st.prepared = Prepared::default();
        st.frontend = Some(frontend);
        st.gpu = Some(gpu);
    });
    Ok(())
}

/// Loads the student that zt_load_front leaves out: the GPU path needs it only when a sentence runs its front half in WebAssembly.
///
/// # Safety
/// The pointer must address its length in readable bytes.
#[no_mangle]
pub unsafe extern "C" fn zt_load_student(model: *const u8, model_len: usize) -> i32 {
    outcome(load_student(input(model, model_len)))
}

fn load_student(model: &[u8]) -> Result<(), String> {
    let md = Archive::parse(model.to_vec()).map_err(|e| format!("model file: {e}"))?;
    let student = TextToAcoustic::load(&md.part("student"))?;
    with(|st| match st.gpu.as_mut() {
        Some(gpu) => {
            gpu.student = Some(student);
            Ok(())
        }
        None => Err("no GPU front loaded: call zt_load_front first".to_string()),
    })
}

/// The conditioner's site names, with their ".norm" suffix: the "name:start:end" entries of the model's `conditioner_sites`, read as
/// Conditioner::load reads them.
fn site_names(meta: &str) -> Result<Vec<String>, String> {
    meta.split(';')
        .map(|entry| -> Result<String, String> {
            let name = entry.rsplitn(3, ':').nth(2).ok_or(format!("conditioner site '{entry}'"))?;
            if !name.ends_with(".norm") {
                return Err(format!("conditioner site '{name}' does not end in .norm"));
            }
            Ok(name.to_string())
        })
        .collect()
}

/// Speaks the UTF-8 text with the given seed, in the given format (see the module comment).
///
/// # Safety
/// `text` must address `text_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn zt_speak(text: *const u8, text_len: usize, seed: u32, format: u32) -> i32 {
    let text = String::from_utf8_lossy(input(text, text_len)).into_owned();
    outcome(speak_text(&text, seed, format))
}

/// The formats of zt_speak and zt_encode: 0 WAV, 1 Ogg Opus, 2 float32.
fn check_format(format: u32) -> Result<(), String> {
    if format > 2 {
        return Err(format!("format {format} is not 0 (WAV), 1 (Ogg Opus) or 2 (float32)"));
    }
    Ok(())
}

/// The front half of `zero-talk` (the `speak` command): the info text (one line per chunk, then the warnings) and the sentences, as
/// (token ids, voice row) of each chunk with at least one phoneme. `voice_of` gives the voice row of the ids.
fn plan_text(frontend: &Frontend, vocab: &Vocab, voice_of: impl Fn(&[usize]) -> usize, text: &str) -> (String, Vec<(Vec<usize>, usize)>) {
    let plan = frontend.plan(text);
    let mut info = String::new();
    for c in &plan.chunks {
        let _ = writeln!(info, "{}\t{}", c.voice_row, c.phonemes);
    }
    let mut sentences = Vec::new();
    let mut dropped: Vec<char> = Vec::new();
    for c in &plan.chunks {
        let (ids, lost) = vocab.encode(&c.phonemes);
        dropped.extend(lost);
        if ids.len() >= 3 {
            let voice = voice_of(ids.as_slice());
            sentences.push((ids, voice));
        }
    }
    if !plan.tags.is_empty() {
        let _ = writeln!(info, "warning: steering is not supported in this build: removed {} tag(s) from the text: {}", plan.tags.len(), plan.tags.join(" "));
    }
    if !plan.unread.is_empty() {
        let _ = writeln!(
            info,
            "warning: {} word(s) could not be read and are left out: {}",
            plan.unread.len(),
            plan.unread.join(" ")
        );
    }
    let mut unique: Vec<char> = Vec::new();
    for c in dropped {
        if !unique.contains(&c) {
            unique.push(c);
        }
    }
    if !unique.is_empty() {
        let shown: Vec<String> = unique.iter().map(|c| format!("'{c}'")).collect();
        let _ = writeln!(
            info,
            "warning: dropped {} character(s) that are not in Kokoro's phoneme vocabulary: {}",
            unique.len(),
            shown.join(" ")
        );
    }
    (info, sentences)
}

/// The last step of `zero-talk`: the audio's duration and its bytes in `format`.
fn set_audio(st: &mut State, audio: &[f32], format: u32) -> Result<(), String> {
    st.duration = audio.len() as f64 / SAMPLE_RATE as f64;
    st.out = match format {
        0 => wav::encode(audio, SAMPLE_RATE),
        1 => opus::encode_ogg(audio, SAMPLE_RATE, opus::DEFAULT_KBPS)?,
        _ => audio.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    Ok(())
}

/// `zero-talk`'s glue (the `speak` command, without the timing and the file): text, chunks, token ids, the sentences' audio.
fn speak_text(text: &str, seed: u32, format: u32) -> Result<(), String> {
    check_format(format)?;
    with(|st| {
        let (Some(frontend), Some(pipeline)) = (st.frontend.as_ref(), st.pipeline.as_ref()) else {
            return Err("no model loaded: call zt_load first".to_string());
        };
        let (info, sentences) = plan_text(frontend, pipeline.vocab(), |ids| pipeline.voice_index(ids), text);
        st.info = info;
        if sentences.is_empty() {
            return Err("nothing to say: the text has no words the model can read; pass a sentence made of letters, for example \"Hello.\"".into());
        }
        let audio = speak(pipeline, Backend::Portable, sentences, seed)?;
        set_audio(st, &audio, format)
    })
}

/// The voice row of a sentence for the GPU path, which keeps no Pipeline: Pipeline::voice_index's rule.
fn voice_index(ids: &[usize], rows: usize) -> usize {
    ids.len().saturating_sub(3).min(rows - 1)
}

/// The front half of zt_speak for the GPU path: the info and the sentences are kept, and the sentences' count is returned.
fn gpu_begin(text: &str) -> Result<i32, String> {
    with(|st| {
        let (Some(frontend), Some(gpu)) = (st.frontend.as_ref(), st.gpu.as_ref()) else {
            return Err("no GPU front loaded: call zt_load_front first".to_string());
        };
        let (info, sentences) = plan_text(frontend, &gpu.vocab, |ids| voice_index(ids, gpu.voices.rows()), text);
        st.info = info;
        st.sentences = sentences;
        st.prepared = Prepared::default();
        if st.sentences.is_empty() {
            return Err("nothing to say: the text has no words the model can read; pass a sentence made of letters, for example \"Hello.\"".into());
        }
        Ok(st.sentences.len() as i32)
    })
}

/// The text's info and sentences for the GPU path (zt_gpu_sentence takes them one at a time). Returns the sentences' count, or -1.
///
/// # Safety
/// `text` must address `text_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn zt_gpu_begin(text: *const u8, text_len: usize) -> i32 {
    let text = String::from_utf8_lossy(input(text, text_len)).into_owned();
    outcome_value(gpu_begin(&text))
}

/// Sentence i of the last text: its token ids and voice row, after the checks of Pipeline::begin (in its order).
fn sentence_inputs<'a>(gpu: &GpuFront, sentences: &'a [(Vec<usize>, usize)], i: usize) -> Result<(&'a [usize], usize), String> {
    let Some((ids, voice)) = sentences.get(i) else {
        return Err(format!("sentence {i}: the last text has {} sentence(s) (zt_gpu_begin)", sentences.len()));
    };
    let (ids, voice) = (ids.as_slice(), *voice);
    if ids.len() < 3 || ids.len() > MAX_TOKENS {
        return Err(format!("{} tokens: a sentence needs 1 to {} phonemes (and a pad id at each end)", ids.len(), MAX_TOKENS - 2));
    }
    if let Some(&bad) = ids.iter().find(|&&t| t >= gpu.vocab.len()) {
        return Err(format!("token id {bad} is not in the vocabulary (ids 0 to {})", gpu.vocab.len() - 1));
    }
    if voice >= gpu.voices.rows() {
        return Err(format!("voice row {voice}: the table has {} rows", gpu.voices.rows()));
    }
    Ok((ids, voice))
}

/// What the front half's first stages give a sentence (gpu_sentence's stages 1 to 3).
struct Planned {
    coords: [f32; COORDS],
    start_frames: usize,
    /// The frame features, [384][frames].
    frames: Act,
    /// The durations in 40 Hz frames, one per token.
    counts: Vec<usize>,
    plan: Plan,
}

/// Stages 1 to 3 of gpu_sentence: the text encoder, the durations (bresenham), the frame block and the plan (plan_frames).
fn plan_sentence(gpu: &GpuFront, ids: &[usize], voice: usize) -> Result<Planned, String> {
    let be = Backend::Portable;
    let coords = gpu.voices.coords(voice);
    let student = gpu.student.as_ref().ok_or("the student is not loaded: call zt_load_student first")?;
    let (feats, dur) = student.encode(be, ids, &coords);
    let counts = bresenham(&dur, ids);
    let start_frames = counts[0];
    let (frames, raw_f0, raw_energy) = student.frames(be, &feats, &counts);
    let plan = plan_frames(&counts, &raw_f0, &raw_energy, ids, &gpu.vocab, frames.t)?;
    Ok(Planned { coords, start_frames, frames, counts, plan })
}

/// plan_neutral on a sentence's raw F0 and energy, with its length check: the plan has one F0 value per frame, `frames` of them. Shared by
/// plan_sentence and zt_gpu_plan.
fn plan_frames(counts40: &[usize], raw_f0: &[f32], raw_energy: &[f32], ids: &[usize], vocab: &Vocab, frames: usize) -> Result<Plan, String> {
    let plan = plan_neutral(counts40.to_vec(), raw_f0, raw_energy, ids, vocab);
    if plan.f0_hz.len() != frames {
        return Err(format!("F0 has {} frames, the plan {frames}", plan.f0_hz.len()));
    }
    Ok(plan)
}

/// Stage 5 of gpu_sentence: the conditioner's moments of the latents' statistics, per site (its mean, then its variance) flattened,
/// and the channels of each site.
fn site_moments(gpu: &GpuFront, coords: &[f32; COORDS], ids: &[usize], stats: &[f32]) -> Result<(Vec<f32>, Vec<u32>), String> {
    let moments = gpu.cond.moments(coords, ids, stats)?;
    let sites = gpu.cond.sites(&moments, gpu.sites.iter().map(String::as_str))?;
    let (mut flat, mut channels) = (Vec::new(), Vec::with_capacity(sites.len()));
    for (mean, variance) in &sites {
        channels.push(mean.len() as u32);
        flat.extend_from_slice(mean);
        flat.extend_from_slice(variance);
    }
    Ok((flat, channels))
}

/// Stage 6 of gpu_sentence: the harmonic source's spectrum for the F0 `f0_hz` (Hz per frame), its random phase seeded `seed`.
fn source_spectrum(gpu: &GpuFront, seed: u32, f0_hz: &[f32]) -> Act {
    let mut source = gpu.source.stream(seed);
    let mut wave = source.push(f0_hz);
    wave.extend(source.finish());
    gpu.stft.transform(&wave)
}

/// The harmonic source of the sentence prepared by zt_gpu_front, computed as zt_gpu_source asks for it. The stream takes the F0 in
/// groups of four frames (any split of the pushes gives the same wave), so the wave grows with the calls, and a spectrum column is
/// transformed once its window is in the wave.
struct SourceRun {
    stream: SourceStream<'static>,
    /// The source's samples so far: a prefix of the whole sentence's wave.
    wave: Vec<f32>,
    /// The frames pushed to the stream. With all of them pushed the stream is finished (its last push finishes it) and the wave whole.
    pushed: usize,
    /// The spectrum's columns final so far: zt_gpu_har's columns [0, cols) equal the whole sentence's, bit for bit.
    cols: usize,
}

/// The frames per push of a SourceRun: the group size of the source's stream.
const PUSH: usize = 4;

impl SourceRun {
    /// Runs the source on, and the spectrum with it, until the columns [0, want) of `har` are final (possibly more: the wave comes in
    /// groups of four frames). Column c is the STFT of the wave's samples 5c - 10 .. 5c + 9, reflected at the ends of the whole wave as
    /// Stft::transform reflects them, so it is final once the wave holds sample 5c + 9 (or, with all the frames pushed, the whole wave
    /// does; an unfinished wave is a whole number of 4-frame groups, 1,200 samples, so column 0's sample 10 is always there). The new columns come from a transform of the wave from sample 5a on, a = cols - 2 (or 0 when cols < 2): its first two
    /// columns have the wrong reflection on the left, and its local column f is the global column a + f, so the new columns are local
    /// columns g - a for g in [cols, ready), and the last columns, which may reflect on the right, are not copied.
    fn grow(&mut self, f0_hz: &[f32], stft: &Stft, har: &mut [f32], want: usize) {
        let total = 60 * f0_hz.len() + 1;
        let need = 5 * (want - 1) + 10;
        while self.wave.len() < need && self.pushed < f0_hz.len() {
            let end = (self.pushed + PUSH).min(f0_hz.len());
            self.wave.extend(self.stream.push(&f0_hz[self.pushed..end]));
            self.pushed = end;
            if end == f0_hz.len() {
                self.wave.extend(self.stream.finish());
            }
        }
        // The columns c with 5c + 9 in the wave: floor((len - 5) / 5) of them.
        let ready = if self.pushed == f0_hz.len() { total } else { self.wave.len().saturating_sub(5) / 5 };
        if ready > self.cols {
            let a = self.cols.max(2) - 2;
            let local = stft.transform(&self.wave[5 * a..]);
            for r in 0..har.len() / total {
                for g in self.cols..ready {
                    har[r * total + g] = local.data[r * local.t + g - a];
                }
            }
            self.cols = ready;
        }
    }
}

/// Pipeline::begin's steps up to the harmonic source's spectrum (the crate's pipeline module), for sentence i of the last text, with the
/// source seeded `seed + i` as speak_into seeds it. The generator is not run; its inputs are kept for the getters below. The previous
/// sentence is dropped first (as zt_gpu_front does), so a failed call leaves nothing for zt_gpu_source or zt_gpu_moments to use.
fn gpu_sentence(i: usize, seed: u32) -> Result<(), String> {
    with(|st| {
        st.prepared = Prepared::default();
        let Some(gpu) = st.gpu.as_ref() else {
            return Err("no GPU front loaded: call zt_load_front first".to_string());
        };
        let (ids, voice) = sentence_inputs(gpu, &st.sentences, i)?;
        let planned = plan_sentence(gpu, ids, voice)?;
        let n = planned.frames.t;
        let student = gpu.student.as_ref().ok_or("the student is not loaded: call zt_load_student first")?;
        let latents = prepass(student, Backend::Portable, &planned.frames, &planned.plan);
        let (moments, channels) = site_moments(gpu, &planned.coords, ids, &latent_statistics(&latents))?;
        let har = source_spectrum(gpu, seed.wrapping_add(i as u32), &planned.plan.f0_hz);
        st.prepared = Prepared {
            frames: n,
            start_frames: planned.start_frames,
            x: latents.data,
            har: har.data,
            coords: planned.coords,
            moments,
            channels,
            ..Prepared::default()
        };
        Ok(())
    })
}

/// zt_gpu_front: stages 1 to 3 of gpu_sentence for sentence i of the last text, with the source seeded `seed + i`. What the pre-pass,
/// zt_gpu_source and zt_gpu_moments need is kept (see Prepared). The previous sentence is dropped first, so a failure leaves nothing
/// for zt_gpu_source or zt_gpu_moments to use.
fn gpu_front(i: usize, seed: u32) -> Result<(), String> {
    with(|st| {
        st.prepared = Prepared::default();
        let Some(gpu) = st.gpu.as_ref() else {
            return Err("no GPU front loaded: call zt_load_front first".to_string());
        };
        let (ids, voice) = sentence_inputs(gpu, &st.sentences, i)?;
        let planned = plan_sentence(gpu, ids, voice)?;
        st.prepared = Prepared {
            stage: Stage::Planned,
            frames: planned.frames.t,
            start_frames: planned.start_frames,
            coords: planned.coords,
            feats: planned.frames.data,
            f0: planned.plan.scaled_f0,
            energy: planned.plan.energy,
            f0_hz: planned.plan.f0_hz,
            source_seed: seed.wrapping_add(i as u32),
            ids: ids.to_vec(),
            token_ids: ids.iter().map(|&t| t as u32).collect(),
            counts: planned.counts.iter().map(|&c| c as u32).collect(),
            ..Prepared::default()
        };
        Ok(())
    })
}

/// zt_gpu_prepare: sentence i of the last text, for the GPU's text encoder and frame stage: its token ids, its coordinates and its source
/// seed (seed + i). The previous sentence is dropped first, so a failure leaves nothing prepared.
fn gpu_prepare(i: usize, seed: u32) -> Result<(), String> {
    with(|st| {
        st.prepared = Prepared::default();
        let Some(gpu) = st.gpu.as_ref() else {
            return Err("no GPU front loaded: call zt_load_front first".to_string());
        };
        let (ids, voice) = sentence_inputs(gpu, &st.sentences, i)?;
        st.prepared = Prepared {
            stage: Stage::Prepared,
            coords: gpu.voices.coords(voice),
            source_seed: seed.wrapping_add(i as u32),
            ids: ids.to_vec(),
            token_ids: ids.iter().map(|&t| t as u32).collect(),
            ..Prepared::default()
        };
        Ok(())
    })
}

/// zt_gpu_counts' durations must be finite, not negative and below MAX_DURATION: a NaN or an infinity would give wrong counts, and a
/// larger duration would not fit the counts (bresenham rounds through i64 and takes usize).
fn check_durations(dur: &[f32]) -> Result<(), String> {
    for (i, &d) in dur.iter().enumerate() {
        if d.is_nan() {
            return Err(format!("duration {i} is NaN"));
        }
        if d.is_infinite() {
            return Err(format!("duration {i} is infinite"));
        }
        if d < 0.0 {
            return Err(format!("duration {i} is negative ({d})"));
        }
        if d >= MAX_DURATION {
            return Err(format!("duration {i} is {d}: the bound is {MAX_DURATION} 40 Hz units"));
        }
    }
    Ok(())
}

/// The frames of the counts (2 per count, as Student::frames totals them), checked: the sum must not overflow usize (the counts' type),
/// must fit zt_gpu_frames' u32, and must be at least one frame.
fn frame_total(counts40: &[usize]) -> Result<usize, String> {
    let total = counts40.iter().try_fold(0usize, |sum, &c| c.checked_mul(2).and_then(|f| sum.checked_add(f)));
    let Some(frames) = total else {
        return Err("the counts' frames overflow usize: the durations are too long".to_string());
    };
    if frames == 0 {
        return Err("the durations make no frames".to_string());
    }
    if u32::try_from(frames).is_err() {
        return Err(format!("the counts make {frames} frames: zt_gpu_frames returns u32"));
    }
    Ok(frames)
}

/// zt_gpu_plan's raw F0 and energy must be finite: the first value that is not is reported with its index.
fn check_finite(values: &[f32], name: &str) -> Result<(), String> {
    match values.iter().position(|v| !v.is_finite()) {
        None => Ok(()),
        Some(t) if values[t].is_nan() => Err(format!("{name} {t} is NaN")),
        Some(t) => Err(format!("{name} {t} is infinite")),
    }
}

/// zt_gpu_counts: the 40 Hz counts of the sentence prepared, from its durations (one per token, softplus applied), by bresenham as
/// plan_sentence does. The frames are 2 per count, and the start token's count gives the trim. A failure leaves nothing prepared.
fn gpu_counts(dur: &[f32]) -> Result<(), String> {
    with(|st| {
        let mut p = std::mem::take(&mut st.prepared);
        if p.stage != Stage::Prepared {
            return Err("no sentence prepared by zt_gpu_prepare: call zt_gpu_prepare first".to_string());
        }
        if dur.len() != p.ids.len() {
            return Err(format!("{} durations for the {} tokens of the sentence prepared", dur.len(), p.ids.len()));
        }
        check_durations(dur)?;
        let counts = bresenham(dur, &p.ids);
        p.frames = frame_total(&counts)?;
        p.start_frames = counts[0];
        p.counts = counts.iter().map(|&c| c as u32).collect();
        p.stage = Stage::Counted;
        st.prepared = p;
        Ok(())
    })
}

/// zt_gpu_plan: the plan of the sentence counted, from its raw scaled F0 and raw energy (Student::frames' outputs, computed on the GPU;
/// zt_gpu_frames values each), by plan_frames as plan_sentence does it. A failure leaves nothing prepared.
fn gpu_plan(f0: &[f32], energy: &[f32]) -> Result<(), String> {
    with(|st| {
        let mut p = std::mem::take(&mut st.prepared);
        let Some(gpu) = st.gpu.as_ref() else {
            return Err("no GPU front loaded: call zt_load_front first".to_string());
        };
        if p.stage != Stage::Counted {
            return Err("no counts: call zt_gpu_prepare and zt_gpu_counts first".to_string());
        }
        if f0.len() != p.frames || energy.len() != p.frames {
            return Err(format!("{} frames given, the counts make {} (zt_gpu_frames)", f0.len(), p.frames));
        }
        check_finite(f0, "raw F0")?;
        check_finite(energy, "raw energy")?;
        let counts40: Vec<usize> = p.counts.iter().map(|&c| c as usize).collect();
        let plan = plan_frames(&counts40, f0, energy, &p.ids, &gpu.vocab, p.frames)?;
        p.f0 = plan.scaled_f0;
        p.energy = plan.energy;
        p.f0_hz = plan.f0_hz;
        p.stage = Stage::Planned;
        st.prepared = p;
        Ok(())
    })
}

/// What zt_gpu_source and zt_gpu_moments need: the sentence's plan (zt_gpu_front's, or zt_gpu_plan's).
fn check_planned(p: &Prepared) -> Result<(), String> {
    if p.stage != Stage::Planned {
        return Err("no sentence prepared by zt_gpu_front or zt_gpu_plan: call zt_gpu_front first, or zt_gpu_prepare, zt_gpu_counts and zt_gpu_plan".to_string());
    }
    Ok(())
}

/// zt_gpu_source: the harmonic source's spectrum of the sentence prepared by zt_gpu_front (stage 6 of gpu_sentence), as far as
/// `frames` frames of it need (SourceRun::grow). The first call starts the run and allocates the spectrum, zeros of the whole size.
fn gpu_source(frames: u32) -> Result<(), String> {
    with(|st| {
        let Some(gpu) = st.gpu.as_ref() else {
            return Err("no GPU front loaded: call zt_load_front first".to_string());
        };
        let p = &mut st.prepared;
        check_planned(p)?;
        if p.run.is_none() {
            p.har = vec![0.0; 22 * (60 * p.frames + 1)];
        }
        let seed = p.source_seed;
        let run = p.run.get_or_insert_with(|| SourceRun { stream: gpu.source.stream(seed), wave: Vec::new(), pushed: 0, cols: 0 });
        run.grow(&p.f0_hz, &gpu.stft, &mut p.har, 60 * (frames as usize).min(p.frames) + 1);
        Ok(())
    })
}

/// zt_gpu_moments: the moments of the sentence prepared by zt_gpu_front, from the latents' statistics (stage 5 of gpu_sentence).
fn gpu_moments(stats: &[f32]) -> Result<(), String> {
    with(|st| {
        let Some(gpu) = st.gpu.as_ref() else {
            return Err("no GPU front loaded: call zt_load_front first".to_string());
        };
        let p = &mut st.prepared;
        check_planned(p)?;
        (p.moments, p.channels) = site_moments(gpu, &p.coords, &p.ids, stats)?;
        Ok(())
    })
}

/// Prepares sentence i of the last text for the generator (see gpu_sentence). Returns 0, or -1 (zt_error_*).
#[no_mangle]
pub extern "C" fn zt_gpu_sentence(i: u32, seed: u32) -> i32 {
    outcome(gpu_sentence(i as usize, seed))
}

/// Stages 1 to 3 of zt_gpu_sentence for sentence i of the last text, for the GPU pre-pass (js/dist/webgpu): zt_gpu_front_*_ptr give its
/// inputs, zt_gpu_source and zt_gpu_moments the rest (see the module comment). Returns 0, or -1 (zt_error_*).
#[no_mangle]
pub extern "C" fn zt_gpu_front(i: u32, seed: u32) -> i32 {
    outcome(gpu_front(i as usize, seed))
}

/// Prepares sentence i of the last text for the GPU's text encoder and frame stage (zt_gpu_counts, zt_gpu_plan). Returns 0, or -1
/// (zt_error_*).
#[no_mangle]
pub extern "C" fn zt_gpu_prepare(i: u32, seed: u32) -> i32 {
    outcome(gpu_prepare(i as usize, seed))
}

/// The 40 Hz counts of the sentence prepared, from its `n` durations at `dur_ptr` (softplus applied, one per token). Returns 0, or -1
/// (zt_error_*).
///
/// # Safety
/// `dur_ptr` must address `n` float32 values (zt_alloc's blocks), or `n` must be 0.
#[no_mangle]
pub unsafe extern "C" fn zt_gpu_counts(dur_ptr: *const f32, n: usize) -> i32 {
    outcome(gpu_counts(floats(dur_ptr, n)))
}

/// The plan of the sentence counted, from its raw scaled F0 and raw energy at `f0_ptr` and `energy_ptr` (`count` float32 values each, the
/// frames: zt_gpu_frames). Returns 0, or -1 (zt_error_*).
///
/// # Safety
/// `f0_ptr` and `energy_ptr` must address `count` float32 values each (zt_alloc's blocks), or `count` must be 0.
#[no_mangle]
pub unsafe extern "C" fn zt_gpu_plan(f0_ptr: *const f32, energy_ptr: *const f32, count: usize) -> i32 {
    outcome(gpu_plan(floats(f0_ptr, count), floats(energy_ptr, count)))
}

/// The harmonic source's spectrum for the sentence of zt_gpu_front (zt_gpu_har_*), as far as `frames` frames of the sentence need it:
/// columns [0, 60 min(frames, F) + 1) are final on return. Calls with growing frames extend the work; frames >= F is the whole spectrum.
/// Returns 0, or -1 (zt_error_*).
#[no_mangle]
pub extern "C" fn zt_gpu_source(frames: u32) -> i32 {
    outcome(gpu_source(frames))
}

/// The moments of the sentence of zt_gpu_front from the 1024 latent statistics at `stats_ptr` (latent_statistics' layout: the 512 channel
/// means, then the 512 values ln(sqrt(var) + 1e-4)). Returns 0, or -1 (zt_error_*).
///
/// # Safety
/// `stats_ptr` must address `stats_len` float32 values (zt_alloc's blocks).
#[no_mangle]
pub unsafe extern "C" fn zt_gpu_moments(stats_ptr: *const f32, stats_len: usize) -> i32 {
    outcome(if stats_len == STATISTICS {
        gpu_moments(floats(stats_ptr, stats_len))
    } else {
        Err(format!("{stats_len} latent statistics: {STATISTICS} are needed (512 channel means, then 512 log standard deviations)"))
    })
}

/// The soft limiter (zero_talk::post::soft_limit) in place on `count` float32 samples at `ptr`.
///
/// # Safety
/// `ptr` must address `count` float32 values from zt_alloc(4 * count), or `count` must be 0.
#[no_mangle]
pub unsafe extern "C" fn zt_post_limit(ptr: *mut f32, count: usize) {
    if count > 0 {
        soft_limit(std::slice::from_raw_parts_mut(ptr, count));
    }
}

/// Encodes `count` float32 samples at `ptr`, already limited, as zt_speak's last step does: the bytes at zt_out_ptr, the duration at
/// zt_duration. Returns 0, or -1 (zt_error_*).
///
/// # Safety
/// `ptr` must address `count` float32 values (zt_alloc's blocks), or `count` must be 0.
#[no_mangle]
pub unsafe extern "C" fn zt_encode(ptr: *const f32, count: usize, format: u32) -> i32 {
    let audio = floats(ptr, count);
    outcome(check_format(format).and_then(|()| with(|st| set_audio(st, audio, format))))
}

#[no_mangle]
pub extern "C" fn zt_out_ptr() -> *const u8 {
    with(|st| st.out.as_ptr())
}

#[no_mangle]
pub extern "C" fn zt_out_len() -> usize {
    with(|st| st.out.len())
}

#[no_mangle]
pub extern "C" fn zt_duration() -> f64 {
    with(|st| st.duration)
}

#[no_mangle]
pub extern "C" fn zt_info_ptr() -> *const u8 {
    with(|st| st.info.as_ptr())
}

#[no_mangle]
pub extern "C" fn zt_info_len() -> usize {
    with(|st| st.info.len())
}

#[no_mangle]
pub extern "C" fn zt_error_ptr() -> *const u8 {
    with(|st| st.error.as_ptr())
}

#[no_mangle]
pub extern "C" fn zt_error_len() -> usize {
    with(|st| st.error.len())
}

// The getters of the sentence last prepared. Their pointers are into this module's memory and stay valid until the call that replaces
// their array: zt_gpu_begin, zt_gpu_prepare, zt_gpu_counts, zt_gpu_plan, zt_gpu_front or zt_gpu_sentence. zt_gpu_source allocates
// the spectrum on its first call and fills it in place after that; zt_gpu_moments replaces the moments.

/// The frames (80 Hz) of the sentence.
#[no_mangle]
pub extern "C" fn zt_gpu_frames() -> u32 {
    with(|st| st.prepared.frames as u32)
}

/// The start token's length in 40 Hz frames.
#[no_mangle]
pub extern "C" fn zt_gpu_start_frames() -> u32 {
    with(|st| st.prepared.start_frames as u32)
}

/// The samples to drop from the start of the sentence (zero_talk::speak::trimmed_samples).
#[no_mangle]
pub extern "C" fn zt_gpu_trim() -> u32 {
    with(|st| trimmed_samples(st.prepared.start_frames) as u32)
}

#[no_mangle]
pub extern "C" fn zt_gpu_x_ptr() -> *const f32 {
    with(|st| st.prepared.x.as_ptr())
}

#[no_mangle]
pub extern "C" fn zt_gpu_x_len() -> u32 {
    with(|st| st.prepared.x.len() as u32)
}

#[no_mangle]
pub extern "C" fn zt_gpu_har_ptr() -> *const f32 {
    with(|st| st.prepared.har.as_ptr())
}

#[no_mangle]
pub extern "C" fn zt_gpu_har_len() -> u32 {
    with(|st| st.prepared.har.len() as u32)
}

/// The frame features of the sentence of zt_gpu_front, [384][frames] row-major (channel-major), for the GPU pre-pass.
#[no_mangle]
pub extern "C" fn zt_gpu_front_frames_ptr() -> *const f32 {
    with(|st| st.prepared.feats.as_ptr())
}

/// The length of the frame features: 384 * frames after zt_gpu_front, 0 after zt_gpu_plan (the features stay on the GPU).
#[no_mangle]
pub extern "C" fn zt_gpu_front_frames_len() -> u32 {
    with(|st| st.prepared.feats.len() as u32)
}

/// The scaled F0 of the sentence of zt_gpu_front (the acoustic blocks' F0 input), [frames].
#[no_mangle]
pub extern "C" fn zt_gpu_front_f0_ptr() -> *const f32 {
    with(|st| st.prepared.f0.as_ptr())
}

/// The energy of the sentence of zt_gpu_front (the acoustic blocks' energy input), [frames].
#[no_mangle]
pub extern "C" fn zt_gpu_front_energy_ptr() -> *const f32 {
    with(|st| st.prepared.energy.as_ptr())
}

/// The token ids of the sentence prepared (zt_gpu_prepare, or zt_gpu_front), little-endian u32: zt_gpu_ids_len of them.
#[no_mangle]
pub extern "C" fn zt_gpu_ids_ptr() -> *const u32 {
    with(|st| st.prepared.token_ids.as_ptr())
}

#[no_mangle]
pub extern "C" fn zt_gpu_ids_len() -> u32 {
    with(|st| st.prepared.token_ids.len() as u32)
}

/// The 40 Hz counts of the sentence (u32, one per token: zt_gpu_ids_len of them), after zt_gpu_counts or zt_gpu_front.
#[no_mangle]
pub extern "C" fn zt_gpu_counts_ptr() -> *const u32 {
    with(|st| st.prepared.counts.as_ptr())
}

#[no_mangle]
pub extern "C" fn zt_gpu_coords_ptr() -> *const f32 {
    with(|st| st.prepared.coords.as_ptr())
}

#[no_mangle]
pub extern "C" fn zt_gpu_moments_ptr() -> *const f32 {
    with(|st| st.prepared.moments.as_ptr())
}

#[no_mangle]
pub extern "C" fn zt_gpu_moments_len() -> u32 {
    with(|st| st.prepared.moments.len() as u32)
}

#[no_mangle]
pub extern "C" fn zt_gpu_site_channels_ptr() -> *const u32 {
    with(|st| st.prepared.channels.as_ptr())
}

#[no_mangle]
pub extern "C" fn zt_gpu_site_count() -> u32 {
    with(|st| st.gpu.as_ref().map_or(0, |g| g.sites.len() as u32))
}

#[no_mangle]
pub extern "C" fn zt_gpu_site_names_ptr() -> *const u8 {
    with(|st| st.gpu.as_ref().map_or("".as_ptr(), |g| g.names.as_ptr()))
}

#[no_mangle]
pub extern "C" fn zt_gpu_site_names_len() -> u32 {
    with(|st| st.gpu.as_ref().map_or(0, |g| g.names.len() as u32))
}
