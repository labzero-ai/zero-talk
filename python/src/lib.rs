//! The extension module `zero_talk._native`: the engine of the `zero-talk` crate, wrapped for Python.
//!
//! The package `zero_talk` (in `python/zero_talk`) re-exports these names and holds the type stubs. Every engine call that takes
//! time (loading, speaking, making a chunk, encoding, writing a file) runs with the GIL released.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use pyo3::exceptions::{PyFileNotFoundError, PyValueError, PyOSError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyMemoryView};

/// A loaded voice, shared by the Python objects that use it. The engine keeps a word cache in its text front end, so every use
/// takes the lock; the lock is held only while the engine works, never while Python code runs.
type Shared = Arc<Mutex<zero_talk::Voice>>;

/// Raises `ValueError` for an identifier that is not a voice, and names the voices that are.
fn check_voice_id(id: &str) -> PyResult<()> {
    let voices = zero_talk::voices();
    if voices.contains(&id) {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "no voice named '{id}'; the voices are: {}. zero_talk.voices() lists them.",
        voices.join(", ")
    )))
}

/// An engine error as a Python `ValueError`, with the engine's message (which says what to change).
fn engine_error(err: zero_talk::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Takes the lock of a voice. It fails only if an earlier use of the voice panicked, and then the voice must be loaded again.
fn lock(voice: &Shared) -> zero_talk::Result<MutexGuard<'_, zero_talk::Voice>> {
    voice.lock().map_err(|_| {
        zero_talk::Error::from(
            "the voice stopped after an internal error; load it again with zero_talk.load_voice()".to_string(),
        )
    })
}

/// The lowercase extension of a file name, if it has one.
fn extension(path: &Path) -> Option<String> {
    path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase)
}

/// The samples as a read-only, one-dimensional `memoryview` of format `f` (float32, native byte order). The memory is an immutable
/// copy that the memoryview keeps alive, so `numpy.asarray` and other buffer consumers read it without another copy.
fn float32_memoryview(py: Python<'_>, samples: &[f32]) -> PyResult<Py<PyAny>> {
    // SAFETY: an f32 has no padding and can be read as four bytes; the slice covers exactly the samples.
    let raw = unsafe { std::slice::from_raw_parts(samples.as_ptr().cast::<u8>(), std::mem::size_of_val(samples)) };
    let bytes = PyBytes::new(py, raw);
    let view = PyMemoryView::from(bytes.as_any())?.call_method1("cast", ("f",))?;
    Ok(view.unbind())
}

/// Loads a voice by its identifier. With `model_dir`, the model is read from that folder (`model.bin` and `frontend.bin`)
/// instead of the files built into the package.
#[pyfunction]
#[pyo3(signature = (id, model_dir=None))]
fn load_voice(py: Python<'_>, id: &str, model_dir: Option<PathBuf>) -> PyResult<PyVoice> {
    check_voice_id(id)?;
    let loaded = match model_dir {
        None => py.detach(|| zero_talk::Voice::load(id)),
        Some(dir) => {
            let (model, frontend) = (dir.join("model.bin"), dir.join("frontend.bin"));
            for file in [&model, &frontend] {
                if !file.is_file() {
                    return Err(PyFileNotFoundError::new_err(format!(
                        "{}: no such file; model_dir must hold model.bin and frontend.bin (zero_talk.model_info() gives their sizes and hashes)",
                        file.display()
                    )));
                }
            }
            py.detach(move || zero_talk::Voice::from_files(model, frontend))
        }
    }
    .map_err(engine_error)?;
    Ok(PyVoice { inner: Arc::new(Mutex::new(loaded)) })
}

/// The identifiers of the voices that `load_voice` accepts.
#[pyfunction]
fn voices() -> Vec<&'static str> {
    zero_talk::voices().to_vec()
}

/// The sizes and SHA-256 hashes of the model files, from the model crate: a dict with the voice `id` and `files`, keyed by file
/// name (`model.bin`, `frontend.bin`), each with `bytes` and `sha256`.
#[pyfunction]
fn model_info<'py>(py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyDict>> {
    check_voice_id(id)?;
    let model = PyDict::new(py);
    model.set_item("bytes", zero_talk_model_11m::MODEL_BYTES)?;
    model.set_item("sha256", zero_talk_model_11m::MODEL_SHA256)?;
    let frontend = PyDict::new(py);
    frontend.set_item("bytes", zero_talk_model_11m::FRONTEND_BYTES)?;
    frontend.set_item("sha256", zero_talk_model_11m::FRONTEND_SHA256)?;
    let files = PyDict::new(py);
    files.set_item("model.bin", model)?;
    files.set_item("frontend.bin", frontend)?;
    let info = PyDict::new(py);
    info.set_item("id", id)?;
    info.set_item("files", files)?;
    Ok(info)
}

/// A voice: a model loaded and ready to speak. Get one from `load_voice`.
#[pyclass(name = "Voice", module = "zero_talk", frozen)]
pub struct PyVoice {
    inner: Shared,
}

#[pymethods]
impl PyVoice {
    /// Speaks a text and returns the speech as an `Audio`.
    ///
    /// The text is English: plain sentences, with numbers, dates and prices written as digits read out in words. Words that cannot
    /// be read, and characters the model has no sound for, are left out and named in `Audio.warnings`. A text with nothing to say
    /// raises `ValueError`.
    fn speak(&self, py: Python<'_>, text: &str) -> PyResult<PyAudio> {
        let voice = Arc::clone(&self.inner);
        let text = text.to_owned();
        let audio = py
            .detach(move || {
                let voice = lock(&voice)?;
                voice.speak(&text)
            })
            .map_err(engine_error)?;
        PyAudio::new(py, audio)
    }

    /// Starts speaking a text piece by piece. Returns a `Stream`: iterate it to get `Chunk` objects, each made when it is reached.
    /// The chunks, concatenated, are the samples that `speak` gives for the same text. A text with nothing to say raises
    /// `ValueError` here, before the first chunk.
    fn stream(&self, py: Python<'_>, text: &str) -> PyResult<PyStream> {
        let voice = Arc::clone(&self.inner);
        let text = text.to_owned();
        let chunks = py.detach(move || Chunks::start(voice, &text)).map_err(engine_error)?;
        Ok(PyStream { chunks })
    }
}

/// Speech from `Voice.speak`: mono samples at 24 kHz, with the warnings about what was left out of the text.
#[pyclass(name = "Audio", module = "zero_talk", frozen)]
pub struct PyAudio {
    inner: zero_talk::Audio,
    samples: Py<PyAny>,
}

impl PyAudio {
    fn new(py: Python<'_>, inner: zero_talk::Audio) -> PyResult<Self> {
        let samples = float32_memoryview(py, inner.samples())?;
        Ok(PyAudio { inner, samples })
    }
}

#[pymethods]
impl PyAudio {
    /// The samples: a read-only `memoryview` of float32, mono, from -1 to 1. `numpy.asarray(audio.samples)` reads the same memory
    /// without a copy.
    #[getter]
    fn samples(&self, py: Python<'_>) -> Py<PyAny> {
        self.samples.clone_ref(py)
    }

    /// The sample rate in hertz: always 24000.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// The length of the speech in seconds.
    #[getter]
    fn seconds(&self) -> f64 {
        self.inner.seconds()
    }

    /// What was left out of the text, one message per problem. An empty list when everything was spoken.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings().to_vec()
    }

    /// Writes the speech to a file. The extension picks the format: `.wav` (16-bit PCM WAV) or `.ogg` (Ogg Opus, 24 kbit/s).
    /// Both carry an AI-generated marker in their metadata. Another extension raises `ValueError`; a file that cannot be written
    /// raises `OSError`.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let bytes = match extension(&path).as_deref() {
            Some("wav") => py.detach(|| self.inner.to_wav()),
            Some("ogg") => py.detach(|| self.inner.to_ogg()),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "{}: save() writes .wav (WAV) or .ogg (Ogg Opus); name the file with one of those extensions",
                    path.display()
                )))
            }
        }
        .map_err(engine_error)?;
        std::fs::write(&path, bytes).map_err(|e| PyOSError::new_err(format!("{}: {}", path.display(), e)))?;
        Ok(())
    }

    /// The speech as a complete WAV file, in memory: 16-bit PCM, mono, 24 kHz. Its metadata carries an AI-generated marker.
    fn to_wav<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.inner.to_wav()).map_err(engine_error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// The speech as a complete Ogg Opus file, in memory: 24 kbit/s, mono, 24 kHz. Its metadata carries an AI-generated marker.
    fn to_ogg<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.inner.to_ogg()).map_err(engine_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// A piece of the speech of a text, from `Stream`: mono samples at 24 kHz.
#[pyclass(name = "Chunk", module = "zero_talk", frozen)]
pub struct PyChunk {
    samples: Py<PyAny>,
}

impl PyChunk {
    fn new(py: Python<'_>, chunk: zero_talk::Chunk) -> PyResult<Self> {
        let samples = float32_memoryview(py, chunk.samples())?;
        Ok(PyChunk { samples })
    }
}

#[pymethods]
impl PyChunk {
    /// The samples of this piece: a read-only `memoryview` of float32, mono, from -1 to 1 (see `Audio.samples`).
    #[getter]
    fn samples(&self, py: Python<'_>) -> Py<PyAny> {
        self.samples.clone_ref(py)
    }
}

/// Wraps a value that is not `Send`: the engine's stream holds a boxed iterator. The only Python object that holds one, `PyStream`, is
/// `unsendable`, so Python code cannot use it from another thread, and `detach` runs its closure on the calling thread. The value
/// therefore never moves between threads.
struct SameThread<T>(T);

// SAFETY: see `SameThread`. The wrapped value is used only on the thread that created it.
unsafe impl<T> Send for SameThread<T> {}

/// The audio of a text as it is made: the chunks of `Voice.stream`, read one at a time.
///
/// The engine's stream borrows the voice. The voice is kept alive by the `Arc` in the same struct, and the stream is declared
/// first, so it is dropped first. Each use of the stream takes the voice's lock.
struct Chunks {
    stream: SameThread<zero_talk::Stream<'static>>,
    _voice: Shared,
}

impl Chunks {
    fn start(voice: Shared, text: &str) -> zero_talk::Result<Chunks> {
        let stream = {
            let guard = lock(&voice)?;
            let stream = guard.stream(text)?;
            // SAFETY: the stream borrows the voice that lives inside the `Arc` of `voice`. That `Arc` is stored in the returned struct,
            // so the voice stays where it is as long as the stream exists, and the stream is dropped before the `Arc` (fields drop in
            // declaration order). Every later use of the stream takes the same lock as the engine's other calls.
            unsafe { std::mem::transmute::<zero_talk::Stream<'_>, zero_talk::Stream<'static>>(stream) }
        };
        Ok(Chunks { stream: SameThread(stream), _voice: voice })
    }

    fn next_chunk(&mut self) -> zero_talk::Result<Option<zero_talk::Chunk>> {
        let _guard = lock(&self._voice)?;
        self.stream.0.next().transpose()
    }
}

/// The chunks of a text, one at a time, from `Voice.stream`. Iterate it in the thread that made it.
#[pyclass(name = "Stream", module = "zero_talk", unsendable)]
pub struct PyStream {
    chunks: Chunks,
}

#[pymethods]
impl PyStream {
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<PyChunk>> {
        let chunks = &mut slf.chunks;
        match py.detach(move || chunks.next_chunk()).map_err(engine_error)? {
            None => Ok(None),
            Some(chunk) => Ok(Some(PyChunk::new(py, chunk)?)),
        }
    }
}

/// Runs the `zero-talk` command with these arguments (without the program name) and returns its exit status.
#[pyfunction]
fn _run_cli(py: Python<'_>, args: Vec<String>) -> i32 {
    py.detach(|| zero_talk::cli::run(args))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load_voice, m)?)?;
    m.add_function(wrap_pyfunction!(voices, m)?)?;
    m.add_function(wrap_pyfunction!(model_info, m)?)?;
    m.add_function(wrap_pyfunction!(_run_cli, m)?)?;
    m.add_class::<PyVoice>()?;
    m.add_class::<PyAudio>()?;
    m.add_class::<PyChunk>()?;
    m.add_class::<PyStream>()?;
    Ok(())
}
