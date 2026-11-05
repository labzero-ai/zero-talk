//! The `zero-talk` command: text to speech in the terminal. The binary calls [`main`]; the functions here are not part of the
//! library's interface, and code that uses the engine should call [`Voice`](crate::Voice) instead.
use std::cell::Cell;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::archive::io_message;
use crate::audio::Audio;
use crate::kernels::Backend;
use crate::opus::{self, OggStream, STREAM_PAGE_BYTES};
use crate::pipeline::SAMPLE_RATE;
use crate::play;
use crate::source::SEED;
use crate::voice::{Sentences, Voice};
use crate::wav;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Why a command did not run: a mistake in the call (the help says how to call it), or anything else.
#[derive(Debug)]
pub(crate) enum Fail {
    Usage(String),
    Other(String),
}

impl From<crate::error::Error> for Fail {
    fn from(e: crate::error::Error) -> Self {
        Fail::Other(e.to_string())
    }
}

/// Runs the command with the process's arguments and returns the exit status: 0 when done, 2 when it could not run.
pub fn main() -> i32 {
    match std::env::args_os().skip(1).map(|a| a.into_string()).collect::<Result<Vec<String>, _>>() {
        Ok(args) => run(args),
        Err(bad) => {
            eprintln!("zero-talk: an argument is not valid UTF-8 ({bad:?}); text and file names must be UTF-8.");
            2
        }
    }
}

/// Runs the command with these arguments (without the program name), for hosts such as the Python package's `zero-talk`
/// script, whose own process arguments start with the interpreter. Returns the exit status like [`main`].
pub fn run(args: Vec<String>) -> i32 {
    if args.first().is_none() {
        eprint!("{}", help());
        return 2;
    }
    if args.first().map(String::as_str) == Some("-V") || args.first().map(String::as_str) == Some("--version") {
        println!("zero-talk {VERSION}");
        if cfg!(feature = "embed") {
            println!("model: Zero Talk 11M");
        }
        return 0;
    }
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", help());
        return 0;
    }
    match command(args) {
        Ok(()) => 0,
        Err(Fail::Usage(msg)) => {
            eprintln!("zero-talk: {msg}\nRun 'zero-talk --help' for usage.");
            2
        }
        Err(Fail::Other(msg)) => {
            eprintln!("zero-talk: {msg}");
            2
        }
    }
}

fn help() -> String {
    let backends = Backend::available();
    let backend_line = if backends.len() > 1 {
        let names: Vec<String> = backends.iter().enumerate().map(|(i, b)| format!("{}{}", b.name(), if i == 0 { " (default)" } else { "" })).collect();
        format!("  --backend NAME    matrix-multiply library: {}\n", names.join(", "))
    } else {
        String::new()
    };
    let files = if cfg!(feature = "embed") { "" } else { " --model MODEL --frontend FILE" };
    format!(
        "\
zero-talk {VERSION}: reads English text aloud with Zero Talk 11M

Usage: zero-talk (TEXT | --file FILE) [-o OUT] [options]

  TEXT              the text to read (quote it); or --file FILE to read it from a file
  -o, --out FILE    the audio file to write: .ogg is Ogg Opus, .wav is 16-bit WAV (24 kHz, mono), and both carry an
                    AI-generated marker in their metadata. Without -o the audio is played as it is made (with ffplay,
                    paplay or aplay; else when it is made)
  --stream          write the audio to standard output as it is made, for a pipe; --format NAME: ogg (default), wav or pcm;
                    raw pcm has no metadata, so it cannot carry the AI-generated marker
  --bitrate KBPS    Ogg Opus bit rate, {lo} to {hi} kbit/s (default {kbps})
  --seed N          random seed (default {seed})
  --model FILE      the model file ({built_in})
  --frontend FILE   the text front-end file ({built_in})
{backend_line}  -h, --help        show this help
  -V, --version     show the version

Words in other scripts than Latin are left out, with a warning.
Text with a ! needs single quotes in zsh and bash: zero-talk 'Hello there!'

To play:  zero-talk \"Thanks for coming. See you soon.\"{files}
To write: zero-talk \"Thanks for coming. See you soon.\"{files} -o thanks.ogg
To pipe:  zero-talk \"Thanks for coming. See you soon.\"{files} --stream | ffplay -nodisp -autoexit -
Exit status: 0 done, 2 could not run.
",
        lo = opus::KBPS.start(),
        hi = opus::KBPS.end(),
        kbps = opus::DEFAULT_KBPS,
        seed = SEED,
        built_in = if cfg!(feature = "embed") { "default: the file built into this program" } else { "required: this build has none inside" },
    )
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Ogg,
    Wav,
    Pcm,
}

struct Args {
    text: Option<String>,
    file: Option<PathBuf>,
    model: Option<PathBuf>,
    frontend: Option<PathBuf>,
    out: Option<PathBuf>, // none: play the audio, or write it to standard output with `stream`
    stream: bool,
    format: Option<Format>,
    bitrate: Option<u32>,
    seed: u32,
    backend: Backend,
}

fn split_flag(arg: &str) -> (String, Option<String>) {
    match arg.split_once('=') {
        Some((a, v)) if a.starts_with("--") => (a.to_string(), Some(v.to_string())),
        _ => (arg.to_string(), None),
    }
}

fn value(flag: &str, inline: Option<String>, args: &mut impl Iterator<Item = String>) -> Result<String, Fail> {
    inline.or_else(|| args.next()).ok_or_else(|| Fail::Usage(format!("{flag} needs a value")))
}

fn parse(args: Vec<String>) -> Result<Args, Fail> {
    let mut a = Args {
        text: None,
        file: None,
        model: None,
        frontend: None,
        out: None,
        stream: false,
        format: None,
        bitrate: None,
        seed: SEED,
        backend: Backend::Portable,
    };
    let mut rest = args.into_iter();
    while let Some(arg) = rest.next() {
        let (flag, inline) = split_flag(&arg);
        match flag.as_str() {
            "--file" => a.file = Some(PathBuf::from(value(&flag, inline, &mut rest)?)),
            "--model" => a.model = Some(PathBuf::from(value(&flag, inline, &mut rest)?)),
            "--frontend" => a.frontend = Some(PathBuf::from(value(&flag, inline, &mut rest)?)),
            "-o" | "--out" => a.out = Some(PathBuf::from(value(&flag, inline, &mut rest)?)),
            "--stream" => a.stream = true,
            "--format" => {
                a.format = Some(match value(&flag, inline, &mut rest)?.as_str() {
                    "ogg" => Format::Ogg,
                    "wav" => Format::Wav,
                    "pcm" => Format::Pcm,
                    other => return Err(Fail::Usage(format!("--format needs ogg, wav or pcm, not '{other}'"))),
                });
            }
            "--bitrate" => {
                let v = value(&flag, inline, &mut rest)?;
                a.bitrate = Some(v.parse().ok().filter(|k| opus::KBPS.contains(k)).ok_or_else(|| {
                    Fail::Usage(format!("--bitrate needs a whole number of kbit/s from {} to {}, not '{v}'", opus::KBPS.start(), opus::KBPS.end()))
                })?);
            }
            "--seed" => {
                let v = value(&flag, inline, &mut rest)?;
                a.seed = v.parse().map_err(|_| Fail::Usage(format!("--seed needs a whole number, not '{v}'")))?;
            }
            "--backend" => {
                let name = value(&flag, inline, &mut rest)?;
                a.backend = Backend::parse(&name).ok_or_else(|| {
                    let names: Vec<&str> = Backend::available().iter().map(|b| b.name()).collect();
                    Fail::Usage(format!("backend '{name}' is not available in this build (this build has: {})", names.join(", ")))
                })?;
            }
            _ if flag.starts_with('-') && flag.len() > 1 => return Err(Fail::Usage(format!("unknown option '{flag}' for 'zero-talk'"))),
            _ if a.text.is_none() => a.text = Some(arg),
            _ => return Err(Fail::Usage(format!("unexpected argument '{arg}': the text is one argument (quote it)"))),
        }
    }
    if a.text.is_some() == a.file.is_some() {
        return Err(Fail::Usage("give the text as an argument or with --file FILE (not both, not neither)".into()));
    }
    if a.out.as_deref() == Some(Path::new("-")) {
        (a.out, a.stream) = (None, true); // the Unix name for standard output
    }
    if a.stream && a.out.is_some() {
        return Err(Fail::Usage("--stream writes to standard output: leave out -o FILE".into()));
    }
    if a.format.is_some() && !a.stream {
        return Err(Fail::Usage("--format goes with --stream (a file's extension says its format)".into()));
    }
    Ok(a)
}

fn command(args: Vec<String>) -> Result<(), Fail> {
    let a = parse(args)?;
    let pipe = a.stream.then(|| a.format.unwrap_or(Format::Ogg));
    if pipe.is_some() && std::io::stdout().is_terminal() {
        return Err(Fail::Other(
            "standard output is a terminal: pipe the audio to a player (zero-talk TEXT --stream | ffplay -nodisp -autoexit -) or write a file with -o FILE".into(),
        ));
    }
    let ogg = match &a.out {
        None => pipe == Some(Format::Ogg),
        Some(out) => match out.extension().map(|e| e.to_string_lossy().to_lowercase()).as_deref() {
            Some("ogg") => true,
            Some("wav") => false,
            _ => return Err(Fail::Usage(format!("{}: the output file must end in .ogg (Ogg Opus) or .wav (WAV); MP3 is not built", out.display()))),
        },
    };
    if !ogg && a.bitrate.is_some() {
        return Err(Fail::Usage("--bitrate is for Ogg Opus output; WAV, PCM and playing have no bit rate".into()));
    }
    if !cfg!(feature = "embed") && (a.model.is_none() || a.frontend.is_none()) {
        let missing: Vec<&str> = [(a.model.is_none(), "--model MODEL"), (a.frontend.is_none(), "--frontend FILE")].iter().filter(|m| m.0).map(|m| m.1).collect();
        return Err(Fail::Usage(format!("this build has no data files inside: add {}", missing.join(" and "))));
    }
    let text = match (&a.text, &a.file) {
        (Some(t), _) => t.clone(),
        (_, Some(f)) => std::fs::read_to_string(f).map_err(|e| Fail::Other(format!("{}: {}", f.display(), io_message(&e))))?,
        _ => unreachable!("parse gives exactly one of the text and the file"),
    };
    let t0 = Instant::now();
    let mut voice = Voice::open(a.model.as_deref(), a.frontend.as_deref())?;
    voice.set_backend(a.backend);
    voice.set_seed(a.seed);
    let (sentences, warnings) = voice.prepare(&text)?;
    for w in &warnings {
        eprintln!("zero-talk: warning: {w}");
    }
    let run = Run { voice: &voice, a: &a, sentences, t0, t1: Instant::now(), c1: cpu_seconds() };
    if let Some(format) = pipe {
        let kbps = a.bitrate.unwrap_or(opus::DEFAULT_KBPS);
        let what = match format {
            Format::Ogg => format!("Ogg Opus, {kbps} kbit/s"),
            Format::Wav => "WAV".to_string(),
            Format::Pcm => "16-bit PCM".to_string(),
        };
        let mut out = std::io::stdout().lock();
        return stream(&mut out, format, 0, &format!("streamed {what} to standard output"), run);
    }
    if a.out.is_none() {
        if let Some(mut player) = play::Player::open(SAMPLE_RATE)? {
            let format = if player.wav { Format::Wav } else { Format::Pcm };
            let streamed = stream(player.input(), format, PLAY_LEAD, "played", run);
            return streamed.and(player.finish());
        }
    }
    let samples = render(&run)?;
    let (t1, c1) = (run.t1, run.c1);
    let (wall, cpu) = (t1.elapsed().as_secs_f64(), cpu_seconds() - c1);
    let seconds = samples.len() as f64 / SAMPLE_RATE as f64;
    let Some(out) = &a.out else {
        println!("zero-talk: playing ({seconds:.2} s of audio, {} sentence(s)) ({cpu:.2} s of CPU).", run.sentences.len());
        return play::play(&samples, SAMPLE_RATE);
    };
    let audio = Audio::new(samples, Vec::new());
    let (bytes, detail) = if ogg {
        let bytes = audio.to_ogg()?;
        let detail = format!(", Ogg Opus, {:.1} kbit/s on average, 24 kHz mono", bytes.len() as f64 * 8.0 / seconds / 1000.0);
        (bytes, detail)
    } else {
        (audio.to_wav()?, ", 24 kHz mono 16-bit".to_string())
    };
    std::fs::write(out, bytes).map_err(|e| Fail::Other(format!("{}: {}", out.display(), io_message(&e))))?;
    println!(
        "zero-talk: wrote {} ({seconds:.2} s of audio, {} sentence(s){detail}) in {:.2} s, {:.1}x faster than real time ({} backend; {cpu:.2} s of CPU, before the encoding).",
        out.display(),
        run.sentences.len(),
        t0.elapsed().as_secs_f64(),
        seconds / wall,
        a.backend.name()
    );
    Ok(())
}

/// The audio is held back this long before the first write when it is played: a short text can run out of audio while the player
/// starts, so the first 0.3 s are given to the player together.
const PLAY_LEAD: usize = SAMPLE_RATE as usize * 3 / 10;

/// What a run has to hand to the output: the voice, the arguments, the sentences and the times the run started.
struct Run<'a> {
    voice: &'a Voice,
    a: &'a Args,
    sentences: Sentences,
    t0: Instant,
    t1: Instant,
    c1: f64,
}

/// Renders all the audio of a run.
fn render(run: &Run) -> Result<Vec<f32>, Fail> {
    let mut samples = Vec::new();
    for piece in run.voice.joiner(run.sentences.clone()) {
        samples.extend_from_slice(&piece.map_err(Fail::Other)?);
    }
    Ok(samples)
}

/// The reader's end of the audio. It notes when the reader has closed it, so that the program can stop without a message.
struct Pipe<'a> {
    out: &'a mut dyn Write,
    closed: &'a Cell<bool>,
}

impl Write for Pipe<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.out.write(bytes).inspect_err(|e| self.closed.set(self.closed.get() || e.kind() == std::io::ErrorKind::BrokenPipe))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.out.flush().inspect_err(|e| self.closed.set(self.closed.get() || e.kind() == std::io::ErrorKind::BrokenPipe))
    }
}

enum Sink<'a> {
    Ogg(OggStream<Pipe<'a>>),
    Raw(Pipe<'a>),
}

impl Sink<'_> {
    fn push(&mut self, audio: &[f32]) -> Result<(), String> {
        match self {
            Sink::Ogg(ogg) => ogg.push(audio),
            Sink::Raw(pipe) => pipe.write_all(&wav::pcm_bytes(audio)).and_then(|()| pipe.flush()).map_err(|e| format!("write failed: {e}")),
        }
    }

    fn finish(self) -> Result<(), String> {
        match self {
            Sink::Ogg(ogg) => ogg.finish().map(|_| ()),
            Sink::Raw(_) => Ok(()),
        }
    }
}

/// The audio goes to `out` as it is made, in `format`; the first `lead` samples are held back until there are that many. A reader
/// that closes its end ends the run quietly (a player that failed is told by its exit status, which the caller checks).
fn stream(out: &mut dyn Write, format: Format, lead: usize, label: &str, run: Run) -> Result<(), Fail> {
    let closed = Cell::new(false);
    let (mut first, mut samples, mut held, mut released) = (None, 0usize, Vec::new(), false);
    let kbps = run.a.bitrate.unwrap_or(opus::DEFAULT_KBPS);
    let result = (|| -> Result<(), String> {
        let mut pipe = Pipe { out, closed: &closed };
        let mut sink = match format {
            Format::Ogg => Sink::Ogg(OggStream::new(pipe, SAMPLE_RATE, kbps, Some(STREAM_PAGE_BYTES))?),
            Format::Wav => {
                pipe.write_all(&wav::header(SAMPLE_RATE, None)).and_then(|()| pipe.flush()).map_err(|e| format!("write failed: {e}"))?;
                Sink::Raw(pipe)
            }
            Format::Pcm => Sink::Raw(pipe),
        };
        for piece in run.voice.joiner(run.sentences.clone()) {
            let piece = piece.map_err(|e| e.to_string())?;
            samples += piece.len();
            if released {
                sink.push(&piece)?;
                continue;
            }
            held.extend_from_slice(&piece);
            if held.len() < lead {
                continue;
            }
            (released, first) = (true, Some(Instant::now()));
            sink.push(&std::mem::take(&mut held))?;
        }
        if !held.is_empty() {
            first.get_or_insert_with(Instant::now);
            sink.push(&held)?; // a text shorter than the hold
        }
        sink.finish()
    })();
    if let Err(e) = result {
        return if closed.get() { Ok(()) } else { Err(Fail::Other(e)) };
    }
    let seconds = samples as f64 / SAMPLE_RATE as f64;
    let (wall, cpu) = (run.t1.elapsed().as_secs_f64(), cpu_seconds() - run.c1);
    eprintln!(
        "zero-talk: {label} ({seconds:.2} s of audio, {} sentence(s)) in {:.2} s, first audio {:.3} s after the start, {:.1}x faster than real time ({} backend; {cpu:.2} s of CPU).",
        run.sentences.len(),
        run.t0.elapsed().as_secs_f64(),
        first.map_or(f64::NAN, |f: Instant| f.duration_since(run.t0).as_secs_f64()),
        seconds / wall,
        run.a.backend.name()
    );
    Ok(())
}

/// The CPU time of this process in seconds (wall time since the first call where the clock is not available).
fn cpu_seconds() -> f64 {
    #[cfg(all(any(target_os = "macos", target_os = "linux"), target_pointer_width = "64"))]
    {
        #[repr(C)]
        struct Timespec {
            sec: i64,
            nsec: i64,
        }
        extern "C" {
            fn clock_gettime(clock: i32, tp: *mut Timespec) -> i32;
        }
        const PROCESS_CPUTIME: i32 = if cfg!(target_os = "macos") { 12 } else { 2 };
        let mut ts = Timespec { sec: 0, nsec: 0 };
        if unsafe { clock_gettime(PROCESS_CPUTIME, &mut ts) } == 0 {
            return ts.sec as f64 + ts.nsec as f64 * 1e-9;
        }
    }
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}
