//! Plays audio through the system's own player: a temporary WAV file, and the first player of the platform that starts.
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

use crate::wav;

use crate::cli::Fail;

/// The players to try in order: the program and its arguments before the file name.
#[cfg(target_os = "macos")]
const PLAYERS: &[(&str, &[&str])] = &[("afplay", &[])];
#[cfg(all(unix, not(target_os = "macos")))]
const PLAYERS: &[(&str, &[&str])] = &[("paplay", &[]), ("pw-play", &[]), ("aplay", &["-q"]), ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "quiet"])];
// The file name goes in the environment, so that a path with a quote in it cannot break the script.
#[cfg(windows)]
const PLAYERS: &[(&str, &[&str])] = &[("powershell", &["-NoProfile", "-Command", "(New-Object System.Media.SoundPlayer $env:SPEAK_WAV).PlaySync()"])];
#[cfg(not(any(unix, windows)))]
const PLAYERS: &[(&str, &[&str])] = &[];

/// A WAV file in the temporary folder that is deleted when this goes out of scope.
struct TempWav(PathBuf);

impl TempWav {
    fn create(bytes: &[u8]) -> Result<Self, Fail> {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!("zero-talk-{}-{nanos}.wav", std::process::id()));
        let written = std::fs::OpenOptions::new().write(true).create_new(true).open(&path).and_then(|mut f| f.write_all(bytes));
        let temp = TempWav(path); // from here on the file is removed, even when writing failed half way
        written.map_err(|e| Fail::Other(format!("{}: {}", temp.0.display(), crate::archive::io_message(&e))))?;
        Ok(temp)
    }
}

impl Drop for TempWav {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn command(program: &str, args: &[&str], wav: &Path) -> Command {
    let mut c = Command::new(program);
    c.args(args);
    if cfg!(windows) {
        c.env("SPEAK_WAV", wav);
    } else {
        c.arg(wav);
    }
    c
}

/// The players that read audio from a pipe, in the order they are tried: the program, its arguments, and whether it wants a WAV header
/// (ffplay reads the rate and channels from it, which spares options that differ between its versions: `-ac` is gone in FFmpeg 8) or raw
/// 16-bit mono samples. macOS has no such player of its own (`afplay` takes a file only).
#[cfg_attr(not(all(unix, not(target_os = "macos"))), allow(unused_variables))] // only the Linux players take the rate as an option
fn stream_players(rate: u32) -> Vec<(&'static str, Vec<String>, bool)> {
    let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let ffplay = ("ffplay", args(&["-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "-"]), true);
    #[cfg(all(unix, not(target_os = "macos")))]
    return vec![("paplay", args(&["--raw", &format!("--rate={rate}"), "--format=s16le", "--channels=1"]), false), ("aplay", args(&["-q", "-t", "raw", "-f", "S16_LE", "-r", &rate.to_string(), "-c", "1"]), false), ffplay];
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    return vec![ffplay];
}

/// A player that is running and reads samples from its standard input.
pub struct Player {
    program: &'static str,
    child: Child,
    /// The player wants a WAV header before the samples.
    pub wav: bool,
}

impl Player {
    /// Starts the first of the pipe-reading players that the system has; `None` when there is none.
    pub fn open(rate: u32) -> Result<Option<Player>, Fail> {
        for (program, args, wav) in stream_players(rate) {
            match Command::new(program).args(&args).stdin(Stdio::piped()).stdout(Stdio::null()).spawn() {
                Ok(child) => return Ok(Some(Player { program, child, wav })),
                Err(e) if e.kind() == ErrorKind::NotFound => continue,
                Err(e) => return Err(Fail::Other(format!("{program}: {e}"))),
            }
        }
        Ok(None)
    }

    pub fn input(&mut self) -> &mut ChildStdin {
        self.child.stdin.as_mut().expect("the player's input is piped")
    }

    /// Closes the player's input, waits until the sound has ended, and says whether the player succeeded.
    pub fn finish(mut self) -> Result<(), Fail> {
        drop(self.child.stdin.take());
        let status = self.child.wait().map_err(|e| Fail::Other(format!("{}: {e}", self.program)))?;
        if status.success() { Ok(()) } else { Err(Fail::Other(format!("{} failed ({status})", self.program))) }
    }
}

/// Plays the samples and returns when the sound has ended.
pub fn play(audio: &[f32], rate: u32) -> Result<(), Fail> {
    let file = TempWav::create(&wav::encode(audio, rate))?;
    for (program, args) in PLAYERS {
        match command(program, args, &file.0).status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => return Err(Fail::Other(format!("{program} failed ({status})"))),
            Err(e) if e.kind() == ErrorKind::NotFound => continue,
            Err(e) => return Err(Fail::Other(format!("{program}: {e}"))),
        }
    }
    let tried: Vec<&str> = PLAYERS.iter().map(|p| p.0).collect();
    Err(Fail::Other(format!("no audio player found (tried {}); write a file instead: add -o OUT.ogg", tried.join(", "))))
}
