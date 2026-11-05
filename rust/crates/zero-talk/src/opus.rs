//! Ogg Opus files: mono audio coded by `opus-pure`, a pure-Rust port of libopus.
use opus_pure::{Application, OggOpusWriter, OpusEncoder, OpusHead, OpusTags, MAX_PACKET_BYTES};

use crate::marker;
use crate::wav::pcm16;

pub const DEFAULT_KBPS: u32 = 24;
/// The bit rates `zero-talk` accepts, in kbit/s. 24 is the default.
pub const KBPS: std::ops::RangeInclusive<u32> = 8..=128;

/// The Ogg page size on a stream that is read as it is written: about 3 packets (60 ms at the default rate) instead of the crate's
/// 4,096 bytes, so a reader gets the audio in small pieces. A page leaves when the next packet is written.
pub const STREAM_PAGE_BYTES: usize = 160;

fn fail(e: opus_pure::Error) -> String {
    format!("the Opus encoder failed: {e}")
}

/// A mono Ogg Opus stream of audio at `rate` Hz (8, 12, 16, 24 or 48 kHz), at about `kbps` kbit/s on average, that takes its samples in
/// pieces: each 20 ms frame is coded, and its page written to the sink, as soon as the samples for it are there. The samples are rounded
/// to 16 bits first, as in the WAV. It decodes to exactly the length of the
/// samples: the encoder's delay is announced in the header and the padding that flushes it is trimmed by the last page (RFC 7845).
pub struct OggStream<W: std::io::Write> {
    encoder: OpusEncoder,
    writer: OggOpusWriter<W>,
    pending: Vec<f32>, // rounded samples that do not fill a frame yet
    frame: usize,      // Opus codes 20 ms frames at the encoder's own rate
    ticks: usize,      // the container counts in 48 kHz ticks; every Opus rate divides 48 kHz
    pre_skip: u64,
    taken: usize,
    frames: usize, // frames coded so far
}

impl<W: std::io::Write> OggStream<W> {
    /// Writes the two header pages to `sink` at once. `page_bytes` is the page size (`None`: the crate's default, as a file has it).
    pub fn new(sink: W, rate: u32, kbps: u32, page_bytes: Option<usize>) -> Result<Self, String> {
        let mut encoder = OpusEncoder::new(rate as i32, 1, Application::Audio).map_err(fail)?;
        encoder.bitrate_bps = kbps as i32 * 1000;
        let mut tags = OpusTags::new();
        tags.push("ENCODER", "opus-pure 0.2.2").map_err(fail)?; // the opus-pure version that Cargo.toml pins: change it with an upgrade
        tags.push("COMMENT", marker::COMMENT).map_err(fail)?;
        tags.push("DIGITALSOURCETYPE", marker::SOURCE).map_err(fail)?;
        let head = OpusHead::for_encoder(&encoder, rate);
        let pre_skip = u64::from(head.pre_skip);
        let mut writer = OggOpusWriter::with_tags(sink, head, tags).map_err(fail)?;
        if let Some(bytes) = page_bytes {
            writer.set_page_target(bytes);
        }
        Ok(OggStream { encoder, writer, pending: Vec::new(), frame: rate as usize / 50, ticks: 48_000 / rate as usize, pre_skip, taken: 0, frames: 0 })
    }

    fn code(&mut self, block: &[f32], last: Option<u32>) -> Result<(), String> {
        let mut packet = vec![0u8; MAX_PACKET_BYTES];
        let n = self.encoder.encode(block, self.frame, &mut packet).map_err(fail)?;
        match last {
            Some(duration) => self.writer.write_packet_with_duration(&packet[..n], duration),
            None => self.writer.write_packet(&packet[..n]),
        }
        .map_err(fail)?;
        self.frames += 1;
        Ok(())
    }

    /// Takes more audio and writes the pages that are complete, then flushes the sink.
    pub fn push(&mut self, audio: &[f32]) -> Result<(), String> {
        self.taken += audio.len();
        self.pending.extend(audio.iter().map(|&x| pcm16(x) as f32 / 32768.0));
        let whole = self.pending.len() / self.frame * self.frame;
        let coded: Vec<f32> = self.pending.drain(..whole).collect();
        for block in coded.chunks(self.frame) {
            self.code(block, None)?;
        }
        match self.writer.get_mut() {
            Some(sink) => sink.flush().map_err(|e| format!("write failed: {e}")),
            None => Ok(()),
        }
    }

    /// Codes the rest of the audio and enough silence to flush the encoder, and writes the last page, which says where the audio stopped.
    pub fn finish(mut self) -> Result<W, String> {
        let frames = (self.taken + (self.pre_skip as usize).div_ceil(self.ticks)).div_ceil(self.frame);
        let final_granule = self.pre_skip + (self.taken * self.ticks) as u64;
        let mut block = std::mem::take(&mut self.pending);
        while self.frames < frames {
            block.resize(self.frame, 0.0);
            let last = (self.frames + 1 == frames).then(|| (final_granule - self.writer.granule() as u64) as u32);
            self.code(&block, last)?;
            block.clear();
        }
        self.writer.finish().map_err(fail)
    }
}

/// A complete Ogg Opus file of the mono audio: the stream above, in memory.
pub fn encode_ogg(audio: &[f32], rate: u32, kbps: u32) -> Result<Vec<u8>, String> {
    let mut stream = OggStream::new(Vec::new(), rate, kbps, None)?;
    stream.push(audio)?;
    stream.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The granule position of the last page: the pre-skip plus the audio's length at 48 kHz.
    fn last_granule(file: &[u8]) -> u64 {
        let at = file.windows(4).rposition(|w| w == b"OggS").unwrap();
        u64::from_le_bytes(file[at + 6..at + 14].try_into().unwrap())
    }

    #[test]
    fn the_file_announces_exactly_the_input_length() {
        let audio: Vec<f32> = (0..24_000 + 123).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect();
        let file = encode_ogg(&audio, 24_000, 24).unwrap();
        assert!(file.starts_with(b"OggS") && file.windows(8).any(|w| w == b"OpusHead"));
        let pre_skip = u16::from_le_bytes(file[28 + 10..28 + 12].try_into().unwrap()) as u64;
        assert_eq!(last_granule(&file), pre_skip + 2 * audio.len() as u64);
        assert!((1_000..8_000).contains(&file.len()), "{} bytes for one second at 24 kbps", file.len());
    }

    fn signal(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.3 * (i as f32 * 0.05).sin() * (i as f32 * 0.0007).cos()).collect()
    }

    #[test]
    fn a_stream_given_in_pieces_is_the_file() {
        let audio = signal(2 * 24_000 + 77);
        let mut stream = OggStream::new(Vec::new(), 24_000, 24, None).unwrap();
        let (mut at, mut sizes) = (0, [1, 479, 481, 5_000, 12_345].into_iter().cycle());
        while at < audio.len() {
            let end = (at + sizes.next().unwrap()).min(audio.len());
            stream.push(&audio[at..end]).unwrap();
            at = end;
        }
        assert_eq!(stream.finish().unwrap(), encode_ogg(&audio, 24_000, 24).unwrap());
    }

    #[test]
    fn small_pages_make_more_pages_and_end_in_the_same_place() {
        let audio = signal(2 * 24_000 + 77);
        let mut stream = OggStream::new(Vec::new(), 24_000, 24, Some(STREAM_PAGE_BYTES)).unwrap();
        stream.push(&audio).unwrap();
        let (small, whole) = (stream.finish().unwrap(), encode_ogg(&audio, 24_000, 24).unwrap());
        let pages = |f: &[u8]| f.windows(4).filter(|w| w == b"OggS").count();
        assert!(pages(&small) > 5 * pages(&whole), "{} pages against {}", pages(&small), pages(&whole));
        assert_eq!(last_granule(&small), last_granule(&whole));
    }

    #[test]
    fn the_comments_are_the_encoder_then_the_marker() {
        let file = encode_ogg(&signal(24_000), 24_000, 24).unwrap();
        let reader = opus_pure::OggOpusReader::new(std::io::Cursor::new(file)).unwrap();
        assert_eq!(reader.tags().vendor, "opus-pure 0.2.2");
        assert_eq!(
            reader.tags().comments,
            [
                "ENCODER=opus-pure 0.2.2".to_string(),
                format!("COMMENT={}", marker::COMMENT),
                format!("DIGITALSOURCETYPE={}", marker::SOURCE),
            ]
        );
    }

    #[test]
    fn a_rate_the_encoder_does_not_take_is_an_error() {
        assert!(encode_ogg(&[0.0; 1000], 44_100, 24).is_err());
    }
}
