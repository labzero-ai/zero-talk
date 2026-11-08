//! 16-bit PCM mono WAV files. Each file has a `LIST` chunk that carries the AI-generated marker (see `marker`), so its header is
//! `HEADER_BYTES` long, and the samples follow it.
use crate::marker::{COMMENT, SOFTWARE};

pub fn pcm16(x: f32) -> i16 {
    (x * 32767.0).round().clamp(-32768.0, 32767.0) as i16
}

/// The length of the header of a WAV file: the `RIFF` and `fmt ` chunks (36 bytes), the `LIST` chunk with the marker (108 bytes) and
/// the header of the `data` chunk (8 bytes).
pub const HEADER_BYTES: usize = 152;

/// An item of the `INFO` list: its four-character id, its size, the text and a 0x00 terminator, and a 0x00 pad byte when the size is
/// odd.
fn info_item(id: &[u8; 4], text: &str) -> Vec<u8> {
    let size = text.len() + 1;
    let mut item = id.to_vec();
    item.extend_from_slice(&(size as u32).to_le_bytes());
    item.extend_from_slice(text.as_bytes());
    item.push(0);
    if size % 2 == 1 {
        item.push(0);
    }
    item
}

/// The header of a mono 16-bit WAV of `samples` samples: `HEADER_BYTES` long, with the marker in its `LIST` chunk. `None` leaves the
/// RIFF and data sizes open (0xFFFFFFFF), as ffmpeg writes a WAV that goes to a pipe; the `LIST` chunk always has its real size.
pub fn header(rate: u32, samples: Option<usize>) -> Vec<u8> {
    let info = [info_item(b"ICMT", COMMENT), info_item(b"ISFT", SOFTWARE)].concat();
    let list = 4 + info.len() as u32; // "INFO" and the items
    let (riff, data) = samples.map_or((u32::MAX, u32::MAX), |n| (44 + list + 2 * n as u32, 2 * n as u32));
    let mut h = Vec::with_capacity(HEADER_BYTES);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&riff.to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(2 * rate).to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"LIST");
    h.extend_from_slice(&list.to_le_bytes());
    h.extend_from_slice(b"INFO");
    h.extend_from_slice(&info);
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data.to_le_bytes());
    h
}

/// The samples as little-endian 16-bit PCM.
pub fn pcm_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 * samples.len());
    push_pcm(&mut out, samples);
    out
}

fn push_pcm(out: &mut Vec<u8>, samples: &[f32]) {
    for &s in samples {
        out.extend_from_slice(&pcm16(s).to_le_bytes());
    }
}

/// A complete WAV file: the header and the samples.
pub fn encode(samples: &[f32], rate: u32) -> Vec<u8> {
    let mut out = header(rate, Some(samples.len()));
    out.reserve_exact(2 * samples.len());
    push_pcm(&mut out, samples);
    out
}

#[cfg(test)]
mod tests {
    use crate::marker::{COMMENT, SOFTWARE};

    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn header_and_samples() {
        let w = super::encode(&[0.0, 1.0, -1.0], 24_000);
        assert_eq!(w.len(), super::HEADER_BYTES + 6);
        assert_eq!(&w[..4], b"RIFF");
        assert_eq!(u32_at(&w, 4) as usize, w.len() - 8);
        assert_eq!(u32_at(&w, 24), 24_000);
        let first = super::HEADER_BYTES;
        assert_eq!(i16::from_le_bytes(w[first + 2..first + 4].try_into().unwrap()), 32767);
        assert_eq!(i16::from_le_bytes(w[first + 4..first + 6].try_into().unwrap()), -32767);
    }

    #[test]
    fn the_header_carries_the_marker_in_a_list_chunk_before_the_data() {
        let h = super::header(24_000, Some(3));
        let mut marker = b"LIST".to_vec();
        marker.extend(100u32.to_le_bytes());
        marker.extend_from_slice(b"INFO");
        marker.extend_from_slice(b"ICMT");
        marker.extend(66u32.to_le_bytes());
        marker.extend_from_slice(COMMENT.as_bytes());
        marker.push(0);
        marker.extend_from_slice(b"ISFT");
        marker.extend(14u32.to_le_bytes());
        marker.extend_from_slice(SOFTWARE.as_bytes());
        marker.push(0);
        let data = super::HEADER_BYTES - 8;
        assert_eq!(&h[36..data], &marker[..]);
        assert_eq!(&h[data..data + 4], b"data");
        assert_eq!(u32_at(&h, data + 4), 6);
        assert_eq!(u32_at(&h, 4), 44 + 100 + 6);
    }

    #[test]
    fn a_streaming_header_leaves_the_riff_and_data_sizes_open() {
        let h = super::header(24_000, None);
        assert_eq!(h.len(), super::HEADER_BYTES);
        assert_eq!(u32_at(&h, 4), u32::MAX);
        assert_eq!(&h[36..40], b"LIST");
        assert_eq!(u32_at(&h, 40), 100);
        let data = super::HEADER_BYTES - 8;
        assert_eq!(&h[data..data + 4], b"data");
        assert_eq!(u32_at(&h, data + 4), u32::MAX);
    }

    #[test]
    fn an_info_item_of_odd_size_has_a_pad_byte() {
        assert_eq!(super::info_item(b"ABCD", "xy"), b"ABCD\x03\0\0\0xy\0\0".to_vec());
        assert_eq!(super::info_item(b"ABCD", "xyz"), b"ABCD\x04\0\0\0xyz\0".to_vec());
    }
}
