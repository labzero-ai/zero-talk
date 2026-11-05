//! The Zero Talk container, version 1: the model file and the front-end file. Each holds named tensors and metadata, with a
//! checksum over the rest of the file. Tensors are returned as float32 in PyTorch's shape and memory layout.
use std::collections::HashMap;
use std::path::Path;

pub type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    F32,
    F16,
    /// 4- or 8-bit codes with one float16 scale per `block` consecutive weights of a row.
    Block { bits: usize, block: usize, transposed: bool },
    /// int8 codes with one float16 scale per column of the last axis.
    I8Column,
    /// An opaque blob (text tables); shape is `[length]`.
    Bytes,
}

#[derive(Debug)]
pub struct Entry {
    pub dtype: DType,
    pub shape: Vec<usize>,
    offset: usize,
    nbytes: usize,
}

pub struct Archive {
    bytes: Vec<u8>,
    pub meta: HashMap<String, String>,
    pub entries: HashMap<String, Entry>,
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.b.get(self.pos..self.pos + n).ok_or("truncated header")?;
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|e| e.to_string())
    }
}

impl Archive {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), io_message(&e)))?;
        Self::parse(bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < 16 || &bytes[..4] != b"ZERO" {
            return Err("not a Zero Talk file".into());
        }
        let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        if word(4) != 1 {
            return Err(format!("unsupported file version {}", word(4)));
        }
        if word(8) != crc32(&bytes[12..]) {
            return Err("checksum mismatch (corrupt or truncated file)".into());
        }
        let header_end = (word(12) as usize).checked_add(16).filter(|&end| end <= bytes.len()).ok_or("header longer than the file")?;
        let mut c = Cursor { b: &bytes[..header_end], pos: 16 };
        let mut meta = HashMap::new();
        for _ in 0..c.u32()? {
            let k = c.str()?;
            meta.insert(k, c.str()?);
        }
        let mut entries = HashMap::new();
        for _ in 0..c.u32()? {
            let name = c.str()?;
            let code = c.u8()?;
            let shape = (0..c.u8()?).map(|_| c.u32().map(|d| d as usize)).collect::<Result<Vec<_>>>()?;
            let (offset, nbytes) = (c.u64()? as usize, c.u64()? as usize);
            let n: usize = shape.iter().product();
            let (dtype, expected) = match code {
                0 => (DType::F32, 4 * n),
                1 => (DType::F16, 2 * n),
                2 => {
                    let (bits, block, layout) = (c.u8()? as usize, c.u32()? as usize, c.u8()?);
                    let transposed = layout == 1;
                    if bits != 4 && bits != 8 {
                        return Err(format!("{name}: unsupported code width {bits}"));
                    }
                    // A transposed weight is a ConvTranspose1d's [in, out, k]; otherwise the rows are the first axis.
                    if shape.is_empty() || (transposed && shape.len() != 3) {
                        return Err(format!("{name}: a block-coded tensor needs an axis (three if transposed)"));
                    }
                    let rows = if transposed { shape[1] } else { shape[0] };
                    if rows == 0 || block == 0 {
                        return Err(format!("{name}: a block-coded tensor with zero rows or a zero block size"));
                    }
                    let blocks = (n / rows).div_ceil(block);
                    (DType::Block { bits, block, transposed }, (n * bits).div_ceil(8) + 2 * rows * blocks)
                }
                3 => {
                    let cols = *shape.last().ok_or_else(|| format!("{name}: an int8 tensor needs an axis"))?;
                    (DType::I8Column, n + 2 * cols)
                }
                4 => (DType::Bytes, n),
                _ => return Err(format!("{name}: unknown dtype {code}")),
            };
            if nbytes != expected || offset.checked_add(nbytes).map_or(true, |end| end > bytes.len()) {
                return Err(format!("{name}: bad size or offset"));
            }
            entries.insert(name, Entry { dtype, shape, offset, nbytes });
        }
        Ok(Archive { bytes, meta, entries })
    }

    pub fn meta(&self, key: &str) -> Result<&str> {
        self.meta.get(key).map(|s| s.as_str()).ok_or(format!("missing metadata '{key}'"))
    }

    /// A metadata value parsed as a number (or anything else `FromStr`).
    pub fn meta_as<T: std::str::FromStr>(&self, key: &str) -> Result<T> {
        self.meta(key)?.parse().map_err(|_| format!("bad metadata '{key}'"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.entry(name)?.shape)
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.entries.get(name).ok_or(format!("missing tensor '{name}'"))
    }

    /// A byte blob as stored (the front end's text tables).
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let e = self.entry(name)?;
        if e.dtype != DType::Bytes {
            return Err(format!("{name}: not a byte blob"));
        }
        Ok(&self.bytes[e.offset..e.offset + e.nbytes])
    }

    /// The tensor as float32, in PyTorch's shape and layout (4-bit and float16 values expand exactly).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entry(name)?;
        let raw = &self.bytes[e.offset..e.offset + e.nbytes];
        let n: usize = e.shape.iter().product();
        Ok(match e.dtype {
            DType::Bytes => return Err(format!("{name}: a byte blob, not a numeric tensor")),
            DType::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            DType::F16 => raw.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
            DType::I8Column => {
                let cols = e.shape[e.shape.len() - 1];
                let scales = &raw[n..];
                (0..n)
                    .map(|i| {
                        let s = 2 * (i % cols);
                        raw[i] as i8 as f32 * f16_to_f32(u16::from_le_bytes([scales[s], scales[s + 1]]))
                    })
                    .collect()
            }
            DType::Block { bits, block, transposed } => {
                let rows = if transposed { e.shape[1] } else { e.shape[0] };
                let len = n / rows;
                let blocks = len.div_ceil(block);
                let scales = &raw[(n * bits).div_ceil(8)..];
                let mut w: Vec<f32> = (0..n)
                    .map(|i| {
                        let q = if bits == 8 {
                            raw[i] as i8 as i32
                        } else {
                            let nib = (raw[i / 2] >> (4 * (i % 2))) & 0xF;
                            if nib >= 8 { nib as i32 - 16 } else { nib as i32 }
                        };
                        let s = 2 * ((i / len) * blocks + (i % len) / block);
                        q as f32 * f16_to_f32(u16::from_le_bytes([scales[s], scales[s + 1]]))
                    })
                    .collect();
                if transposed {
                    // Stored as the equivalent direct convolution W'[out][in][j]; PyTorch's is W[in][out][k-1-j].
                    let (cin, cout, k) = (e.shape[0], e.shape[1], e.shape[2]);
                    let stored = std::mem::take(&mut w);
                    w = vec![0.0; n];
                    for i in 0..cin {
                        for o in 0..cout {
                            for j in 0..k {
                                w[(i * cout + o) * k + j] = stored[(o * cin + i) * k + (k - 1 - j)];
                            }
                        }
                    }
                }
                w
            }
        })
    }
}

/// A model part: the tensors whose names start with `prefix.`, addressed without it.
pub struct Part<'a> {
    ar: &'a Archive,
    prefix: String,
}

impl Archive {
    pub fn part(&self, prefix: &str) -> Part<'_> {
        Part { ar: self, prefix: format!("{prefix}.") }
    }
}

impl Part<'_> {
    pub fn meta(&self, key: &str) -> Result<&str> {
        self.ar.meta(key)
    }
    pub fn meta_as<T: std::str::FromStr>(&self, key: &str) -> Result<T> {
        self.ar.meta_as(key)
    }
    pub fn has(&self, name: &str) -> bool {
        self.ar.has(&format!("{}{name}", self.prefix))
    }
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        self.ar.shape(&format!("{}{name}", self.prefix))
    }
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        self.ar.f32(&format!("{}{name}", self.prefix))
    }
}

/// IEEE 754 binary16 to binary32 (exact, including subnormals).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1F;
    let man = (h & 0x3FF) as u32;
    match exp {
        0 => sign * man as f32 * (1.0 / 16_777_216.0), // man * 2^-24
        31 => f32::from_bits(((h as u32 & 0x8000) << 16) | 0x7F80_0000 | (man << 13)),
        _ => f32::from_bits(((h as u32 & 0x8000) << 16) | ((exp as u32 + 112) << 23) | (man << 13)),
    }
}

/// CRC-32 (IEEE 802.3, as zlib.crc32).
pub fn crc32(data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut t = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                k += 1;
            }
            t[i] = c;
            i += 1;
        }
        t
    };
    !data.iter().fold(!0u32, |c, &b| TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8))
}

/// An I/O error without the "(os error N)" tail.
pub fn io_message(e: &std::io::Error) -> String {
    let s = e.to_string();
    s.split(" (os error").next().unwrap_or(&s).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A container file: the 16-byte prefix (magic, version, checksum of `rest`) and then `rest`.
    fn file(rest: &[u8]) -> Vec<u8> {
        let mut f = b"ZERO".to_vec();
        f.extend_from_slice(&1u32.to_le_bytes());
        f.extend_from_slice(&crc32(rest).to_le_bytes());
        f.extend_from_slice(rest);
        f
    }

    /// `rest` with a header of no metadata and the given tensor entries, then the data.
    fn body(entries: &[u8], data: &[u8]) -> Vec<u8> {
        let mut header = 0u32.to_le_bytes().to_vec();
        header.extend_from_slice(&1u32.to_le_bytes());
        header.extend_from_slice(entries);
        let mut rest = (header.len() as u32).to_le_bytes().to_vec();
        rest.extend_from_slice(&header);
        rest.extend_from_slice(data);
        rest
    }

    /// One tensor entry named `w`: code, shape, offset, nbytes, then the code's own fields in `extra`.
    fn entry(code: u8, shape: &[u32], offset: u64, nbytes: u64, extra: &[u8]) -> Vec<u8> {
        let mut e = 1u32.to_le_bytes().to_vec();
        e.push(b'w');
        e.push(code);
        e.push(shape.len() as u8);
        for d in shape {
            e.extend_from_slice(&d.to_le_bytes());
        }
        e.extend_from_slice(&offset.to_le_bytes());
        e.extend_from_slice(&nbytes.to_le_bytes());
        e.extend_from_slice(extra);
        e
    }

    #[test]
    fn a_header_longer_than_the_file_is_an_error() {
        let mut rest = 1000u32.to_le_bytes().to_vec();
        rest.extend_from_slice(&[0; 8]);
        assert!(Archive::parse(file(&rest)).is_err());
    }

    #[test]
    fn a_transposed_block_tensor_with_one_axis_is_an_error() {
        // code 2, shape [4], layout 1 (transposed), 4 bits, block 4: a 1-D transposed weight has no shape[1]
        let e = entry(2, &[4], 0, 0, &[4, 4, 0, 0, 0, 1]);
        assert!(Archive::parse(file(&body(&e, &[]))).is_err());
    }

    #[test]
    fn an_offset_that_overflows_is_an_error() {
        // a float32 [1] at offset u64::MAX: offset + nbytes wraps to a small number
        let e = entry(0, &[1], u64::MAX, 4, &[]);
        assert!(Archive::parse(file(&body(&e, &[0; 4]))).is_err());
    }
}
