// Reader of the model container: the format of the model file.
// Runs in Node and in the browser. The input bytes are only read.

const MAGIC = 'ZERO';
const F32 = 0, F16 = 1, BLOCK = 2, I8_COLUMN = 3, BYTES = 4;
const utf8 = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

const CRC_TABLE = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  CRC_TABLE[i] = c;
}

/** CRC-32 (IEEE 802.3, as zlib.crc32). */
function crc32(bytes) {
  let c = -1;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

/** IEEE binary16 to binary32: exact, subnormals and signed zeros included (NaN payloads are not kept; the model has none). */
function f16ToF32(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const man = h & 0x3ff;
  if (exp === 0) return sign * man * 2 ** -24;
  if (exp === 31) return man ? NaN : sign * Infinity;
  return sign * (1024 + man) * 2 ** (exp - 25);
}

/**
 * Parses a model file (Uint8Array or ArrayBuffer), checking magic, version and CRC-32 (not the CRC with verify false). Returns a reader with
 * meta (string -> string), names, has(name), shape(name), f32(name) (float32 in PyTorch's shape and row-major layout)
 * and part(prefix): the same reader over the tensors named `prefix.<name>`, addressed without the prefix.
 */
export function parseTensors(bytes, { verify = true } = {}) {
  const u8 = ArrayBuffer.isView(bytes)
    ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    : new Uint8Array(bytes);
  if (u8.length < 16 || String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== MAGIC) throw new Error('not a Zero Talk model file');
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (dv.getUint32(4, true) !== 1) throw new Error(`unsupported file version ${dv.getUint32(4, true)}`);
  if (verify && dv.getUint32(8, true) !== crc32(u8.subarray(12))) throw new Error('checksum mismatch (corrupt or truncated file)');
  const end = 16 + dv.getUint32(12, true); // the metadata and tensor table end here
  if (end > u8.length) throw new Error('truncated header');

  let pos = 16;
  const take = (n) => {
    if (pos + n > end) throw new Error('truncated header');
    pos += n;
    return pos - n;
  };
  const u8At = () => u8[take(1)];
  const u32 = () => dv.getUint32(take(4), true);
  const u64 = () => {
    const p = take(8);
    return dv.getUint32(p + 4, true) * 2 ** 32 + dv.getUint32(p, true);
  };
  const str = () => {
    const n = u32();
    const p = take(n);
    return utf8.decode(u8.subarray(p, p + n));
  };

  const meta = {};
  for (let k = u32(); k > 0; k--) {
    const key = str();
    meta[key] = str();
  }

  const entries = new Map();
  for (let t = u32(); t > 0; t--) {
    const name = str();
    const code = u8At();
    const shape = [];
    const ndim = u8At();
    for (let d = 0; d < ndim; d++) shape.push(u32());
    const offset = u64();
    const nbytes = u64();
    const n = shape.reduce((a, b) => a * b, 1);
    const e = { code, shape, offset, nbytes, bits: 0, block: 0, transposed: false };
    let expected;
    if (code === F32) expected = 4 * n;
    else if (code === F16) expected = 2 * n;
    else if (code === BLOCK) {
      e.bits = u8At();
      e.block = u32();
      e.transposed = u8At() === 1;
      if (e.bits !== 4 && e.bits !== 8) throw new Error(`${name}: unsupported code width ${e.bits}`);
      if (e.transposed && shape.length !== 3) throw new Error(`${name}: the transposed layout needs 3 dimensions`);
      const rows = e.transposed ? shape[1] : shape[0];
      expected = Math.ceil((n * e.bits) / 8) + 2 * rows * Math.ceil(n / rows / e.block);
    } else if (code === I8_COLUMN) expected = n + 2 * shape[shape.length - 1];
    else if (code === BYTES) expected = n;
    else throw new Error(`${name}: unknown dtype ${code}`);
    if (nbytes !== expected || offset + nbytes > u8.length) throw new Error(`${name}: bad size or offset`);
    entries.set(name, e);
  }

  const names = [...entries.keys()];
  const entry = (name) => {
    const e = entries.get(name);
    if (!e) throw new Error(`missing tensor '${name}'`);
    return e;
  };

  const expand = (name) => {
    const { code, shape, offset, nbytes, bits, block, transposed } = entry(name);
    if (code === BYTES) throw new Error(`${name}: a byte blob, not a numeric tensor`);
    const n = shape.reduce((a, b) => a * b, 1);
    const raw = u8.subarray(offset, offset + nbytes);
    const rd = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const out = new Float32Array(n);
    if (code === F32) {
      for (let i = 0; i < n; i++) out[i] = rd.getFloat32(4 * i, true);
      return out;
    }
    if (code === F16) {
      for (let i = 0; i < n; i++) out[i] = f16ToF32(rd.getUint16(2 * i, true));
      return out;
    }
    if (code === I8_COLUMN) {
      const cols = shape[shape.length - 1];
      const scale = new Float32Array(cols);
      for (let c = 0; c < cols; c++) scale[c] = f16ToF32(rd.getUint16(n + 2 * c, true));
      for (let i = 0; i < n; i++) out[i] = rd.getInt8(i) * scale[i % cols];
      return out;
    }
    // BLOCK: one float16 scale per `block` consecutive weights of a row; low nibble first, two's complement.
    const rows = transposed ? shape[1] : shape[0];
    const len = n / rows;
    const blocks = Math.ceil(len / block);
    const scaleAt = Math.ceil((n * bits) / 8);
    const scale = new Float32Array(rows * blocks);
    for (let k = 0; k < scale.length; k++) scale[k] = f16ToF32(rd.getUint16(scaleAt + 2 * k, true));
    for (let i = 0; i < n; i++) {
      let q;
      if (bits === 8) q = rd.getInt8(i);
      else {
        q = (rd.getUint8(i >> 1) >> (4 * (i & 1))) & 0xf;
        if (q >= 8) q -= 16;
      }
      out[i] = q * scale[Math.floor(i / len) * blocks + Math.floor((i % len) / block)];
    }
    if (!transposed) return out;
    // Stored as the equivalent direct convolution W'[out][in][j]; PyTorch's layout is W[in][out][k-1-j].
    const [cin, cout, k] = shape;
    const w = new Float32Array(n);
    for (let i = 0; i < cin; i++) {
      for (let o = 0; o < cout; o++) {
        for (let j = 0; j < k; j++) w[(i * cout + o) * k + j] = out[(o * cin + i) * k + (k - 1 - j)];
      }
    }
    return w;
  };

  const reader = (p) => ({ // p is '' or 'prefix.'
    meta,
    names: names.filter((n) => n.startsWith(p)).map((n) => n.slice(p.length)),
    has: (name) => entries.has(p + name),
    shape: (name) => entry(p + name).shape.slice(),
    f32: (name) => expand(p + name),
    part: (prefix) => reader(`${p}${prefix}.`),
  });
  return reader('');
}
