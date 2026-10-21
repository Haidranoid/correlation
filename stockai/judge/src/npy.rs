//! Minimal NumPy `.npy` files: format 1.0 writer and a reader for formats 1.0-3.0, for
//! little-endian `<i4`, `<f4` and `<f8` arrays and `|i1` (int8) arrays in C order.
//!
//! Layout: magic `\x93NUMPY`, major and minor version bytes, header length (u16 LE in 1.0, u32 LE
//! in 2.0/3.0), then the header: a Python dict literal such as
//! `{'descr': '<f4', 'fortran_order': False, 'shape': (8503, 5911), }`, padded with spaces and
//! ended by a newline so that the data starts on a 64-byte boundary.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

const MAGIC: &[u8; 6] = b"\x93NUMPY";
const ALIGN: usize = 64;

/// An element type the files can hold.
pub trait Element: Copy + Send + Sync + 'static {
    const DESCR: &'static str;
    const SIZE: usize;
    fn put_le(self, out: &mut Vec<u8>);
    fn get_le(b: &[u8]) -> Self;
}

macro_rules! element {
    ($t:ty, $descr:literal) => {
        impl Element for $t {
            const DESCR: &'static str = $descr;
            const SIZE: usize = std::mem::size_of::<$t>();
            fn put_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn get_le(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b.try_into().expect("element size"))
            }
        }
    };
}
element!(i8, "|i1");
element!(i32, "<i4");
element!(f32, "<f4");
element!(f64, "<f8");

/// The full 1.0 preamble (magic, version, length, padded header) for an array of `shape`.
pub fn header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    // A one-element tuple needs its trailing comma: `(5911,)`, not `(5911)`.
    let shape_text = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}");
    let unpadded = MAGIC.len() + 2 + 2 + dict.len() + 1;
    let total = unpadded.div_ceil(ALIGN) * ALIGN;
    let header_len = total - MAGIC.len() - 4;
    assert!(header_len <= u16::MAX as usize, "npy header too long");
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.resize(total - 1, b' ');
    out.push(b'\n');
    out
}

/// Writes `path` through a temporary file in the same directory, renamed into place at the end,
/// so readers see either the old file or the complete new one.
pub fn write_atomic(path: &Path, body: impl FnOnce(&mut BufWriter<File>) -> Result<()>) -> Result<()> {
    let tmp = temp_path(path);
    let result = (|| -> Result<()> {
        let mut w = BufWriter::with_capacity(8 << 20, File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?);
        body(&mut w)?;
        let f = w.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.context(format!("writing {}", path.display())));
    }
    std::fs::rename(&tmp, path).with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

fn temp_path(path: &Path) -> PathBuf {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    path.with_file_name(format!(".{name}.tmp-{}", std::process::id()))
}

/// Writes a C-order array of `shape` (row-major: the last index varies fastest).
pub fn write<T: Element>(path: &Path, shape: &[usize], data: &[T]) -> Result<()> {
    ensure!(shape.iter().product::<usize>() == data.len(), "{}: {} values for shape {shape:?}", path.display(), data.len());
    write_atomic(path, |w| {
        w.write_all(&header(T::DESCR, shape))?;
        let mut buf = Vec::with_capacity(1 << 20);
        for chunk in data.chunks((1 << 20) / T::SIZE) {
            buf.clear();
            for &x in chunk {
                x.put_le(&mut buf);
            }
            w.write_all(&buf)?;
        }
        Ok(())
    })
}

/// A parsed header.
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

/// An array read from a file.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    I1(Vec<i8>),
    I4(Vec<i32>),
    F4(Vec<f32>),
    F8(Vec<f64>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    pub header: Header,
    pub data: Data,
}

/// Reads an `.npy` file of `|i1`, `<i4`, `<f4` or `<f8` values in C order.
pub fn read(path: &Path) -> Result<Array> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let size = file.metadata()?.len() as usize;
    let mut r = BufReader::with_capacity(8 << 20, file);
    let (header, offset) = read_header(&mut r).with_context(|| format!("{}: not a readable .npy file", path.display()))?;
    ensure!(!header.fortran_order, "{}: the array is in Fortran order; save it C-ordered (np.ascontiguousarray)", path.display());
    let count: usize = header.shape.iter().product();
    let elem = match header.descr.as_str() {
        "|i1" => 1,
        "<i4" => 4,
        "<f4" => 4,
        "<f8" => 8,
        other => bail!("{}: dtype {other:?} is not supported (expected '|i1', '<i4', '<f4' or '<f8')", path.display()),
    };
    ensure!(
        size == offset + count * elem,
        "{}: {} data bytes for shape {:?} of {} (expected {})",
        path.display(),
        size.saturating_sub(offset),
        header.shape,
        header.descr,
        count * elem
    );
    let data = match header.descr.as_str() {
        "|i1" => Data::I1(read_values(&mut r, count)?),
        "<i4" => Data::I4(read_values(&mut r, count)?),
        "<f4" => Data::F4(read_values(&mut r, count)?),
        _ => Data::F8(read_values(&mut r, count)?),
    };
    Ok(Array { header, data })
}

fn read_values<T: Element>(r: &mut impl Read, count: usize) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(count);
    let per = (8 << 20) / T::SIZE;
    let mut buf = vec![0u8; per * T::SIZE];
    let mut left = count;
    while left > 0 {
        let k = left.min(per);
        let b = &mut buf[..k * T::SIZE];
        r.read_exact(b)?;
        out.extend(b.chunks_exact(T::SIZE).map(T::get_le));
        left -= k;
    }
    Ok(out)
}

/// Reads the preamble and returns the header and the data offset.
pub fn read_header(r: &mut impl Read) -> Result<(Header, usize)> {
    let mut pre = [0u8; 8];
    r.read_exact(&mut pre)?;
    ensure!(&pre[..6] == MAGIC, "bad magic (not a .npy file)");
    let (major, minor) = (pre[6], pre[7]);
    let (len, offset) = match major {
        1 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            let len = u16::from_le_bytes(b) as usize;
            (len, 10 + len)
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            let len = u32::from_le_bytes(b) as usize;
            (len, 12 + len)
        }
        _ => bail!("unsupported .npy format version {major}.{minor}"),
    };
    let mut text = vec![0u8; len];
    r.read_exact(&mut text)?;
    let text = String::from_utf8(text).context("header is not text")?;
    Ok((parse_header(&text)?, offset))
}

/// Parses the header dict: `{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }`.
pub fn parse_header(text: &str) -> Result<Header> {
    let t = text.trim();
    ensure!(t.starts_with('{') && t.ends_with('}'), "header is not a dict: {t:?}");
    let value_after = |key: &str| -> Result<&str> {
        let k1 = format!("'{key}'");
        let k2 = format!("\"{key}\"");
        let at = t.find(&k1).map(|p| p + k1.len()).or_else(|| t.find(&k2).map(|p| p + k2.len()));
        let rest = &t[at.with_context(|| format!("header has no {key:?}: {t:?}"))?..];
        let rest = rest.trim_start();
        let rest = rest.strip_prefix(':').with_context(|| format!("no ':' after {key:?} in {t:?}"))?;
        Ok(rest.trim_start())
    };
    let d = value_after("descr")?;
    let q = d.chars().next().context("empty descr")?;
    ensure!(q == '\'' || q == '"', "descr is not a string in {t:?} (structured dtypes are not supported)");
    let end = d[1..].find(q).context("unterminated descr")?;
    let descr = d[1..1 + end].to_string();
    let f = value_after("fortran_order")?;
    let fortran_order = if f.starts_with("True") {
        true
    } else if f.starts_with("False") {
        false
    } else {
        bail!("bad fortran_order in {t:?}")
    };
    let s = value_after("shape")?;
    ensure!(s.starts_with('('), "shape is not a tuple in {t:?}");
    let close = s.find(')').context("unterminated shape")?;
    let shape = s[1..close]
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|x| x.trim_end_matches('L').parse::<usize>().with_context(|| format!("bad dimension {x:?} in {t:?}")))
        .collect::<Result<Vec<usize>>>()?;
    Ok(Header { descr, fortran_order, shape })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("stockai-judge-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn header_is_aligned_and_numpy_shaped() {
        for (descr, shape) in [("<f4", vec![8503, 5911]), ("<i4", vec![5911]), ("<f8", vec![3, 4]), ("<f8", vec![0]), ("<f4", vec![123456789, 987654321])] {
            let h = header(descr, &shape);
            assert_eq!(h.len() % 64, 0, "{descr} {shape:?}");
            assert_eq!(*h.last().unwrap(), b'\n');
            assert_eq!(&h[..8], b"\x93NUMPY\x01\x00");
            assert_eq!(u16::from_le_bytes([h[8], h[9]]) as usize, h.len() - 10);
            let text = std::str::from_utf8(&h[10..]).unwrap();
            let parsed = parse_header(text).unwrap();
            assert_eq!(parsed, Header { descr: descr.into(), fortran_order: false, shape: shape.clone() });
        }
        let one = header("<i4", &[5911]);
        assert!(std::str::from_utf8(&one[10..]).unwrap().contains("'shape': (5911,), }"), "1-D shape needs a trailing comma");
    }

    #[test]
    fn parses_headers_numpy_writes() {
        // Exactly what numpy 2.x writes for np.zeros((3, 4)) and np.zeros(7, np.float32), and a
        // version 2.0 style header with double quotes.
        let h = parse_header("{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }                                                          \n").unwrap();
        assert_eq!(h, Header { descr: "<f8".into(), fortran_order: false, shape: vec![3, 4] });
        let h = parse_header("{'descr': '<f4', 'fortran_order': False, 'shape': (7,), }").unwrap();
        assert_eq!(h.shape, vec![7]);
        let h = parse_header("{\"descr\": \"<i4\", \"fortran_order\": True, \"shape\": ()}").unwrap();
        assert_eq!(h, Header { descr: "<i4".into(), fortran_order: true, shape: vec![] });
        assert!(parse_header("not a dict").is_err());
        assert!(parse_header("{'descr': [('a', '<f4')], 'fortran_order': False, 'shape': (2,), }").is_err());
    }

    #[test]
    fn round_trip_f32_f64_i32() {
        let p = scratch("rt_f32.npy");
        let v: Vec<f32> = (0..7 * 13).map(|k| if k % 11 == 0 { f32::NAN } else { k as f32 * -0.37 + 1e-7 }).collect();
        write(&p, &[7, 13], &v).unwrap();
        let a = read(&p).unwrap();
        assert_eq!(a.header, Header { descr: "<f4".into(), fortran_order: false, shape: vec![7, 13] });
        let Data::F4(back) = a.data else { panic!("dtype") };
        assert_eq!(back.len(), v.len());
        for (x, y) in v.iter().zip(&back) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_eq!(std::fs::metadata(&p).unwrap().len() as usize, header("<f4", &[7, 13]).len() + 4 * v.len());

        let p = scratch("rt_f64.npy");
        let v: Vec<f64> = vec![f64::NAN, -0.0, 1.0 / 3.0, f64::MIN_POSITIVE, -1e300, 5.0];
        write(&p, &[6], &v).unwrap();
        let Data::F8(back) = read(&p).unwrap().data else { panic!("dtype") };
        assert_eq!(v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), back.iter().map(|x| x.to_bits()).collect::<Vec<_>>());

        let p = scratch("rt_i32.npy");
        let v: Vec<i32> = vec![11_684, 11_685, 20_266, i32::MIN, i32::MAX, 0];
        write(&p, &[6], &v).unwrap();
        let a = read(&p).unwrap();
        assert_eq!(a.header.shape, vec![6]);
        assert_eq!(a.data, Data::I4(v));

        // int8, as numpy writes it ('|i1': no byte order), e.g. delist_kind.npy.
        let p = scratch("rt_i8.npy");
        let v: Vec<i8> = vec![0, 1, 2, 3, -1, i8::MIN, i8::MAX];
        write(&p, &[7], &v).unwrap();
        let a = read(&p).unwrap();
        assert_eq!(a.header, Header { descr: "|i1".into(), fortran_order: false, shape: vec![7] });
        assert_eq!(a.data, Data::I1(v));
        assert_eq!(std::fs::metadata(&p).unwrap().len() as usize, header("|i1", &[7]).len() + 7);

        // Bigger than one write chunk.
        let p = scratch("rt_big.npy");
        let v: Vec<f64> = (0..300_001).map(|k| (k as f64).sqrt()).collect();
        write(&p, &[300_001], &v).unwrap();
        assert_eq!(read(&p).unwrap().data, Data::F8(v));
    }

    #[test]
    fn rejects_bad_files() {
        let p = scratch("bad_magic.npy");
        std::fs::write(&p, b"PK\x03\x04 not npy at all").unwrap();
        assert!(read(&p).is_err());

        // Fortran order and unsupported dtypes are refused with a clear message.
        let p = scratch("fortran.npy");
        let mut bytes = header("<f8", &[2, 2]);
        let text = String::from_utf8(bytes[10..].to_vec()).unwrap().replace("False", "True ");
        bytes.truncate(10);
        bytes.extend_from_slice(text.as_bytes());
        bytes.extend_from_slice(&[0u8; 32]);
        std::fs::write(&p, &bytes).unwrap();
        let e = read(&p).unwrap_err().to_string();
        assert!(e.contains("Fortran"), "{e}");

        let p = scratch("f2.npy");
        let mut bytes = header("<f2", &[4]);
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(&p, &bytes).unwrap();
        let e = format!("{:#}", read(&p).unwrap_err());
        assert!(e.contains("<f2"), "{e}");

        // Truncated data.
        let p = scratch("short.npy");
        let mut bytes = header("<f4", &[10]);
        bytes.extend_from_slice(&[0u8; 12]);
        std::fs::write(&p, &bytes).unwrap();
        assert!(read(&p).is_err());
    }

    #[test]
    fn atomic_write_leaves_no_temp_file() {
        let p = scratch("atomic.npy");
        write(&p, &[3], &[1.0f32, 2.0, 3.0]).unwrap();
        let dir = p.parent().unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().contains(".tmp-")).collect();
        assert!(leftovers.is_empty());
        // A failing body leaves the old file in place.
        let r = write_atomic(&p, |_| bail!("simulated failure"));
        assert!(r.is_err());
        assert_eq!(read(&p).unwrap().data, Data::F4(vec![1.0, 2.0, 3.0]));
    }
}
