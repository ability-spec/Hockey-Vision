//! Python values, for outputs that must match the Python pipeline byte for byte: an ordered dict
//! (`Obj`), `json.dumps` (ensure_ascii, with or without indent), the `csv` module's cells, and the
//! CPython 3.9 arithmetic whose last bit the outputs depend on (`round`, `math.hypot`, `//`, `%`).
//!
//! Python keeps ints and floats apart (`0` vs `0.0` in a CSV), and some outputs depend on which one a
//! sum started from, so `Val` does too.

use crate::pyfmt::{csv_field, py_float};

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Val>),
    Obj(Obj),
}

/// A Python dict: keys in insertion order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Obj(pub Vec<(String, Val)>);

static NONE: Val = Val::None;

impl Obj {
    pub fn new() -> Self {
        Obj(Vec::new())
    }
    /// `d[k] = v`: replaces in place, or appends a new key.
    pub fn set(&mut self, k: &str, v: impl Into<Val>) {
        let v = v.into();
        match self.0.iter_mut().find(|(key, _)| key == k) {
            Some(slot) => slot.1 = v,
            None => self.0.push((k.to_string(), v)),
        }
    }
    /// Builder for a new dict: appends `k` (callers pass distinct keys).
    pub fn with(mut self, k: &str, v: impl Into<Val>) -> Self {
        debug_assert!(!self.contains(k), "duplicate key {k}");
        self.0.push((k.to_string(), v.into()));
        self
    }
    /// `d.get(k)`, None when missing.
    pub fn get(&self, k: &str) -> &Val {
        self.0
            .iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v)
            .unwrap_or(&NONE)
    }
    pub fn contains(&self, k: &str) -> bool {
        self.0.iter().any(|(key, _)| key == k)
    }
}

impl Val {
    pub fn is_none(&self) -> bool {
        matches!(self, Val::None)
    }
    /// Python truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            Val::None => false,
            Val::Bool(b) => *b,
            Val::Int(i) => *i != 0,
            Val::Float(f) => *f != 0.0,
            Val::Str(s) => !s.is_empty(),
            Val::List(l) => !l.is_empty(),
            Val::Obj(o) => !o.0.is_empty(),
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Val::Int(i) => Some(*i as f64),
            Val::Float(f) => Some(*f),
            Val::Bool(b) => Some(*b as i64 as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }
    /// `a + b` for Python numbers (bools count as ints).
    pub fn add(&self, other: &Val) -> Val {
        match (self.int(), other.int()) {
            (Some(a), Some(b)) => Val::Int(a + b),
            _ => Val::Float(self.as_f64().unwrap_or(0.0) + other.as_f64().unwrap_or(0.0)),
        }
    }
    fn int(&self) -> Option<i64> {
        match self {
            Val::Int(i) => Some(*i),
            Val::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }
    /// `round(v, n)`: ints stay ints.
    pub fn round(&self, n: i32) -> Val {
        match self {
            Val::Float(f) => Val::Float(py_round(*f, n)),
            other => other.clone(),
        }
    }
    /// A cell as `csv.writer` writes it.
    pub fn csv(&self) -> String {
        match self {
            Val::None => String::new(),
            Val::Bool(b) => if *b { "True" } else { "False" }.into(),
            Val::Int(i) => i.to_string(),
            Val::Float(f) => py_float(*f),
            Val::Str(s) => csv_field(s),
            other => csv_field(&dumps(other)),
        }
    }
}

impl From<f64> for Val {
    fn from(v: f64) -> Self {
        Val::Float(v)
    }
}
impl From<i64> for Val {
    fn from(v: i64) -> Self {
        Val::Int(v)
    }
}
impl From<usize> for Val {
    fn from(v: usize) -> Self {
        Val::Int(v as i64)
    }
}
impl From<bool> for Val {
    fn from(v: bool) -> Self {
        Val::Bool(v)
    }
}
impl From<&str> for Val {
    fn from(v: &str) -> Self {
        Val::Str(v.to_string())
    }
}
impl From<String> for Val {
    fn from(v: String) -> Self {
        Val::Str(v)
    }
}
impl From<&String> for Val {
    fn from(v: &String) -> Self {
        Val::Str(v.clone())
    }
}
impl From<Obj> for Val {
    fn from(v: Obj) -> Self {
        Val::Obj(v)
    }
}
impl From<Vec<Val>> for Val {
    fn from(v: Vec<Val>) -> Self {
        Val::List(v)
    }
}
impl<T: Into<Val>> From<Option<T>> for Val {
    fn from(v: Option<T>) -> Self {
        v.map_or(Val::None, Into::into)
    }
}
impl From<&Val> for Val {
    fn from(v: &Val) -> Self {
        v.clone()
    }
}

// ---- json.dumps

/// `json.dumps(v)`: separators ", " and ": ".
pub fn dumps(v: &Val) -> String {
    let mut s = String::new();
    write_json(&mut s, v, None, 0);
    s
}

/// `json.dumps(v, indent=n)`.
pub fn dumps_indent(v: &Val, indent: usize) -> String {
    let mut s = String::new();
    write_json(&mut s, v, Some(indent), 0);
    s
}

fn newline(s: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        s.push('\n');
        s.extend(std::iter::repeat(' ').take(n * level));
    }
}

fn write_json(s: &mut String, v: &Val, indent: Option<usize>, level: usize) {
    let sep = if indent.is_some() { "," } else { ", " };
    match v {
        Val::None => s.push_str("null"),
        Val::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
        Val::Int(i) => s.push_str(&i.to_string()),
        Val::Float(f) => s.push_str(&json_float(*f)),
        Val::Str(t) => json_str(s, t),
        Val::List(items) => {
            if items.is_empty() {
                s.push_str("[]");
                return;
            }
            s.push('[');
            for (k, item) in items.iter().enumerate() {
                if k > 0 {
                    s.push_str(sep);
                }
                newline(s, indent, level + 1);
                write_json(s, item, indent, level + 1);
            }
            newline(s, indent, level);
            s.push(']');
        }
        Val::Obj(o) => {
            if o.0.is_empty() {
                s.push_str("{}");
                return;
            }
            s.push('{');
            for (k, (key, item)) in o.0.iter().enumerate() {
                if k > 0 {
                    s.push_str(sep);
                }
                newline(s, indent, level + 1);
                json_str(s, key);
                s.push_str(": ");
                write_json(s, item, indent, level + 1);
            }
            newline(s, indent, level);
            s.push('}');
        }
    }
}

fn json_float(f: f64) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        py_float(f)
    }
}

/// A JSON string with ensure_ascii, as Python's encoder escapes it.
fn json_str(s: &mut String, t: &str) {
    s.push('"');
    for c in t.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            '\u{8}' => s.push_str("\\b"),
            '\u{c}' => s.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    s.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => s.push(c),
        }
    }
    s.push('"');
}

// ---- CPython arithmetic

/// `round(x, n)` for a float: the exact binary value rounded half-to-even to n decimals (Rust's
/// `{:.n}` does the same), read back.
pub fn py_round(x: f64, n: i32) -> f64 {
    if !x.is_finite() || n > 300 {
        return x;
    }
    // Fast path: x is already the double nearest some k / 10**n (e.g. a value Python rounded and
    // wrote), so rounding it again gives x back.
    if (0..=15).contains(&n) {
        let p = 10f64.powi(n);
        let k = (x * p).round();
        if k.abs() < 9e15 && k / p == x {
            return x;
        }
    }
    format!("{:.*}", n.max(0) as usize, x).parse().unwrap()
}

/// `f"{x:.nf}"`.
pub fn fmt_fixed(x: f64, n: usize) -> String {
    format!("{:.*}", n, x)
}

/// CPython 3.9's `math.hypot(x, y)` (vector_norm), which differs from libm's correctly rounded
/// hypot in the last bit about a third of the time. Checked bit for bit against Python by
/// `examples/check_pymath.rs`.
pub fn hypot(x: f64, y: f64) -> f64 {
    let v = [x.abs(), y.abs()];
    let found_nan = v[0].is_nan() || v[1].is_nan();
    let max = if v[1] > v[0] { v[1] } else { v[0] };
    if max.is_infinite() || v[0].is_infinite() || v[1].is_infinite() {
        return f64::INFINITY;
    }
    if found_nan {
        return f64::NAN;
    }
    if max == 0.0 {
        return max;
    }
    // 3.8 / 3.9: divide by the max (3.10 switched to power-of-two scaling and a correction step).
    let (mut csum, mut frac) = (1.0f64, 0.0f64);
    for x in v {
        let x = x / max;
        let x = x * x;
        let old = csum;
        csum += x;
        frac += (old - csum) + x;
    }
    max * (csum - 1.0 + frac).sqrt()
}

/// Python's `divmod(vx, wx)` for floats: (`vx // wx`, `vx % wx`).
pub fn py_divmod(vx: f64, wx: f64) -> (f64, f64) {
    let mut m = vx % wx;
    let mut div = (vx - m) / wx;
    if m != 0.0 {
        if (wx < 0.0) != (m < 0.0) {
            m += wx;
            div -= 1.0;
        }
    } else {
        m = 0.0f64.copysign(wx);
    }
    let floordiv = if div != 0.0 {
        let mut f = div.floor();
        if div - f > 0.5 {
            f += 1.0;
        }
        f
    } else {
        0.0f64.copysign(vx / wx)
    };
    (floordiv, m)
}

/// `datetime.now(timezone.utc).isoformat(timespec="seconds")`.
pub fn utc_now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_like_python() {
        assert_eq!(py_round(2.675, 2), 2.67);
        assert_eq!(py_round(0.125, 2), 0.12);
        assert_eq!(py_round(-0.04, 1).to_bits(), (-0.0f64).to_bits());
        assert_eq!(fmt_fixed(2.5, 0), "2");
    }

    #[test]
    fn divmod_like_python() {
        assert_eq!(py_divmod(125.5, 60.0), (2.0, 5.5));
        assert_eq!(py_divmod(59.99999999999999, 60.0).0, 0.0);
    }

    #[test]
    fn json_like_python() {
        let v: Val = Obj::new()
            .with("a", 1.0)
            .with("b", Val::List(vec![]))
            .with("c", "Slafkovský \"x\"")
            .with("d", Val::List(vec![Val::Int(1), Val::None]))
            .into();
        assert_eq!(
            dumps(&v),
            "{\"a\": 1.0, \"b\": [], \"c\": \"Slafkovsk\\u00fd \\\"x\\\"\", \"d\": [1, null]}"
        );
        assert_eq!(
            dumps_indent(&v, 2),
            "{\n  \"a\": 1.0,\n  \"b\": [],\n  \"c\": \"Slafkovsk\\u00fd \\\"x\\\"\",\n  \"d\": [\n    1,\n    null\n  ]\n}"
        );
    }
}
