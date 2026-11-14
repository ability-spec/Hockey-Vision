//! Write values the way Python's `repr()` and `csv` module do, so files written by the Rust port
//! diff cleanly against the Python pipeline's.

/// `repr(float)`: shortest round-trip digits, scientific notation below 1e-4 and from 1e16 up.
pub fn py_float(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let a = v.abs();
    if !(1e-4..1e16).contains(&a) {
        let s = format!("{v:e}");
        let (mantissa, exp) = s.split_once('e').unwrap();
        let exp: i32 = exp.parse().unwrap();
        return format!(
            "{mantissa}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        );
    }
    let s = format!("{v}");
    if s.contains('.') {
        s
    } else {
        s + ".0"
    }
}

/// `py_float` appended to `out`, without an intermediate String for the common case.
pub fn push_py_float(out: &mut String, v: f64) {
    use std::fmt::Write;
    let a = v.abs();
    if v.is_finite() && v != 0.0 && (1e-4..1e16).contains(&a) {
        let start = out.len();
        write!(out, "{v}").unwrap();
        if !out[start..].contains('.') {
            out.push_str(".0");
        }
    } else {
        out.push_str(&py_float(v));
    }
}

/// `csv_field` appended to `out`.
pub fn push_csv_field(out: &mut String, s: &str) {
    if s.contains([',', '"', '\n', '\r']) {
        out.push('"');
        out.push_str(&s.replace('"', "\"\""));
        out.push('"');
    } else {
        out.push_str(s);
    }
}

/// A csv.writer field with QUOTE_MINIMAL.
pub fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_repr() {
        assert_eq!(py_float(1.0), "1.0");
        assert_eq!(py_float(-12.34), "-12.34");
        assert_eq!(py_float(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(py_float(1.5e-5), "1.5e-05");
        assert_eq!(py_float(-0.00012), "-0.00012");
        assert_eq!(py_float(1e16), "1e+16");
        assert_eq!(py_float(123456789.0), "123456789.0");
    }

    #[test]
    fn quotes_like_csv_module() {
        assert_eq!(csv_field("Zack Ostapchuk"), "Zack Ostapchuk");
        assert_eq!(csv_field("Smith, Jr."), "\"Smith, Jr.\"");
        assert_eq!(csv_field("a\"b"), "\"a\"\"b\"");
    }
}
