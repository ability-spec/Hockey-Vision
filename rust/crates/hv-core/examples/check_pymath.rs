//! Check hv_core::pyval's hypot and divmod against CPython on a file of (x, y, hypot, t, t // 60,
//! t % 60) records written by Python (little-endian f64s).
fn main() {
    let path = std::env::args().nth(1).expect("usage: check_pymath <file>");
    let bytes = std::fs::read(path).unwrap();
    let v: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let (mut n, mut bad_h, mut bad_libm, mut bad_d) = (0, 0, 0, 0);
    for r in v.chunks_exact(6) {
        n += 1;
        bad_h += (hv_core::pyval::hypot(r[0], r[1]).to_bits() != r[2].to_bits()) as usize;
        bad_libm += (r[0].hypot(r[1]).to_bits() != r[2].to_bits()) as usize;
        let (d, m) = hv_core::pyval::py_divmod(r[3], 60.0);
        bad_d += (d.to_bits() != r[4].to_bits() || m.to_bits() != r[5].to_bits()) as usize;
    }
    println!("{n} values: hypot differs on {bad_h} (libm's hypot would differ on {bad_libm}), divmod on {bad_d}");
}
