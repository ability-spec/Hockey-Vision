//! Homography estimation and point transforms, matching what `homography.py` gets from OpenCV.
//!
//! [`find_homography`] is a port of `cv2.findHomography(src, dst, 0)`: OpenCV's normalised DLT
//! (`HomographyEstimatorCallback::runKernel`) followed, for more than four points, by the
//! Levenberg-Marquardt refinement OpenCV 5 runs on it: `cv::LevMarq` (modules/geometry/src/
//! levmarq.cpp) over all nine entries of H, 10 iterations, geodesic acceleration, default
//! settings otherwise. Ported step for step so that fits agree with cv2's even on the badly
//! conditioned frames where a different (even a better converged) optimiser would drift.
//! Python passes `float32` points, so callers round to `f32` first ([`f32r`]) to see the same
//! inputs OpenCV does.

use nalgebra::{Matrix3, SMatrix, SVector, SymmetricEigen};

pub type Mat3 = Matrix3<f64>;
type Mat9 = SMatrix<f64, 9, 9>;
type Vec9 = SVector<f64, 9>;

/// Round to the nearest `f32`, as numpy's `np.float32(...)` does.
#[inline]
pub fn f32r(v: f64) -> f64 {
    v as f32 as f64
}

#[inline]
pub fn f32r2(p: [f64; 2]) -> [f64; 2] {
    [f32r(p[0]), f32r(p[1])]
}

/// `cv2.perspectiveTransform` of one point (float32 path: `w` compared with FLT_EPSILON and the
/// result stored as `f32`).
#[inline]
pub fn transform(h: &Mat3, p: [f64; 2]) -> [f64; 2] {
    let (x, y) = (f32r(p[0]), f32r(p[1]));
    let w = h[(2, 0)] * x + h[(2, 1)] * y + h[(2, 2)];
    if w.abs() > f32::EPSILON as f64 {
        let w = 1.0 / w;
        [
            f32r((h[(0, 0)] * x + h[(0, 1)] * y + h[(0, 2)]) * w),
            f32r((h[(1, 0)] * x + h[(1, 1)] * y + h[(1, 2)]) * w),
        ]
    } else {
        [0.0, 0.0]
    }
}

pub fn invert(h: &Mat3) -> Option<Mat3> {
    h.try_inverse()
}

/// Image -> rink homography from point pairs, as `cv2.findHomography(src, dst, 0)`.
/// Returns `None` when OpenCV would (fewer than 4 points, or no spread in a coordinate).
pub fn find_homography(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Mat3> {
    let n = src.len();
    if n < 4 || dst.len() != n {
        return None;
    }
    let mut h = dlt(src, dst)?;
    if n > 4 {
        let mut p = Vec9::from_iterator(h.transpose().iter().copied());
        levmarq(src, dst, &mut p, 10);
        h = scale_for(Mat3::new(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
        ));
    }
    Some(h)
}

/// OpenCV's normalised DLT (`runKernel`), H scaled so that H[2][2] = 1.
fn dlt(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Mat3> {
    let n = src.len() as f64;
    let (mut c_src, mut c_dst) = ([0.0f64; 2], [0.0f64; 2]);
    for (s, d) in src.iter().zip(dst) {
        c_src[0] += s[0];
        c_src[1] += s[1];
        c_dst[0] += d[0];
        c_dst[1] += d[1];
    }
    for v in c_src.iter_mut().chain(c_dst.iter_mut()) {
        *v /= n;
    }
    let (mut s_src, mut s_dst) = ([0.0f64; 2], [0.0f64; 2]);
    for (s, d) in src.iter().zip(dst) {
        s_src[0] += (s[0] - c_src[0]).abs();
        s_src[1] += (s[1] - c_src[1]).abs();
        s_dst[0] += (d[0] - c_dst[0]).abs();
        s_dst[1] += (d[1] - c_dst[1]).abs();
    }
    if s_src
        .iter()
        .chain(s_dst.iter())
        .any(|v| v.abs() < f64::EPSILON)
    {
        return None;
    }
    let s_dst = [n / s_dst[0], n / s_dst[1]];
    let s_src = [n / s_src[0], n / s_src[1]];

    let mut ltl = SMatrix::<f64, 9, 9>::zeros();
    for (s, d) in src.iter().zip(dst) {
        let (x, y) = ((d[0] - c_dst[0]) * s_dst[0], (d[1] - c_dst[1]) * s_dst[1]);
        let (sx, sy) = ((s[0] - c_src[0]) * s_src[0], (s[1] - c_src[1]) * s_src[1]);
        let lx = [sx, sy, 1.0, 0.0, 0.0, 0.0, -x * sx, -x * sy, -x];
        let ly = [0.0, 0.0, 0.0, sx, sy, 1.0, -y * sx, -y * sy, -y];
        for j in 0..9 {
            for k in j..9 {
                ltl[(j, k)] += lx[j] * lx[k] + ly[j] * ly[k];
            }
        }
    }
    for j in 0..9 {
        for k in 0..j {
            ltl[(j, k)] = ltl[(k, j)];
        }
    }
    let eig = SymmetricEigen::new(ltl);
    let smallest = eig.eigenvalues.imin();
    let v = eig.eigenvectors.column(smallest);
    let h0 = Mat3::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
    let inv_norm_dst = Mat3::new(
        1.0 / s_dst[0],
        0.0,
        c_dst[0],
        0.0,
        1.0 / s_dst[1],
        c_dst[1],
        0.0,
        0.0,
        1.0,
    );
    let norm_src = Mat3::new(
        s_src[0],
        0.0,
        -c_src[0] * s_src[0],
        0.0,
        s_src[1],
        -c_src[1] * s_src[1],
        0.0,
        0.0,
        1.0,
    );
    Some(scale_for(inv_norm_dst * h0 * norm_src))
}

/// H / H[2][2], unless H[2][2] is within FLT_EPSILON of 0 (OpenCV's `scaleFor`).
fn scale_for(h: Mat3) -> Mat3 {
    let h22 = h[(2, 2)];
    if h22.abs() > f32::EPSILON as f64 {
        h / h22
    } else {
        h
    }
}

/// Residuals (projected - dst) of the nine-entry H, and optionally the Jacobian, as OpenCV 5's
/// `homographyRefineCallback`.
fn residuals(
    src: &[[f64; 2]],
    dst: &[[f64; 2]],
    h: &Vec9,
    jac: Option<&mut Vec<[f64; 9]>>,
) -> Vec<f64> {
    let mut err = Vec::with_capacity(src.len() * 2);
    let mut rows = jac;
    if let Some(j) = rows.as_deref_mut() {
        j.clear();
    }
    for (s, d) in src.iter().zip(dst) {
        let (mx, my) = (s[0], s[1]);
        let ww = h[6] * mx + h[7] * my + h[8];
        let ww = if ww.abs() > f64::EPSILON {
            1.0 / ww
        } else {
            0.0
        };
        let xi = (h[0] * mx + h[1] * my + h[2]) * ww;
        let yi = (h[3] * mx + h[4] * my + h[5]) * ww;
        err.push(xi - d[0]);
        err.push(yi - d[1]);
        if let Some(j) = rows.as_deref_mut() {
            j.push([
                mx * ww,
                my * ww,
                ww,
                0.0,
                0.0,
                0.0,
                -mx * ww * xi,
                -my * ww * xi,
                -ww * xi,
            ]);
            j.push([
                0.0,
                0.0,
                0.0,
                mx * ww,
                my * ww,
                ww,
                -mx * ww * yi,
                -my * ww * yi,
                -ww * yi,
            ]);
        }
    }
    err
}

/// (J^T J, J^T r)
fn normal_equations(jac: &[[f64; 9]], r: &[f64]) -> (Mat9, Vec9) {
    let mut a = Mat9::zeros();
    let mut v = Vec9::zeros();
    for (row, &ri) in jac.iter().zip(r) {
        for i in 0..9 {
            v[i] += row[i] * ri;
            for k in i..9 {
                a[(i, k)] += row[i] * row[k];
            }
        }
    }
    for i in 0..9 {
        for k in 0..i {
            a[(i, k)] = a[(k, i)];
        }
    }
    (a, v)
}

fn jt_times(jac: &[[f64; 9]], r: &[f64]) -> Vec9 {
    let mut v = Vec9::zeros();
    for (row, &ri) in jac.iter().zip(r) {
        for i in 0..9 {
            v[i] += row[i] * ri;
        }
    }
    v
}

/// Solve the damped normal equations, as `cv::solve(a, b, DECOMP_SVD)` does. The LM diagonal is
/// clamped to at least 1e-6, so the matrix is positive definite and its Cholesky solution is the
/// SVD's (to rounding) at a fraction of the cost; SVD, with OpenCV's truncation of singular values
/// up to 2 DBL_EPSILON times their sum, is the fallback when rounding defeats Cholesky.
fn solve_svd(a: &Mat9, b: &Vec9) -> Option<Vec9> {
    if let Some(c) = a.cholesky() {
        let x = c.solve(b);
        if x.iter().all(|v| v.is_finite()) {
            return Some(x);
        }
    }
    let svd = a.svd(true, true);
    let eps = 2.0 * f64::EPSILON * svd.singular_values.sum();
    svd.solve(b, eps).ok()
}

/// `cv::LevMarq::optimize` with the settings findHomography uses: dense backend, 10 iterations,
/// geodesic acceleration (hGeo 1e-4, geoScale 0.5), everything else default.
fn levmarq(src: &[[f64; 2]], dst: &[[f64; 2]], x: &mut Vec9, max_iterations: usize) {
    const MIN_DIAG: f64 = 1e-6;
    const MAX_DIAG: f64 = 1e32;
    const MAX_LAMBDA: f64 = 1e32;
    const H_GEO: f64 = 1e-4;
    const GEO_SCALE: f64 = 0.5;
    const STEP_NORM_TOL: f64 = 1e-6;
    const REL_ENERGY_DELTA_TOL: f64 = 1e-6;
    const MIN_GRADIENT_TOL: f64 = 1e-6;
    const INITIAL_LAMBDA: f64 = 1e-4;
    const INITIAL_UP: f64 = 2.0;
    const INITIAL_DOWN: f64 = 3.0;

    let mut jac = Vec::with_capacity(src.len() * 2);
    let r = residuals(src, dst, x, Some(&mut jac));
    let mut old_energy = r.iter().map(|e| e * e).sum::<f64>();
    let (mut jtj, mut jtb) = normal_equations(&jac, &r);

    let mut up = INITIAL_UP;
    let mut lambda = INITIAL_LAMBDA;
    let mut iter = 0;
    let (mut small_gradient, mut small_step, mut small_energy_delta) = (false, false, false);
    let mut done = false;
    while !done {
        let gradient_max = jtb.amax();
        let diag = jtj.diagonal();
        let mut enough = false;
        while !enough && !done {
            let lm_diag = (diag * lambda).map(|v| v.clamp(MIN_DIAG, MAX_DIAG));
            let mut a = jtj;
            for i in 0..9 {
                a[(i, i)] = lm_diag[i] + diag[i];
            }
            let step = solve_svd(&a, &(-jtb));
            let mut cost_change = 0.0;
            let mut energy = 0.0;
            let (mut x_norm, mut quality) = (0.0, 0.0);
            let mut probe = *x;
            if let Some(mut s) = step {
                let jac_cost_change =
                    -0.5 * s.component_mul(&(jtb - lm_diag.component_mul(&s))).sum();
                x_norm = s.norm_squared();
                // Geodesic acceleration: second-order correction along the step.
                let geo_x = *x + s * H_GEO;
                let b_v = residuals(src, dst, &geo_x, None);
                let jtbv = jt_times(&jac, &b_v);
                let jtrvv = (jtbv + jtb * (H_GEO - 1.0) + lm_diag.component_mul(&s) * H_GEO)
                    / (H_GEO * H_GEO);
                if let Some(xgeo) = solve_svd(&a, &(-jtrvv)) {
                    if (xgeo.dot(&xgeo) / s.dot(&s)).sqrt() < 1.0 {
                        s += xgeo * GEO_SCALE;
                    }
                }
                probe = *x + s;
                energy = residuals(src, dst, &probe, None)
                    .iter()
                    .map(|e| e * e)
                    .sum::<f64>();
                if energy.is_nan() {
                    return; // OpenCV gives up and keeps the current x
                }
                cost_change = old_energy - energy;
                quality = cost_change / jac_cost_change;
            }
            if step.is_none() || cost_change < 0.0 {
                lambda *= up;
                up *= 2.0;
            } else {
                enough = true;
                lambda *= (1.0 / INITIAL_DOWN).max(1.0 - (2.0 * quality - 1.0).powi(3));
                up = INITIAL_UP;
                small_gradient = gradient_max < MIN_GRADIENT_TOL;
                small_step = x_norm < STEP_NORM_TOL;
                small_energy_delta = cost_change / energy < REL_ENERGY_DELTA_TOL;
                *x = probe;
                old_energy = energy;
            }
            iter += 1;
            done = iter >= max_iterations
                || lambda >= MAX_LAMBDA
                || small_gradient
                || small_step
                || small_energy_delta;
        }
        if !done {
            let r = residuals(src, dst, x, Some(&mut jac));
            (jtj, jtb) = normal_equations(&jac, &r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(h: &Mat3, p: [f64; 2]) -> [f64; 2] {
        let w = h[(2, 0)] * p[0] + h[(2, 1)] * p[1] + h[(2, 2)];
        [
            (h[(0, 0)] * p[0] + h[(0, 1)] * p[1] + h[(0, 2)]) / w,
            (h[(1, 0)] * p[0] + h[(1, 1)] * p[1] + h[(1, 2)]) / w,
        ]
    }

    #[test]
    fn recovers_an_exact_homography() {
        let truth = Mat3::new(0.12, 0.01, -80.0, 0.002, -0.3, 120.0, 0.0001, 0.0012, 1.0);
        let src: Vec<[f64; 2]> = [
            (100.0, 600.0),
            (1200.0, 650.0),
            (640.0, 300.0),
            (300.0, 200.0),
            (1000.0, 220.0),
            (700.0, 500.0),
        ]
        .iter()
        .map(|&(x, y)| [x, y])
        .collect();
        let dst: Vec<[f64; 2]> = src.iter().map(|&p| apply(&truth, p)).collect();
        let h = find_homography(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            let p = apply(&h, *s);
            assert!(
                (p[0] - d[0]).abs() < 1e-6 && (p[1] - d[1]).abs() < 1e-6,
                "{p:?} vs {d:?}"
            );
        }
        assert!((h[(2, 2)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn collinear_spread_is_rejected() {
        let src = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let dst = [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]];
        assert!(find_homography(&src, &dst).is_none());
    }
}

#[cfg(test)]
mod opencv_parity {
    use super::*;

    /// Noisy points and cv2.findHomography(src, dst, 0)'s result for them (OpenCV 4.x / 5.0).
    #[test]
    fn matches_cv2_find_homography_on_noisy_points() {
        let src = [
            [109.63093566894531, 170.50357055664062],
            [1025.63134765625, 419.15667724609375],
            [120.48466491699219, 311.8514099121094],
            [613.1856689453125, 115.01201629638672],
            [940.2587280273438, 81.84385681152344],
            [500.7720947265625, 372.05291748046875],
            [551.203857421875, 422.4949645996094],
        ];
        let dst = [
            [-55.69783020019531, 56.03811264038086],
            [30.40382957458496, -2.778958797454834],
            [-43.11418151855469, 18.850284576416016],
            [-4.343588352203369, 75.40080261230469],
            [29.31399917602539, 80.6236801147461],
            [-11.181835174560547, 7.352626323699951],
            [-2.294921636581421, -4.153625011444092],
        ];
        let cv = Mat3::new(
            0.13383729847010398,
            0.022886224960614106,
            -90.57811679308874,
            0.0012894576489418884,
            -0.32259764242865424,
            129.47225286913326,
            0.00014216808368159014,
            0.0016687749754581939,
            1.0,
        );
        let h = find_homography(&src, &dst).unwrap();
        // Compare where the two put the points, in rink feet.
        for p in &src {
            let (a, b) = (transform(&h, *p), transform(&cv, *p));
            assert!(
                (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4,
                "{a:?} vs {b:?}\n{h}"
            );
        }
    }
}
