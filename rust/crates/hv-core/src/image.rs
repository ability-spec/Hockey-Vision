//! 8-bit BGR images and the OpenCV / FFmpeg pixel operations run_models.py depends on, reproduced
//! bit for bit (each checked against cv2 on real frames, see rust/bench/check_pixels.py):
//!
//! - `YuvToBgr`: libswscale's C yuv420p -> bgr24 converter, as OpenCV's FFmpeg backend decodes
//!   (BT.601 coefficients whatever the stream says, limited range, chroma shared per 2x2 block).
//! - `resize_linear`: cv2.resize(INTER_LINEAR) for 8-bit images: fixed-point coefficients and the
//!   rounding of OpenCV's vectorised vertical pass.
//! - `resize_area_int`: cv2.resize(INTER_AREA) for whole-number shrink factors.
//! - `letterbox`: Ultralytics' LetterBox + its preprocess (BGR -> RGB, CHW, / 255).
//! - `CutDetector`: video_io.CutDetector (HSV histogram correlation between frames).

#[derive(Clone, Debug, Default)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    /// BGR, row-major, no padding.
    pub data: Vec<u8>,
}

impl Image {
    pub fn new(w: usize, h: usize) -> Self {
        Image { w, h, data: vec![0; w * h * 3] }
    }

    /// `frame[y1:y2, x1:x2]` (coordinates already inside the image).
    pub fn crop(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> Image {
        let (w, h) = (x2 - x1, y2 - y1);
        let mut out = Vec::with_capacity(w * h * 3);
        for y in y1..y2 {
            out.extend_from_slice(&self.data[(y * self.w + x1) * 3..(y * self.w + x2) * 3]);
        }
        Image { w, h, data: out }
    }
}

// ---- yuv420p -> bgr24 (libswscale yuv2rgb.c, C path)

const TABLE_HEADROOM: i64 = 512;
const LUMA_HEADROOM: i64 = 512;

/// swscale's lookup tables for one source range / matrix.
pub struct YuvToBgr {
    ytab: Vec<u8>,
    rv: Vec<i64>,
    gu: Vec<i64>,
    bu: Vec<i64>,
    gv: Vec<i64>,
}

impl YuvToBgr {
    /// Tables for `inv` (ff_yuv2rgb_coeffs row) and range; `bt601_limited()` is what OpenCV uses.
    pub fn new(inv: [i64; 4], full_range: bool) -> Self {
        let (contrast, saturation, brightness) = (1i64 << 16, 1i64 << 16, 0i64);
        let (mut crv, mut cbu, mut cgu, mut cgv) = (inv[0], inv[1], -inv[2], -inv[3]);
        let (mut cy, mut oy) = (1i64 << 16, 0i64);
        if !full_range {
            cy = (cy * 255) / 219;
            oy = 16 << 16;
        } else {
            crv = (crv * 224) / 255;
            cbu = (cbu * 224) / 255;
            cgu = (cgu * 224) / 255;
            cgv = (cgv * 224) / 255;
        }
        cy = (cy * contrast) >> 16;
        crv = (crv * contrast * saturation) >> 32;
        cbu = (cbu * contrast * saturation) >> 32;
        cgu = (cgu * contrast * saturation) >> 32;
        cgv = (cgv * contrast * saturation) >> 32;
        oy -= 256 * brightness;
        let d = cy.max(1);
        // C division truncates toward zero, as Rust's does.
        crv = (crv * (1 << 16) + 0x8000) / d;
        cbu = (cbu * (1 << 16) + 0x8000) / d;
        cgu = (cgu * (1 << 16) + 0x8000) / d;
        cgv = (cgv * (1 << 16) + 0x8000) / d;

        let size = 1024 + 2 * LUMA_HEADROOM;
        let yoffs = if full_range { 384 } else { 326 } + LUMA_HEADROOM;
        let mut yb = -(384 << 16) - LUMA_HEADROOM * cy - oy;
        let mut ytab = Vec::with_capacity(size as usize);
        for _ in 0..size {
            ytab.push(((yb + 0x8000) >> 16).clamp(0, 255) as u8);
            yb += cy;
        }
        let n = (256 + 2 * TABLE_HEADROOM) as usize;
        let clip = |i: usize| (i as i64 - TABLE_HEADROOM).clamp(0, 255);
        let fill = |inc: i64| -> Vec<i64> {
            let base = yoffs - (inc >> 9);
            (0..n).map(|i| base + ((clip(i) * inc) >> 16)).collect()
        };
        let fill_gv = |inc: i64| -> Vec<i64> {
            let off = -(inc >> 9);
            (0..n).map(|i| off + ((clip(i) * inc) >> 16)).collect()
        };
        YuvToBgr { rv: fill(crv), gu: fill(cgu), bu: fill(cbu), gv: fill_gv(cgv), ytab }
    }

    /// SWS_CS_DEFAULT (ITU-R BT.601), limited range: what OpenCV's VideoCapture gives for yuv420p.
    pub fn bt601_limited() -> Self {
        Self::new([104597, 132201, 25675, 53279], false)
    }

    /// One yuv420p frame (planes back to back, even width and height) into `out`.
    pub fn convert(&self, yuv: &[u8], w: usize, h: usize, out: &mut Image) {
        let (ys, cw) = (w * h, w / 2);
        let (yp, up, vp) = (&yuv[..ys], &yuv[ys..ys + ys / 4], &yuv[ys + ys / 4..ys + ys / 2]);
        out.w = w;
        out.h = h;
        out.data.resize(w * h * 3, 0);
        let hr = TABLE_HEADROOM as usize;
        for y in 0..h {
            let crow = (y / 2) * cw;
            let yrow = &yp[y * w..(y + 1) * w];
            let orow = &mut out.data[y * w * 3..(y + 1) * w * 3];
            for cx in 0..cw {
                let (u, v) = (up[crow + cx] as usize + hr, vp[crow + cx] as usize + hr);
                let (r, g, b) = (self.rv[v], self.gu[u] + self.gv[v], self.bu[u]);
                for k in 0..2 {
                    let x = cx * 2 + k;
                    let yv = yrow[x] as i64;
                    orow[x * 3] = self.ytab[(b + yv) as usize];
                    orow[x * 3 + 1] = self.ytab[(g + yv) as usize];
                    orow[x * 3 + 2] = self.ytab[(r + yv) as usize];
                }
            }
        }
    }
}

// ---- cv2.resize(INTER_LINEAR), 8-bit

const COEF_SCALE: f32 = 2048.0;

/// Per destination index: (source index, weight of it, weight of the next one), and the first
/// index from which the next source pixel is out of range (OpenCV's xmax).
fn linear_coefs(ssize: usize, dsize: usize, clamp: bool) -> (Vec<(i64, i32, i32)>, usize) {
    let scale = 1.0 / (dsize as f64 / ssize as f64);
    let mut out = Vec::with_capacity(dsize);
    let mut xmax = dsize;
    for d in 0..dsize {
        let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
        let mut s = f.floor() as i64;
        let mut f = f - s as f32;
        if clamp {
            if s < 0 {
                f = 0.0;
                s = 0;
            }
            if s + 1 >= ssize as i64 {
                xmax = xmax.min(d);
                if s >= ssize as i64 - 1 {
                    f = 0.0;
                    s = ssize as i64 - 1;
                }
            }
        }
        let a0 = ((1.0f32 - f) * COEF_SCALE).round_ties_even() as i32;
        let a1 = (f * COEF_SCALE).round_ties_even() as i32;
        out.push((s, a0, a1));
    }
    (out, xmax)
}

/// cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR).
pub fn resize_linear(src: &Image, dw: usize, dh: usize) -> Image {
    if dw == src.w && dh == src.h {
        return src.clone();
    }
    let (xc, xmax) = linear_coefs(src.w, dw, true);
    let (yc, _) = linear_coefs(src.h, dh, false);
    let rowlen = dw * 3;
    // Horizontal pass, computed lazily per source row (each is used by up to two output rows).
    let hrow = |sy: usize, buf: &mut Vec<i32>| {
        buf.clear();
        let row = &src.data[sy * src.w * 3..(sy + 1) * src.w * 3];
        for (dx, &(sx, a0, a1)) in xc.iter().enumerate() {
            let sx = sx as usize;
            for c in 0..3 {
                let mut v = row[sx * 3 + c] as i32 * a0;
                if dx < xmax {
                    v += row[(sx + 1) * 3 + c] as i32 * a1;
                }
                buf.push(v);
            }
        }
    };
    let mut out = Image::new(dw, dh);
    let (mut b0buf, mut b1buf) = (Vec::with_capacity(rowlen), Vec::with_capacity(rowlen));
    let mut cached: [Option<usize>; 2] = [None, None];
    let last = src.h as i64 - 1;
    for (dy, &(sy, b0, b1)) in yc.iter().enumerate() {
        let r0 = sy.clamp(0, last) as usize;
        let r1 = (sy + 1).clamp(0, last) as usize;
        // Reuse rows computed for the previous output row.
        if cached[0] != Some(r0) {
            if cached[1] == Some(r0) {
                std::mem::swap(&mut b0buf, &mut b1buf);
                cached[0] = Some(r0);
                cached[1] = None;
            } else {
                hrow(r0, &mut b0buf);
                cached[0] = Some(r0);
            }
        }
        if cached[1] != Some(r1) {
            if r1 == r0 {
                b1buf.clone_from(&b0buf);
            } else {
                hrow(r1, &mut b1buf);
            }
            cached[1] = Some(r1);
        }
        let orow = &mut out.data[dy * rowlen..(dy + 1) * rowlen];
        for x in 0..rowlen {
            // OpenCV's vectorised VResizeLinear (used for every pixel, including the row tail).
            let s0 = (b0buf[x] >> 4) as i32;
            let s1 = (b1buf[x] >> 4) as i32;
            let v = ((b0 * s0) >> 16) + ((b1 * s1) >> 16);
            orow[x] = ((v + 2) >> 2).clamp(0, 255) as u8;
        }
    }
    out
}

/// cv2.resize(INTER_AREA) by whole-number factors (fx, fy): each output pixel the mean of an
/// fx-by-fy block, rounded half to even.
pub fn resize_area_int(src: &Image, fx: usize, fy: usize) -> Image {
    let (dw, dh) = (src.w / fx, src.h / fy);
    let mut out = Image::new(dw, dh);
    let area = (fx * fy) as f64;
    for dy in 0..dh {
        for dx in 0..dw {
            for c in 0..3 {
                let mut sum = 0u32;
                for y in dy * fy..(dy + 1) * fy {
                    let row = &src.data[(y * src.w + dx * fx) * 3..];
                    for k in 0..fx {
                        sum += row[k * 3 + c] as u32;
                    }
                }
                out.data[(dy * dw + dx) * 3 + c] = (sum as f64 / area).round_ties_even() as u8;
            }
        }
    }
    out
}

// ---- Ultralytics LetterBox + preprocess

#[derive(Clone, Copy, Debug)]
pub struct Letterbox {
    /// Input size the model sees (h, w).
    pub shape: (usize, usize),
    /// Resized image size before padding (w, h).
    pub unpad: (usize, usize),
    pub top: usize,
    pub left: usize,
}

/// Python's round() of a float to an int (half to even).
fn py_round_int(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// LetterBox(new_shape, auto, stride) parameters for an image of (w, h).
pub fn letterbox_params(w: usize, h: usize, new_shape: (usize, usize), auto: bool, stride: usize) -> Letterbox {
    let (nh, nw) = (new_shape.0 as f64, new_shape.1 as f64);
    let r = (nh / h as f64).min(nw / w as f64);
    let unpad = (py_round_int(w as f64 * r) as usize, py_round_int(h as f64 * r) as usize);
    let (mut dw, mut dh) = (new_shape.1 as i64 - unpad.0 as i64, new_shape.0 as i64 - unpad.1 as i64);
    if auto {
        dw = dw.rem_euclid(stride as i64);
        dh = dh.rem_euclid(stride as i64);
    }
    let (dw, dh) = (dw as f64 / 2.0, dh as f64 / 2.0);
    let (top, bottom) = (py_round_int(dh - 0.1), py_round_int(dh + 0.1));
    let (left, right) = (py_round_int(dw - 0.1), py_round_int(dw + 0.1));
    let shape = (unpad.1 + (top + bottom) as usize, unpad.0 + (left + right) as usize);
    Letterbox { shape, unpad, top: top as usize, left: left as usize }
}

/// Letterbox `img` and append it to `out` as the model's input: RGB, CHW, float32 / 255,
/// padding 114.
pub fn letterbox_into(img: &Image, lb: &Letterbox, out: &mut Vec<f32>) {
    let resized;
    let src = if (img.w, img.h) != lb.unpad {
        resized = resize_linear(img, lb.unpad.0, lb.unpad.1);
        &resized
    } else {
        img
    };
    let (sh, sw) = lb.shape;
    let pad = 114.0f32 / 255.0;
    let base = out.len();
    out.resize(base + 3 * sh * sw, pad);
    for y in 0..src.h {
        let row = &src.data[y * src.w * 3..(y + 1) * src.w * 3];
        let oy = y + lb.top;
        for x in 0..src.w {
            let o = oy * sw + x + lb.left;
            // BGR -> RGB planes
            out[base + o] = row[x * 3 + 2] as f32 / 255.0;
            out[base + sh * sw + o] = row[x * 3 + 1] as f32 / 255.0;
            out[base + 2 * sh * sw + o] = row[x * 3] as f32 / 255.0;
        }
    }
}

// ---- camera cuts (video_io.CutDetector)

pub const CUT_CORRELATION: f64 = 0.7;
const HSV_SHIFT: i32 = 12;

/// OpenCV's 8-bit BGR->HSV saturation divisors. Rounded (255 << 12) / v, except six entries one
/// lower: what cv2.cvtColor actually computes, derived from all 16.7M colours.
fn sdiv_table() -> [i32; 256] {
    let mut t = [0i32; 256];
    for (v, slot) in t.iter_mut().enumerate().skip(1) {
        *slot = ((255 << HSV_SHIFT) as f64 / v as f64).round_ties_even() as i32;
    }
    for v in [2, 4, 14, 18, 118, 209] {
        t[v] -= 1;
    }
    t
}

fn hdiv_table() -> [i32; 256] {
    let mut t = [0i32; 256];
    for (v, slot) in t.iter_mut().enumerate().skip(1) {
        *slot = ((180 << HSV_SHIFT) as f64 / (6.0 * v as f64)).round_ties_even() as i32;
    }
    t
}

/// Camera cuts from hue/saturation histograms, as video_io.CutDetector.
pub struct CutDetector {
    prev: Option<Vec<f32>>,
    sdiv: [i32; 256],
    hdiv: [i32; 256],
    hbin: [usize; 256],
}

impl Default for CutDetector {
    fn default() -> Self {
        let mut hbin = [0usize; 256];
        for (i, b) in hbin.iter_mut().enumerate() {
            *b = (i as f64 * (32.0 / 180.0)).floor() as usize;
        }
        CutDetector { prev: None, sdiv: sdiv_table(), hdiv: hdiv_table(), hbin }
    }
}

impl CutDetector {
    /// The 32x32 hue/saturation histogram of a quarter-size frame, L2-normalised.
    pub fn histogram(&self, frame: &Image) -> Vec<f32> {
        let small = resize_area_int(frame, 4, 4);
        let mut hist = vec![0f32; 32 * 32];
        let half = 1 << (HSV_SHIFT - 1);
        for px in small.data.chunks_exact(3) {
            let (b, g, r) = (px[0] as i32, px[1] as i32, px[2] as i32);
            let v = b.max(g).max(r);
            let diff = v - b.min(g).min(r);
            let s = (diff * self.sdiv[v as usize] + half) >> HSV_SHIFT;
            let mut h = if v == r {
                g - b
            } else if v == g {
                b - r + 2 * diff
            } else {
                r - g + 4 * diff
            };
            h = (h * self.hdiv[diff as usize] + half) >> HSV_SHIFT;
            if h < 0 {
                h += 180;
            }
            hist[self.hbin[h as usize] * 32 + (s as usize >> 3)] += 1.0;
        }
        let norm = hist.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
        if norm > 0.0 {
            let scale = 1.0 / norm;
            for v in &mut hist {
                *v = (*v as f64 * scale) as f32;
            }
        }
        hist
    }

    /// True when `frame` starts a new shot.
    pub fn update(&mut self, frame: &Image) -> bool {
        let hist = self.histogram(frame);
        let cut = self.prev.as_ref().is_some_and(|p| correlation(p, &hist) < CUT_CORRELATION);
        self.prev = Some(hist);
        cut
    }
}

/// cv2.compareHist(HISTCMP_CORREL).
pub fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let (mut s1, mut s2, mut s11, mut s22, mut s12) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        s12 += x * y;
        s1 += x;
        s11 += x * x;
        s2 += y;
        s22 += y * y;
    }
    let scale = 1.0 / a.len() as f64;
    let num = s12 - s1 * s2 * scale;
    let denom2 = (s11 - s1 * s1 * scale) * (s22 - s2 * s2 * scale);
    if denom2.abs() > f64::EPSILON {
        num / denom2.sqrt()
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_like_ultralytics() {
        // 640x360 frame, rect mode: 640x384 with 12 px above and below.
        let lb = letterbox_params(640, 360, (640, 640), true, 32);
        assert_eq!((lb.shape, lb.unpad, lb.top, lb.left), ((384, 640), (640, 360), 12, 0));
        // A 127x70 crop, square: upscaled to 640x353, padded to 640x640.
        let lb = letterbox_params(127, 70, (640, 640), false, 32);
        assert_eq!((lb.shape, lb.unpad), ((640, 640), (640, 353)));
    }

    #[test]
    fn integer_downscale_is_sampling() {
        // 3x linear downscale picks pixel 3x+1, as OpenCV does.
        let mut img = Image::new(6, 3);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = i as u8;
        }
        let out = resize_linear(&img, 2, 1);
        assert_eq!(&out.data[..3], &img.data[(6 + 1) * 3..(6 + 1) * 3 + 3]);
    }
}
