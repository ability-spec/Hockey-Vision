//! Ultralytics' BYTETracker with bytetrack.yaml (high 0.25, low 0.1, new 0.25, buffer 30, match
//! 0.8, fuse_score), as model.track(persist=True) runs it: the same Kalman filter (float64), the
//! same float32 IoU costs and the same association order, so track ids and boxes match.
//!
//! lap.lapjv(extend_cost=True, cost_limit=t) is solved on the same extended cost matrix with an
//! exact (Hungarian) solver; the assignment is the same whenever the optimum is unique.

use crate::yolo::Det;

const TRACK_HIGH: f32 = 0.25;
const TRACK_LOW: f32 = 0.1;
const NEW_TRACK: f32 = 0.25;
const TRACK_BUFFER: i64 = 30;
const MATCH_THRESH: f64 = 0.8;
const REMOVED_BUFFER: usize = 1000;

const STD_POS: f64 = 1.0 / 20.0;
const STD_VEL: f64 = 1.0 / 160.0;

type Mean = [f64; 8];
type Cov = [[f64; 8]; 8];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    New,
    Tracked,
    Lost,
    Removed,
}

#[derive(Clone, Debug)]
struct STrack {
    tlwh: [f32; 4],
    mean: Option<Mean>,
    cov: Cov,
    is_activated: bool,
    score: f32,
    cls: f32,
    idx: usize,
    track_id: i64,
    state: State,
    frame_id: i64,
    start_frame: i64,
}

/// One tracked box out of the tracker, like a row of BYTETracker.update's output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackOut {
    pub xyxy: [f32; 4],
    pub track_id: i64,
    pub score: f32,
    pub cls: f32,
    /// Index of the detection it was matched to this frame.
    pub idx: usize,
}

fn xyah(tlwh: [f32; 4]) -> [f32; 4] {
    // tlwh_to_xyah on a float32 array
    let mut r = tlwh;
    r[0] += r[2] / 2.0;
    r[1] += r[3] / 2.0;
    r[2] /= r[3];
    r
}

impl STrack {
    fn new(d: &Det, idx: usize) -> Self {
        // Boxes.xywh (float32), then parse_bboxes' float64 array, then xywh2ltwh -> float32.
        let [x1, y1, x2, y2] = d.xyxy;
        let (cx, cy, w, h) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1);
        let (cx, cy, w, h) = (cx as f64, cy as f64, w as f64, h as f64);
        let tlwh = [(cx - w / 2.0) as f32, (cy - h / 2.0) as f32, w as f32, h as f32];
        STrack {
            tlwh,
            mean: None,
            cov: [[0.0; 8]; 8],
            is_activated: false,
            score: d.conf,
            cls: d.cls as f32,
            idx,
            track_id: 0,
            state: State::New,
            frame_id: 0,
            start_frame: 0,
        }
    }

    /// Current box (tlwh): the Kalman state once there is one, as float64.
    fn tlwh64(&self) -> [f64; 4] {
        match &self.mean {
            None => self.tlwh.map(|v| v as f64),
            Some(m) => {
                let mut r = [m[0], m[1], m[2], m[3]];
                r[2] *= r[3];
                r[0] -= r[2] / 2.0;
                r[1] -= r[3] / 2.0;
                r
            }
        }
    }

    /// xyxy as float32, as iou_distance and the output see it.
    fn xyxy32(&self) -> [f32; 4] {
        match &self.mean {
            None => {
                let t = self.tlwh;
                [t[0], t[1], t[2] + t[0], t[3] + t[1]]
            }
            Some(_) => {
                let t = self.tlwh64();
                [t[0] as f32, t[1] as f32, (t[2] + t[0]) as f32, (t[3] + t[1]) as f32]
            }
        }
    }

    fn activate(&mut self, frame_id: i64, next_id: &mut i64) {
        *next_id += 1;
        self.track_id = *next_id;
        let (m, c) = initiate(xyah(self.tlwh));
        self.mean = Some(m);
        self.cov = c;
        self.state = State::Tracked;
        if frame_id == 1 {
            self.is_activated = true;
        }
        self.frame_id = frame_id;
        self.start_frame = frame_id;
    }

    fn update_with(&mut self, det: &STrack, frame_id: i64) {
        self.frame_id = frame_id;
        let (m, c) = update(self.mean.as_ref().unwrap(), &self.cov, xyah(det.tlwh));
        self.mean = Some(m);
        self.cov = c;
        self.state = State::Tracked;
        self.is_activated = true;
        self.score = det.score;
        self.cls = det.cls;
        self.idx = det.idx;
    }
}

// ---- Kalman filter (KalmanFilterXYAH)

fn initiate(measurement: [f32; 4]) -> (Mean, Cov) {
    let m = measurement.map(|v| v as f64);
    let mean = [m[0], m[1], m[2], m[3], 0.0, 0.0, 0.0, 0.0];
    let h = m[3];
    let std = [
        2.0 * STD_POS * h,
        2.0 * STD_POS * h,
        1e-2,
        2.0 * STD_POS * h,
        10.0 * STD_VEL * h,
        10.0 * STD_VEL * h,
        1e-5,
        10.0 * STD_VEL * h,
    ];
    let mut cov = [[0.0; 8]; 8];
    for i in 0..8 {
        cov[i][i] = std[i] * std[i];
    }
    (mean, cov)
}

/// multi_predict for one track (F has ones on the diagonal and at (i, i + 4)).
fn predict(mean: &Mean, cov: &Cov) -> (Mean, Cov) {
    let h = mean[3];
    let std = [STD_POS * h, STD_POS * h, 1e-2, STD_POS * h, STD_VEL * h, STD_VEL * h, 1e-5, STD_VEL * h];
    let mut m = *mean;
    for i in 0..4 {
        m[i] = mean[i] + mean[i + 4];
    }
    // left = F @ cov
    let mut left = *cov;
    for i in 0..4 {
        for k in 0..8 {
            left[i][k] = cov[i][k] + cov[i + 4][k];
        }
    }
    // left @ F.T
    let mut c = left;
    for r in 0..8 {
        for i in 0..4 {
            c[r][i] = left[r][i] + left[r][i + 4];
        }
    }
    for i in 0..8 {
        c[i][i] += std[i] * std[i];
    }
    (m, c)
}

fn update(mean: &Mean, cov: &Cov, measurement: [f32; 4]) -> (Mean, Cov) {
    // project
    let h = mean[3];
    let std = [STD_POS * h, STD_POS * h, 1e-1, STD_POS * h];
    let mut s = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            s[i][j] = cov[i][j];
        }
        s[i][i] += std[i] * std[i];
    }
    // kalman_gain = solve(S, cov[:, :4].T).T  -> K (8x4) with S K^T = cov[:, :4]^T
    let mut rhs = [[0.0f64; 8]; 4]; // 4x8
    for i in 0..4 {
        for j in 0..8 {
            rhs[i][j] = cov[j][i];
        }
    }
    let x = solve4(s, rhs); // 4x8
    let mut k = [[0.0f64; 4]; 8];
    for i in 0..8 {
        for j in 0..4 {
            k[i][j] = x[j][i];
        }
    }
    let innovation: [f64; 4] = std::array::from_fn(|i| measurement[i] as f64 - mean[i]);
    let mut new_mean = *mean;
    for i in 0..8 {
        let mut acc = 0.0;
        for j in 0..4 {
            acc += innovation[j] * k[i][j];
        }
        new_mean[i] = mean[i] + acc;
    }
    // cov - K S K^T
    let mut ks = [[0.0f64; 4]; 8];
    for i in 0..8 {
        for j in 0..4 {
            let mut acc = 0.0;
            for l in 0..4 {
                acc += k[i][l] * s[l][j];
            }
            ks[i][j] = acc;
        }
    }
    let mut new_cov = *cov;
    for i in 0..8 {
        for j in 0..8 {
            let mut acc = 0.0;
            for l in 0..4 {
                acc += ks[i][l] * k[j][l];
            }
            new_cov[i][j] = cov[i][j] - acc;
        }
    }
    (new_mean, new_cov)
}

/// Solve A X = B (A 4x4, B 4x8) by LU with partial pivoting, as LAPACK's gesv does.
fn solve4(mut a: [[f64; 4]; 4], mut b: [[f64; 8]; 4]) -> [[f64; 8]; 4] {
    for col in 0..4 {
        let mut p = col;
        for r in col + 1..4 {
            if a[r][col].abs() > a[p][col].abs() {
                p = r;
            }
        }
        a.swap(col, p);
        b.swap(col, p);
        for r in col + 1..4 {
            let f = a[r][col] / a[col][col];
            for c in col..4 {
                a[r][c] -= f * a[col][c];
            }
            for c in 0..8 {
                b[r][c] -= f * b[col][c];
            }
        }
    }
    let mut x = [[0.0f64; 8]; 4];
    for c in 0..8 {
        for r in (0..4).rev() {
            let mut acc = b[r][c];
            for k in r + 1..4 {
                acc -= a[r][k] * x[k][c];
            }
            x[r][c] = acc / a[r][r];
        }
    }
    x
}

// ---- matching

/// 1 - IoU in float32 (matching.iou_distance via bbox_ioa(iou=True)).
fn iou_distance(a: &[[f32; 4]], b: &[[f32; 4]]) -> Vec<Vec<f32>> {
    let eps = 1e-7f32;
    a.iter()
        .map(|p| {
            let pa = (p[2] - p[0]) * (p[3] - p[1]);
            b.iter()
                .map(|q| {
                    let iw = (p[2].min(q[2]) - p[0].max(q[0])).max(0.0);
                    let ih = (p[3].min(q[3]) - p[1].max(q[1])).max(0.0);
                    let inter = iw * ih;
                    let area = (q[2] - q[0]) * (q[3] - q[1]);
                    let area = area + pa - inter;
                    1.0 - inter / (area + eps)
                })
                .collect()
        })
        .collect()
}

fn fuse_score(cost: &mut [Vec<f32>], scores: &[f32]) {
    for row in cost.iter_mut() {
        for (c, &s) in row.iter_mut().zip(scores) {
            let sim = 1.0 - *c;
            *c = 1.0 - sim * s;
        }
    }
}

/// matching.linear_assignment(cost, thresh): (matches, unmatched rows, unmatched cols).
fn linear_assignment(cost: &[Vec<f32>], n_cols: usize, thresh: f64) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let n_rows = cost.len();
    if n_rows == 0 || n_cols == 0 {
        return (vec![], (0..n_rows).collect(), (0..n_cols).collect());
    }
    // lap.lapjv's extended matrix: [[cost, limit/2], [limit/2, 0]].
    let n = n_rows + n_cols;
    let mut ext = vec![vec![thresh / 2.0; n]; n];
    for (i, row) in ext.iter_mut().enumerate().skip(n_rows) {
        for v in row.iter_mut().skip(n_cols) {
            *v = 0.0;
        }
        let _ = i;
    }
    for i in 0..n_rows {
        for j in 0..n_cols {
            ext[i][j] = cost[i][j] as f64;
        }
    }
    let x = hungarian(&ext);
    let mut col_of = vec![usize::MAX; n_rows];
    let mut row_of = vec![usize::MAX; n_cols];
    for i in 0..n_rows {
        if x[i] < n_cols {
            col_of[i] = x[i];
            row_of[x[i]] = i;
        }
    }
    let matches = (0..n_rows).filter(|&i| col_of[i] != usize::MAX).map(|i| (i, col_of[i])).collect();
    let ua = (0..n_rows).filter(|&i| col_of[i] == usize::MAX).collect();
    let ub = (0..n_cols).filter(|&j| row_of[j] == usize::MAX).collect();
    (matches, ua, ub)
}

/// Minimum-cost assignment of a square matrix: column for each row (Kuhn-Munkres with potentials).
fn hungarian(a: &[Vec<f64>]) -> Vec<usize> {
    let n = a.len();
    let inf = f64::INFINITY;
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    let (mut p, mut way) = (vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut ans = vec![0usize; n];
    for j in 1..=n {
        if p[j] > 0 {
            ans[p[j] - 1] = j - 1;
        }
    }
    ans
}

// ---- the tracker

#[derive(Default)]
pub struct ByteTracker {
    tracks: Vec<STrack>, // every track ever made; the lists below index into it
    tracked: Vec<usize>,
    lost: Vec<usize>,
    removed: Vec<usize>,
    frame_id: i64,
    next_id: i64,
}

fn joint(a: &[usize], b: &[usize], tracks: &[STrack]) -> Vec<usize> {
    let ids: Vec<i64> = a.iter().map(|&i| tracks[i].track_id).collect();
    let mut out = a.to_vec();
    out.extend(b.iter().filter(|&&i| !ids.contains(&tracks[i].track_id)));
    out
}

fn sub(a: &[usize], b: &[usize], tracks: &[STrack]) -> Vec<usize> {
    let ids: Vec<i64> = b.iter().map(|&i| tracks[i].track_id).collect();
    a.iter().copied().filter(|&i| !ids.contains(&tracks[i].track_id)).collect()
}

impl ByteTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// One frame's detections (in order); returns the activated tracks, as BYTETracker.update.
    pub fn update(&mut self, dets: &[Det]) -> Vec<TrackOut> {
        self.frame_id += 1;
        let fid = self.frame_id;
        let (mut activated, mut refind, mut lost_new, mut removed_new) = (vec![], vec![], vec![], vec![]);

        // Split detections into high and low score (dropping zero-size boxes).
        let (mut high, mut low) = (vec![], vec![]);
        for (i, d) in dets.iter().enumerate() {
            let (w, h) = (d.xyxy[2] - d.xyxy[0], d.xyxy[3] - d.xyxy[1]);
            if !(w > 0.0 && h > 0.0) {
                continue;
            }
            if d.conf >= TRACK_HIGH {
                high.push(STrack::new(d, i));
            } else if d.conf > TRACK_LOW {
                low.push(STrack::new(d, i));
            }
        }
        // Detections become tracks in self.tracks when they are activated; until then they live here.
        let (unconfirmed, tracked_now): (Vec<usize>, Vec<usize>) =
            self.tracked.iter().partition(|&&i| !self.tracks[i].is_activated);
        let pool = joint(&tracked_now, &self.lost, &self.tracks);
        for &i in &pool {
            let t = &mut self.tracks[i];
            let mut m = t.mean.unwrap();
            if t.state != State::Tracked {
                m[7] = 0.0;
            }
            let (m, c) = predict(&m, &t.cov);
            t.mean = Some(m);
            t.cov = c;
        }

        // First association: pool vs high-score detections, IoU fused with score.
        let boxes = |ids: &[usize], tracks: &[STrack]| ids.iter().map(|&i| tracks[i].xyxy32()).collect::<Vec<_>>();
        let high_boxes: Vec<[f32; 4]> = high.iter().map(STrack::xyxy32).collect();
        let high_scores: Vec<f32> = high.iter().map(|d| d.score).collect();
        let mut dists = iou_distance(&boxes(&pool, &self.tracks), &high_boxes);
        fuse_score(&mut dists, &high_scores);
        let (matches, u_track, u_det) = linear_assignment(&dists, high.len(), MATCH_THRESH);
        for (it, id) in matches {
            let ti = pool[it];
            let t = &mut self.tracks[ti];
            if t.state == State::Tracked {
                t.update_with(&high[id], fid);
                activated.push(ti);
            } else {
                t.update_with(&high[id], fid); // re_activate(new_id=False): same fields
                refind.push(ti);
            }
        }

        // Second association: remaining tracked vs low-score detections, IoU only.
        let r_tracked: Vec<usize> = u_track.iter().map(|&i| pool[i]).filter(|&i| self.tracks[i].state == State::Tracked).collect();
        let u_track2: Vec<usize> = if !r_tracked.is_empty() && !low.is_empty() {
            let low_boxes: Vec<[f32; 4]> = low.iter().map(STrack::xyxy32).collect();
            let d = iou_distance(&boxes(&r_tracked, &self.tracks), &low_boxes);
            let (m, ut, _) = linear_assignment(&d, low.len(), 0.5);
            for (it, id) in m {
                let ti = r_tracked[it];
                let was_tracked = self.tracks[ti].state == State::Tracked;
                self.tracks[ti].update_with(&low[id], fid);
                if was_tracked { activated.push(ti) } else { refind.push(ti) }
            }
            ut
        } else {
            (0..r_tracked.len()).collect()
        };
        for it in u_track2 {
            let ti = r_tracked[it];
            if self.tracks[ti].state != State::Lost {
                self.tracks[ti].state = State::Lost;
                lost_new.push(ti);
            }
        }

        // Unconfirmed tracks (one frame old) vs the high-score detections left.
        let mut dets_left: Vec<STrack> = u_det.iter().map(|&i| high[i].clone()).collect();
        let u_det2: Vec<usize> = if unconfirmed.is_empty() {
            (0..dets_left.len()).collect()
        } else {
            let left_boxes: Vec<[f32; 4]> = dets_left.iter().map(STrack::xyxy32).collect();
            let left_scores: Vec<f32> = dets_left.iter().map(|d| d.score).collect();
            let mut d = iou_distance(&boxes(&unconfirmed, &self.tracks), &left_boxes);
            fuse_score(&mut d, &left_scores);
            let (m, uu, ud) = linear_assignment(&d, dets_left.len(), 0.7);
            for (it, id) in m {
                let ti = unconfirmed[it];
                self.tracks[ti].update_with(&dets_left[id], fid);
                activated.push(ti);
            }
            for it in uu {
                let ti = unconfirmed[it];
                self.tracks[ti].state = State::Removed;
                removed_new.push(ti);
            }
            ud
        };

        // New tracks.
        for inew in u_det2 {
            let mut t = std::mem::replace(&mut dets_left[inew], STrack::new(&Det { xyxy: [0.0; 4], conf: 0.0, cls: 0 }, 0));
            if t.score < NEW_TRACK {
                continue;
            }
            t.activate(fid, &mut self.next_id);
            self.tracks.push(t);
            activated.push(self.tracks.len() - 1);
        }

        // Lost too long.
        for &ti in &self.lost {
            if fid - self.tracks[ti].frame_id > TRACK_BUFFER {
                self.tracks[ti].state = State::Removed;
                removed_new.push(ti);
            }
        }

        // merge_track_pools
        let tracked: Vec<usize> = self.tracked.iter().copied().filter(|&i| self.tracks[i].state == State::Tracked).collect();
        let tracked = joint(&tracked, &activated, &self.tracks);
        let tracked = joint(&tracked, &refind, &self.tracks);
        let mut lost = sub(&self.lost, &tracked, &self.tracks);
        lost.extend(lost_new);
        let lost = sub(&lost, &self.removed, &self.tracks);
        let (tracked, lost) = self.remove_duplicates(tracked, lost);
        self.tracked = tracked;
        self.lost = lost;
        self.removed.extend(removed_new);
        if self.removed.len() > REMOVED_BUFFER {
            self.removed.drain(..self.removed.len() - REMOVED_BUFFER);
        }

        self.tracked
            .iter()
            .map(|&i| &self.tracks[i])
            .filter(|t| t.is_activated)
            .map(|t| TrackOut { xyxy: t.xyxy32(), track_id: t.track_id, score: t.score, cls: t.cls, idx: t.idx })
            .collect()
    }

    fn remove_duplicates(&self, a: Vec<usize>, b: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
        let ab: Vec<[f32; 4]> = a.iter().map(|&i| self.tracks[i].xyxy32()).collect();
        let bb: Vec<[f32; 4]> = b.iter().map(|&i| self.tracks[i].xyxy32()).collect();
        let d = iou_distance(&ab, &bb);
        let (mut dupa, mut dupb) = (vec![], vec![]);
        for (p, row) in d.iter().enumerate() {
            for (q, &v) in row.iter().enumerate() {
                if v < 0.15 {
                    let tp = self.tracks[a[p]].frame_id - self.tracks[a[p]].start_frame;
                    let tq = self.tracks[b[q]].frame_id - self.tracks[b[q]].start_frame;
                    if tp > tq {
                        dupb.push(q);
                    } else {
                        dupa.push(p);
                    }
                }
            }
        }
        let ra = a.iter().enumerate().filter(|(i, _)| !dupa.contains(i)).map(|(_, &t)| t).collect();
        let rb = b.iter().enumerate().filter(|(i, _)| !dupb.contains(i)).map(|(_, &t)| t).collect();
        (ra, rb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, conf: f32) -> Det {
        Det { xyxy: [x, y, x + 20.0, y + 50.0], conf, cls: 4 }
    }

    #[test]
    fn ids_follow_moving_boxes() {
        let mut t = ByteTracker::new();
        let out = t.update(&[det(10.0, 10.0, 0.9), det(200.0, 10.0, 0.8)]);
        assert_eq!(out.iter().map(|o| o.track_id).collect::<Vec<_>>(), vec![1, 2]);
        for k in 1..10 {
            let s = k as f32 * 2.0;
            // order swapped: ids must follow the boxes, not the order
            let out = t.update(&[det(200.0 + s, 10.0, 0.8), det(10.0 + s, 10.0, 0.9)]);
            let ids: Vec<(i64, usize)> = out.iter().map(|o| (o.track_id, o.idx)).collect();
            assert!(ids.contains(&(1, 1)) && ids.contains(&(2, 0)), "{ids:?}");
        }
    }

    #[test]
    fn hungarian_finds_optimum() {
        let a = vec![vec![4.0, 1.0, 3.0], vec![2.0, 0.0, 5.0], vec![3.0, 2.0, 2.0]];
        assert_eq!(hungarian(&a), vec![1, 0, 2]);
    }
}
