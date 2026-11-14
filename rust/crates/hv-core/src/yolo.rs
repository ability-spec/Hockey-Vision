//! Ultralytics' YOLO postprocessing for these (non end-to-end) models: confidence filter, best
//! class, class filter, class-offset NMS (torchvision's), then boxes scaled back to the image the
//! model was given. Same float32 arithmetic and order as Python, so boxes and scores match.

/// One detection: box (x1, y1, x2, y2), confidence, class index. float32 like the torch tensors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Det {
    pub xyxy: [f32; 4],
    pub conf: f32,
    pub cls: usize,
}

pub const IOU: f32 = 0.7; // default.yaml iou
pub const MAX_DET: usize = 300;
const MAX_WH: f32 = 7680.0;
const MAX_NMS: usize = 30000;

/// ultralytics.utils.nms.non_max_suppression for one image of a (4 + nc, anchors) output.
pub fn nms(pred: &[f32], nc: usize, anchors: usize, conf_thres: f32, classes: Option<&[usize]>) -> Vec<Det> {
    debug_assert_eq!(pred.len(), (4 + nc) * anchors);
    let at = |row: usize, a: usize| pred[row * anchors + a];
    let mut cand: Vec<Det> = Vec::new();
    for a in 0..anchors {
        // amax over classes > conf; then the first class with that max.
        let (mut best, mut j) = (at(4, a), 0);
        for c in 1..nc {
            let v = at(4 + c, a);
            if v > best {
                best = v;
                j = c;
            }
        }
        if !(best > conf_thres) {
            continue;
        }
        if classes.is_some_and(|cl| !cl.contains(&j)) {
            continue;
        }
        let (x, y, w, h) = (at(0, a), at(1, a), at(2, a), at(3, a));
        let (hw, hh) = (w / 2.0, h / 2.0);
        cand.push(Det { xyxy: [x - hw, y - hh, x + hw, y + hh], conf: best, cls: j });
    }
    if cand.len() > MAX_NMS {
        cand.sort_by(|a, b| b.conf.total_cmp(&a.conf));
        cand.truncate(MAX_NMS);
    }
    let keep = torchvision_nms(&cand, IOU);
    keep.into_iter().take(MAX_DET).map(|i| cand[i]).collect()
}

/// torchvision.ops.nms on class-offset boxes: indices kept, highest score first.
fn torchvision_nms(dets: &[Det], iou: f32) -> Vec<usize> {
    let boxes: Vec<[f32; 4]> = dets
        .iter()
        .map(|d| {
            let c = d.cls as f32 * MAX_WH;
            [d.xyxy[0] + c, d.xyxy[1] + c, d.xyxy[2] + c, d.xyxy[3] + c]
        })
        .collect();
    let areas: Vec<f32> = boxes.iter().map(|b| (b[2] - b[0]) * (b[3] - b[1])).collect();
    let mut order: Vec<usize> = (0..dets.len()).collect();
    order.sort_by(|&a, &b| dets[b].conf.total_cmp(&dets[a].conf)); // stable, descending
    let mut suppressed = vec![false; dets.len()];
    let mut keep = vec![];
    for (k, &i) in order.iter().enumerate() {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        let bi = boxes[i];
        for &j in &order[k + 1..] {
            if suppressed[j] {
                continue;
            }
            let bj = boxes[j];
            let w = (bi[2].min(bj[2]) - bi[0].max(bj[0])).max(0.0);
            let h = (bi[3].min(bj[3]) - bi[1].max(bj[1])).max(0.0);
            let inter = w * h;
            let ovr = inter / (areas[i] + areas[j] - inter);
            if ovr as f64 > iou as f64 {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// ops.scale_boxes from the model input (h, w) back to the image (h, w), then clip_boxes.
pub fn scale_boxes(dets: &mut [Det], input: (usize, usize), image: (usize, usize)) {
    let (ih, iw) = (input.0 as f64, input.1 as f64);
    let (oh, ow) = (image.0 as f64, image.1 as f64);
    let gain = (ih / oh).min(iw / ow);
    let pad_x = ((iw - (ow * gain).round_ties_even()) / 2.0 - 0.1).round_ties_even() as f32;
    let pad_y = ((ih - (oh * gain).round_ties_even()) / 2.0 - 0.1).round_ties_even() as f32;
    let g = gain as f32;
    for d in dets.iter_mut() {
        let b = &mut d.xyxy;
        b[0] -= pad_x;
        b[1] -= pad_y;
        b[2] -= pad_x;
        b[3] -= pad_y;
        for v in b.iter_mut() {
            *v /= g;
        }
    }
    clip_boxes(dets, image);
}

/// ops.clip_boxes to an image of (h, w).
pub fn clip_boxes(dets: &mut [Det], image: (usize, usize)) {
    let (h, w) = (image.0 as f32, image.1 as f32);
    for d in dets.iter_mut() {
        let b = &mut d.xyxy;
        b[0] = b[0].clamp(0.0, w);
        b[1] = b[1].clamp(0.0, h);
        b[2] = b[2].clamp(0.0, w);
        b[3] = b[3].clamp(0.0, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suppresses_overlaps_within_a_class_only() {
        // anchors: two overlapping boxes of class 0, one on top of them of class 1
        let anchors = 3;
        #[rustfmt::skip]
        let pred = vec![
            10.0, 11.0, 10.0,   // x
            10.0, 10.0, 10.0,   // y
            8.0, 8.0, 8.0,      // w
            8.0, 8.0, 8.0,      // h
            0.9, 0.8, 0.1,      // class 0
            0.1, 0.1, 0.7,      // class 1
        ];
        let d = nms(&pred, 2, anchors, 0.25, None);
        assert_eq!(d.iter().map(|d| (d.cls, d.conf)).collect::<Vec<_>>(), vec![(0, 0.9), (1, 0.7)]);
    }
}
