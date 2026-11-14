//! Jersey numbers from the digit model's boxes: `assemble_number` and `JerseyVotes` from
//! `run_models.py`, ported for the Rust frame loop (they run on every player, every frame).

use std::hash::Hash;

use std::collections::HashMap;

pub const DIGIT_NMS_IOU: f64 = 0.5;
pub const MIN_READING_CONF: f64 = 0.4;
pub const MIN_JERSEY_VOTES: f64 = 1.2;
pub const PARTIAL_READ_WEIGHT: f64 = 0.5;
pub const SWITCH_RATIO: f64 = 1.5;
pub const SETTLED_VOTES: f64 = 8.0;
pub const SETTLED_LEAD: f64 = 3.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Digit {
    /// The digit model's class name, "0" to "9".
    pub class: String,
    pub confidence: f64,
    pub box_xyxy: [f64; 4],
}

fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Join the digits found on one player into a number: (number, mean confidence), or None.
/// Starts from the most confident digit and adds at most one level neighbour of the same height.
pub fn assemble_number(digits: &[Digit]) -> Option<(String, f64)> {
    let mut sorted: Vec<&Digit> = digits.iter().collect();
    sorted.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<&Digit> = Vec::new();
    for d in sorted {
        if kept
            .iter()
            .all(|k| iou(&d.box_xyxy, &k.box_xyxy) < DIGIT_NMS_IOU)
        {
            kept.push(d);
        }
    }
    let best = *kept.first()?;
    let [x1, y1, x2, y2] = best.box_xyxy;
    let (h, cy) = (y2 - y1, (y1 + y2) / 2.0);
    let mut group = vec![best];
    for d in &kept[1..] {
        let [dx1, dy1, dx2, dy2] = d.box_xyxy;
        let gap = (dx1 - x2).max(x1 - dx2); // horizontal space between the boxes, negative if they overlap
        if ((dy2 - dy1) - h).abs() < 0.35 * h
            && ((dy1 + dy2) / 2.0 - cy).abs() < 0.5 * h
            && -0.3 * h < gap
            && gap < 0.6 * h
        {
            group.push(d);
            break;
        }
    }
    group.sort_by(|a, b| a.box_xyxy[0].total_cmp(&b.box_xyxy[0]));
    let number: String = group.iter().map(|d| d.class.as_str()).collect();
    if number.starts_with('0') {
        return None; // the NHL doesn't allow 0 or 00; a lone "0" is usually a logo
    }
    let conf = group.iter().map(|d| d.confidence).sum::<f64>() / group.len() as f64;
    Some((number, conf))
}

/// Per player track, the summed confidence of every number read on it (in the order first read,
/// like Python's Counter, so ties resolve the same way). Once shown, a number sticks to the track.
#[derive(Clone, Debug)]
pub struct JerseyVotes<K: Hash + Eq + Clone> {
    votes: HashMap<K, Vec<(String, f64)>>,
    shown: HashMap<K, String>,
}

impl<K: Hash + Eq + Clone> Default for JerseyVotes<K> {
    fn default() -> Self {
        JerseyVotes {
            votes: HashMap::new(),
            shown: HashMap::new(),
        }
    }
}

fn get(votes: &[(String, f64)], n: &str) -> f64 {
    votes.iter().find(|(k, _)| k == n).map_or(0.0, |(_, v)| *v)
}

/// A two-digit number is also backed by reads that only caught one of its digits.
fn score(votes: &[(String, f64)], n: &str) -> f64 {
    let partial = if n.chars().count() == 2 {
        let mut seen = Vec::new();
        n.chars()
            .filter(|c| {
                let new = !seen.contains(c);
                seen.push(*c);
                new
            })
            .map(|c| get(votes, &c.to_string()))
            .sum()
    } else {
        0.0
    };
    get(votes, n) + PARTIAL_READ_WEIGHT * partial
}

impl<K: Hash + Eq + Clone> JerseyVotes<K> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, track: Option<&K>, reading: Option<&(String, f64)>) {
        let (Some(track), Some((number, conf))) = (track, reading) else {
            return;
        };
        if *conf < MIN_READING_CONF {
            return;
        }
        let v = self.votes.entry(track.clone()).or_default();
        match v.iter_mut().find(|(k, _)| k == number) {
            Some((_, total)) => *total += conf,
            None => v.push((number.clone(), *conf)),
        }
    }

    /// True once the shown number is so far ahead that more reads are very unlikely to change it.
    pub fn settled(&self, track: &K) -> bool {
        let Some(shown) = self.shown.get(track) else {
            return false;
        };
        let votes = self.votes.get(track).map(Vec::as_slice).unwrap_or(&[]);
        let top = score(votes, shown);
        let two = shown.chars().count() == 2;
        let rival = votes
            .iter()
            .filter(|(n, _)| n != shown && !(two && shown.contains(n.as_str())))
            .map(|(n, _)| score(votes, n))
            .fold(0.0, f64::max);
        top >= SETTLED_VOTES && top >= SETTLED_LEAD * rival
    }

    /// The number to show for `track` now, updating what's shown.
    pub fn number(&mut self, track: &K) -> Option<String> {
        let votes = self.votes.get(track).filter(|v| !v.is_empty())?;
        let mut best = &votes[0].0;
        for (n, _) in votes {
            if score(votes, n) > score(votes, best) {
                best = n;
            }
        }
        if let Some(current) = self.shown.get(track) {
            if score(votes, best) < SWITCH_RATIO * score(votes, current) {
                return Some(current.clone());
            }
        }
        if score(votes, best) >= MIN_JERSEY_VOTES {
            let best = best.clone();
            self.shown.insert(track.clone(), best);
        }
        self.shown.get(track).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digit(class: &str, conf: f64, b: [f64; 4]) -> Digit {
        Digit {
            class: class.into(),
            confidence: conf,
            box_xyxy: b,
        }
    }

    #[test]
    fn joins_two_level_digits_left_to_right() {
        let ds = [
            digit("3", 0.9, [30.0, 10.0, 40.0, 30.0]),
            digit("6", 0.8, [18.0, 10.0, 28.0, 30.0]),
        ];
        let (n, c) = assemble_number(&ds).unwrap();
        assert_eq!(n, "63");
        assert!((c - 0.85).abs() < 1e-12);
    }

    #[test]
    fn ignores_a_digit_of_another_size_and_rejects_leading_zero() {
        let ds = [
            digit("7", 0.9, [30.0, 10.0, 40.0, 30.0]),
            digit("1", 0.8, [42.0, 10.0, 46.0, 18.0]),
        ];
        assert_eq!(assemble_number(&ds).unwrap().0, "7");
        assert!(assemble_number(&[digit("0", 0.9, [0.0, 0.0, 5.0, 10.0])]).is_none());
        assert!(assemble_number(&[]).is_none());
    }

    #[test]
    fn votes_stick_and_settle() {
        let mut v = JerseyVotes::new();
        let t = 7;
        v.add(Some(&t), Some(&("12".to_string(), 0.9)));
        assert_eq!(v.number(&t), None); // one read is below MIN_JERSEY_VOTES
                                        // A single-digit read of "12" backs it rather than competing with it.
        for _ in 0..8 {
            v.add(Some(&t), Some(&("12".to_string(), 0.9)));
            v.add(Some(&t), Some(&("2".to_string(), 0.6)));
        }
        assert_eq!(v.number(&t), Some("12".into()));
        assert!(v.settled(&t));
        // Weak reads don't vote.
        v.add(Some(&t), Some(&("99".to_string(), 0.3)));
        assert_eq!(v.number(&t), Some("12".into()));
        assert_eq!(v.number(&8), None);
    }
}
