//! NHL rink geometry: named keypoints in real-world feet. Port of `rink.py` (the keypoint table;
//! the template drawing stays in Python).
//!
//! Rink coordinates are feet from the centre dot: x along the rink (-100 left end boards to +100
//! right), y across it (+42.5 far boards to -42.5 near boards).

use std::collections::HashMap;
use std::sync::OnceLock;

pub const RINK_LENGTH: f64 = 200.0;
pub const RINK_WIDTH: f64 = 85.0;
pub const HALF_LENGTH: f64 = RINK_LENGTH / 2.0;
pub const HALF_WIDTH: f64 = RINK_WIDTH / 2.0;

pub const GOAL_LINE_X: f64 = 89.0;
pub const BLUE_LINE_X: f64 = 25.0;
pub const NEUTRAL_DOT_X: f64 = 20.0;
pub const END_DOT_X: f64 = 69.0;
pub const FACEOFF_DOT_Y: f64 = 22.0;
pub const CIRCLE_RADIUS: f64 = 15.0;
pub const CORNER_RADIUS: f64 = 28.0;
pub const GOAL_POST_Y: f64 = 3.0;
pub const CREASE_RADIUS: f64 = 6.0;
pub const REF_CREASE_RADIUS: f64 = 10.0;
pub const HASH_OFFSET_X: f64 = 2.79;
pub const HASH_LENGTH: f64 = 2.0;

/// Index of a keypoint in [`keypoints`].
pub type KeypointId = usize;

#[derive(Debug, Clone)]
pub struct Keypoint {
    pub name: String,
    pub xy: [f64; 2],
    pub group: &'static str,
}

pub struct Keypoints {
    pub list: Vec<Keypoint>,
    by_name: HashMap<String, KeypointId>,
}

impl Keypoints {
    /// The id of a keypoint by name. Panics on an unknown name: every name the matcher builds is
    /// one of the rink's, so a miss is a bug.
    pub fn id(&self, name: &str) -> KeypointId {
        *self
            .by_name
            .get(name)
            .unwrap_or_else(|| panic!("unknown rink keypoint {name}"))
    }

    pub fn xy(&self, id: KeypointId) -> [f64; 2] {
        self.list[id].xy
    }

    pub fn name(&self, id: KeypointId) -> &str {
        &self.list[id].name
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
}

/// Where the goal line meets the curved corner boards.
pub fn goal_line_boards_y() -> f64 {
    (HALF_WIDTH - CORNER_RADIUS)
        + (CORNER_RADIUS.powi(2) - (GOAL_LINE_X - (HALF_LENGTH - CORNER_RADIUS)).powi(2)).sqrt()
}

fn build() -> Keypoints {
    let mut list = Vec::new();
    let mut add = |name: String, x: f64, y: f64, group: &'static str| {
        list.push(Keypoint {
            name,
            xy: [x, y],
            group,
        });
    };

    add("center_dot".into(), 0.0, 0.0, "center");
    add("center_circle_far".into(), 0.0, CIRCLE_RADIUS, "center");
    add("center_circle_near".into(), 0.0, -CIRCLE_RADIUS, "center");
    add("center_circle_left".into(), -CIRCLE_RADIUS, 0.0, "center");
    add("center_circle_right".into(), CIRCLE_RADIUS, 0.0, "center");
    add("center_line_far_boards".into(), 0.0, HALF_WIDTH, "lines");
    add("center_line_near_boards".into(), 0.0, -HALF_WIDTH, "lines");

    for (side, sy) in [("far", 1.0), ("near", -1.0)] {
        add(
            format!("ref_crease_{side}_apex"),
            0.0,
            sy * (HALF_WIDTH - REF_CREASE_RADIUS),
            "center",
        );
        add(
            format!("ref_crease_{side}_left"),
            -REF_CREASE_RADIUS,
            sy * HALF_WIDTH,
            "center",
        );
        add(
            format!("ref_crease_{side}_right"),
            REF_CREASE_RADIUS,
            sy * HALF_WIDTH,
            "center",
        );
    }

    let gl_boards_y = goal_line_boards_y();
    let hash_base_y = (CIRCLE_RADIUS.powi(2) - HASH_OFFSET_X.powi(2)).sqrt();
    for (e, sx) in [("L", -1.0), ("R", 1.0)] {
        add(
            format!("{e}_blue_line_far_boards"),
            sx * BLUE_LINE_X,
            HALF_WIDTH,
            "lines",
        );
        add(
            format!("{e}_blue_line_near_boards"),
            sx * BLUE_LINE_X,
            -HALF_WIDTH,
            "lines",
        );
        add(
            format!("{e}_neutral_dot_far"),
            sx * NEUTRAL_DOT_X,
            FACEOFF_DOT_Y,
            "neutral_dots",
        );
        add(
            format!("{e}_neutral_dot_near"),
            sx * NEUTRAL_DOT_X,
            -FACEOFF_DOT_Y,
            "neutral_dots",
        );

        add(
            format!("{e}_goal_line_far_boards"),
            sx * GOAL_LINE_X,
            gl_boards_y,
            "lines",
        );
        add(
            format!("{e}_goal_line_near_boards"),
            sx * GOAL_LINE_X,
            -gl_boards_y,
            "lines",
        );
        add(format!("{e}_goal_center"), sx * GOAL_LINE_X, 0.0, "goal");
        add(
            format!("{e}_goal_post_far"),
            sx * GOAL_LINE_X,
            GOAL_POST_Y,
            "goal",
        );
        add(
            format!("{e}_goal_post_near"),
            sx * GOAL_LINE_X,
            -GOAL_POST_Y,
            "goal",
        );
        add(
            format!("{e}_crease_apex"),
            sx * (GOAL_LINE_X - CREASE_RADIUS),
            0.0,
            "goal",
        );

        add(
            format!("{e}_end_boards_center"),
            sx * HALF_LENGTH,
            0.0,
            "boards",
        );
        for (side, sy) in [("far", 1.0), ("near", -1.0)] {
            add(
                format!("{e}_corner_{side}_side_boards"),
                sx * (HALF_LENGTH - CORNER_RADIUS),
                sy * HALF_WIDTH,
                "boards",
            );
            add(
                format!("{e}_corner_{side}_end_boards"),
                sx * HALF_LENGTH,
                sy * (HALF_WIDTH - CORNER_RADIUS),
                "boards",
            );
        }

        for (side, sy) in [("far", 1.0), ("near", -1.0)] {
            let (cx, cy) = (sx * END_DOT_X, sy * FACEOFF_DOT_Y);
            let p = format!("{e}_{side}");
            add(format!("{p}_endzone_dot"), cx, cy, "endzone_dots");
            add(format!("{p}_circle_top"), cx, cy + CIRCLE_RADIUS, "circles");
            add(
                format!("{p}_circle_bottom"),
                cx,
                cy - CIRCLE_RADIUS,
                "circles",
            );
            add(
                format!("{p}_circle_left"),
                cx - CIRCLE_RADIUS,
                cy,
                "circles",
            );
            add(
                format!("{p}_circle_right"),
                cx + CIRCLE_RADIUS,
                cy,
                "circles",
            );
            for (vert, sv) in [("top", 1.0), ("bottom", -1.0)] {
                let tip_y = cy + sv * (hash_base_y + HASH_LENGTH);
                for (horiz, sh) in [("left", -1.0), ("right", 1.0)] {
                    add(
                        format!("{p}_hash_{vert}_{horiz}"),
                        cx + sh * HASH_OFFSET_X,
                        tip_y,
                        "hash_marks",
                    );
                }
            }
        }
    }
    let by_name = list
        .iter()
        .enumerate()
        .map(|(i, k)| (k.name.clone(), i))
        .collect();
    Keypoints { list, by_name }
}

/// Every rink keypoint, built once.
pub fn keypoints() -> &'static Keypoints {
    static KEYPOINTS: OnceLock<Keypoints> = OnceLock::new();
    KEYPOINTS.get_or_init(build)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keypoint_table_matches_rink_py() {
        let k = keypoints();
        // rink.py builds 7 centre + 6 referee crease + 2 x (4 + 6 + 5 + 2 x (5 + 4)) = 79.
        assert_eq!(k.len(), 79);
        assert_eq!(k.xy(k.id("R_far_endzone_dot")), [69.0, 22.0]);
        assert_eq!(k.xy(k.id("L_blue_line_near_boards")), [-25.0, -42.5]);
        // rink.KEYPOINTS["R_goal_line_far_boards"].xy in Python.
        let y = k.xy(k.id("R_goal_line_far_boards"))[1];
        assert!((y - 36.748_595_461_286_99).abs() < 1e-9, "{y}");
    }
}
