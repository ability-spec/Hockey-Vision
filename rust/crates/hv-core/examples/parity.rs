//! Test helper for tests/test_rust_port.py: runs one hv_core component on a JSON request from
//! stdin and prints the result as JSON, so the tests can compare it with the Python original.
//!
//!     echo '{"op": "nms", ...}' | parity
//!
//! ops: nms (yolo::nms + scale_boxes), track (bytetrack::ByteTracker), assemble
//! (jersey::assemble_number), votes (jersey::JerseyVotes), roster (roster::Roster).
use std::io::Read;

use serde_json::{json, Value};

use hv_core::bytetrack::ByteTracker;
use hv_core::jersey::{assemble_number, Digit, JerseyVotes};
use hv_core::roster::Roster;
use hv_core::yolo::{nms, scale_boxes, Det};

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn det_json(d: &Det) -> Value {
    json!([d.xyxy[0] as f64, d.xyxy[1] as f64, d.xyxy[2] as f64, d.xyxy[3] as f64, d.conf as f64, d.cls])
}

fn digits(v: &Value) -> Vec<Digit> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|d| Digit {
            class: d["class"].as_str().unwrap().to_string(),
            confidence: d["confidence"].as_f64().unwrap(),
            box_xyxy: std::array::from_fn(|i| d["box_xyxy"][i].as_f64().unwrap()),
        })
        .collect()
}

fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let req: Value = serde_json::from_str(&s).unwrap();
    let out = match req["op"].as_str().unwrap() {
        "nms" => {
            let pred = f32s(&req["pred"]);
            let (nc, anchors) = (req["nc"].as_u64().unwrap() as usize, req["anchors"].as_u64().unwrap() as usize);
            let classes: Option<Vec<usize>> =
                req["classes"].as_array().map(|a| a.iter().map(|c| c.as_u64().unwrap() as usize).collect());
            let mut d = nms(&pred, nc, anchors, req["conf"].as_f64().unwrap() as f32, classes.as_deref());
            let shape = |k: &str| (req[k][0].as_u64().unwrap() as usize, req[k][1].as_u64().unwrap() as usize);
            scale_boxes(&mut d, shape("input"), shape("image"));
            Value::Array(d.iter().map(det_json).collect())
        }
        "track" => {
            let mut t = ByteTracker::new();
            let frames = req["frames"].as_array().unwrap();
            Value::Array(
                frames
                    .iter()
                    .map(|f| {
                        let dets: Vec<Det> = f
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|d| {
                                let v = f32s(d);
                                Det { xyxy: [v[0], v[1], v[2], v[3]], conf: v[4], cls: v[5] as usize }
                            })
                            .collect();
                        Value::Array(
                            t.update(&dets)
                                .iter()
                                .map(|o| {
                                    json!([o.xyxy[0] as f64, o.xyxy[1] as f64, o.xyxy[2] as f64, o.xyxy[3] as f64,
                                           o.track_id, o.score as f64, o.cls as f64, o.idx])
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            )
        }
        "assemble" => Value::Array(
            req["cases"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| match assemble_number(&digits(c)) {
                    Some((n, conf)) => json!([n, conf]),
                    None => Value::Null,
                })
                .collect(),
        ),
        "votes" => {
            // steps: ["add", track, number, conf] | ["number", track] | ["settled", track]
            let mut v: JerseyVotes<i64> = JerseyVotes::new();
            let mut out = vec![];
            for step in req["steps"].as_array().unwrap() {
                let track = step[1].as_i64().unwrap();
                match step[0].as_str().unwrap() {
                    "add" => {
                        let r = (step[2].as_str().unwrap().to_string(), step[3].as_f64().unwrap());
                        v.add(Some(&track), Some(&r));
                    }
                    "number" => out.push(json!(v.number(&track))),
                    _ => out.push(json!(v.settled(&track))),
                }
            }
            Value::Array(out)
        }
        "roster" => {
            let teams = [req["teams"][0].as_str().unwrap(), req["teams"][1].as_str().unwrap()];
            let r = Roster::load(req["path"].as_str().unwrap().as_ref(), teams).unwrap();
            let classes = ["team_a_player", "goalie_a", "team_b_player", "goalie_b", "referee", "other"];
            let allowed: Vec<Value> = classes.iter().map(|c| json!(r.allowed(c))).collect();
            let players: Vec<Value> = classes
                .iter()
                .flat_map(|c| (0..100).map(move |n| (c, n)))
                .map(|(c, n)| {
                    let n = n.to_string();
                    let (team, name) = r.player(c, Some(&n));
                    json!([team, name])
                })
                .collect();
            json!({"allowed": allowed, "players": players, "describe": r.describe()})
        }
        op => panic!("unknown op {op}"),
    };
    println!("{out}");
}
