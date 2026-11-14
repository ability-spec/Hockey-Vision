//! `roster.py`: the two teams' rosters for a video, from a fetch_roster.py CSV. Which numbers a
//! player of each model class can wear, and who wears them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Parse CSV text (the csv module's default dialect) into rows of fields.
pub fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = vec![];
    let mut row = vec![];
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    let mut any = false;
    while let Some(c) = chars.next() {
        any = true;
        if quoted {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => quoted = false,
                c => field.push(c),
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            ',' => row.push(std::mem::take(&mut field)),
            '\r' => {}
            '\n' => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
                any = false;
            }
            c => field.push(c),
        }
    }
    if any {
        row.push(field);
        rows.push(row);
    }
    rows
}

pub struct Roster {
    pub path: PathBuf,
    /// (team_a, team_b) abbreviations.
    pub teams: [String; 2],
    /// (team index, is goalie) -> {number: player name}; numbers two players share list both.
    names: HashMap<(usize, bool), Vec<(String, String)>>,
    team_names: HashMap<String, String>,
}

/// Player model class -> (team index, is goalie).
fn class_team(cls: &str) -> Option<(usize, bool)> {
    Some(match cls {
        "team_a_player" => (0, false),
        "goalie_a" => (0, true),
        "team_b_player" => (1, false),
        "goalie_b" => (1, true),
        _ => return None,
    })
}

impl Roster {
    pub fn load(path: &Path, teams: [&str; 2]) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("Roster not found: {} (fetch it with: python fetch_roster.py)", path.display()));
        }
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut rows = parse_csv(text.trim_start_matches('\u{feff}')).into_iter();
        let header = rows.next().unwrap_or_default();
        let col = |name: &str| header.iter().position(|h| h == name).ok_or(format!("{}: no {name} column", path.display()));
        let (c_team, c_pos, c_num, c_player, c_tname) = (col("team")?, col("position")?, col("number")?, col("player")?, col("team_name")?);
        let rows: Vec<Vec<String>> = rows.filter(|r| r.len() == header.len()).collect();
        let teams = teams.map(|t| t.to_uppercase());
        let mut available: Vec<&str> = rows.iter().map(|r| r[c_team].as_str()).collect();
        available.sort();
        available.dedup();
        for t in &teams {
            if !available.contains(&t.as_str()) {
                return Err(format!("No team {t} in {}; it has: {}", path.display(), available.join(", ")));
            }
        }
        let mut names: HashMap<(usize, bool), Vec<(String, String)>> = HashMap::new();
        let mut team_names = HashMap::new();
        for r in &rows {
            let Some(i) = teams.iter().position(|t| *t == r[c_team]) else { continue };
            let goalie = r[c_pos] == "Goalie";
            let num = r[c_num].trim().parse::<i64>().map_err(|_| format!("bad number {:?} in {}", r[c_num], path.display()))?.to_string();
            let list = names.entry((i, goalie)).or_default();
            match list.iter_mut().find(|(n, _)| *n == num) {
                Some((_, existing)) => *existing = format!("{existing} / {}", r[c_player]),
                None => list.push((num, r[c_player].clone())),
            }
            team_names.insert(r[c_team].clone(), r[c_tname].clone());
        }
        Ok(Roster { path: path.to_path_buf(), teams, names, team_names })
    }

    /// The numbers a player of model class `cls` can wear: Some(list) (empty for referees), or
    /// None for a class this doesn't know (no limit).
    pub fn allowed(&self, cls: &str) -> Option<Vec<String>> {
        if cls == "referee" {
            return Some(vec![]);
        }
        let key = class_team(cls)?;
        let mut nums: Vec<String> = self.names.get(&key).map(|l| l.iter().map(|(n, _)| n.clone()).collect()).unwrap_or_default();
        nums.sort_by_key(|n| n.parse::<i64>().unwrap_or(0));
        Some(nums)
    }

    /// (team abbreviation, player name) for a player of class `cls` wearing `number`.
    pub fn player(&self, cls: &str, number: Option<&str>) -> (Option<String>, Option<String>) {
        let Some(key) = class_team(cls) else { return (None, None) };
        let name = number.and_then(|n| {
            self.names.get(&key)?.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone())
        });
        (Some(self.teams[key.0].clone()), name)
    }

    pub fn describe(&self) -> String {
        self.teams
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let n = |g| self.names.get(&(i, g)).map_or(0, Vec::len);
                format!(
                    "team_{} = {t} ({}, {} skaters, {} goalies)",
                    ["a", "b"][i],
                    self.team_names.get(t).map(String::as_str).unwrap_or(""),
                    n(false),
                    n(true)
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_quotes() {
        let rows = parse_csv("a,b\r\n\"x, y\",\"he said \"\"hi\"\"\"\r\n");
        assert_eq!(rows, vec![vec!["a", "b"], vec!["x, y", "he said \"hi\""]]);
    }
}
