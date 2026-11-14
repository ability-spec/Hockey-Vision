"""Compare two export folders (export_data.py vs hockey-vision export) file by file.

    python rust/bench/compare_exports.py <python export dir> <rust export dir>

Byte-for-byte, except the generated_at timestamps in game_report.json and metadata.json. Prints the
first differing lines of any file that differs; exits 1 if any does.
"""

import re
import sys
from pathlib import Path

FILES = ["players.csv", "puck.csv", "frames.csv", "tracks.csv", "possessions.csv", "events.csv",
         "player_stats.csv", "game_report.json", "metadata.json"]
STAMP = re.compile(r'"generated_at": "[^"]*"')


def read(path):
    return STAMP.sub('"generated_at": "-"', path.read_text())


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    bad = 0
    for name in FILES:
        ta, tb = read(a / name), read(b / name)
        la, lb = ta.splitlines(), tb.splitlines()
        if ta == tb:
            print(f"  {name:18} identical ({len(la)} lines)")
            continue
        bad += 1
        diff = [i for i in range(max(len(la), len(lb))) if i >= len(la) or i >= len(lb) or la[i] != lb[i]]
        print(f"  {name:18} DIFFERS: {len(diff)} of {len(la)} lines")
        for i in diff[:3]:
            print(f"    line {i + 1}\n      py: {la[i] if i < len(la) else '<missing>'}\n      rs: {lb[i] if i < len(lb) else '<missing>'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
