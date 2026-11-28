"""Regenerate showcase tables. Newest means most recent entry commit, then folder name."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

import jsonschema
import yaml
from common import ROOT, cell

START, END = "<!-- SHOWCASE:START -->", "<!-- SHOWCASE:END -->"


def replace(path, content):
    original = path.read_text(encoding="utf-8")
    if original.count(START) != 1 or original.count(END) != 1:
        raise ValueError(f"{path}: expected one pair of SHOWCASE markers")
    before, tail = original.split(START)
    _, after = tail.split(END)
    result = before + START + "\n\n" + content + "\n\n" + END + after
    if result != original:
        path.write_text(result, encoding="utf-8")


def table(entries, relative_to):
    if not entries:
        return "No submissions yet. Be the first to share a model you trained and ran locally."
    lines = [
        "| Project | What it does | Base model | Tool | Quant | Runtime | Team | Demo |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, path, d in entries:
        link = quote(os.path.relpath(path.parent, relative_to), safe="/.-")
        team = ", ".join(f"[@{m['github']}](https://github.com/{m['github']})" for m in d["members"])
        demo = (
            "[Watch](" + quote(d["demo_video"], safe=":/?=&%#") + ")"
            if d.get("demo_video")
            else "Not provided"
        )
        values = [
            f"[{cell(d['project_name']).replace('[', '&#91;').replace(']', '&#93;')}]({link}/)",
            cell(d["one_liner"]),
            cell(d["base_model"]),
            cell(d.get("training_tool", "Not stated")),
            cell(d.get("quantization") or "None"),
            cell(d["runtime"]),
            team,
            demo,
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build(root):
    schema = json.loads((root / "schema/model.schema.json").read_text())
    entries = []
    for path in sorted((root / "showcase").glob("*/*/model.yaml")):
        if "_template" in path.parts:
            continue
        data = yaml.safe_load(path.read_text())
        jsonschema.validate(data, schema)
        if data["slug"] != path.parent.name or data["event"] != path.parent.parent.name:
            raise ValueError(f"{path}: slug/event does not match folder")
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", str(path.relative_to(root))],
            cwd=root,
            capture_output=True,
            text=True,
        )
        timestamp = int(result.stdout.strip() or 0)
        entries.append((timestamp, path, data))
    entries.sort(key=lambda row: (row[0], str(row[1])), reverse=True)
    events = sorted(
        p.parent.name for p in (root / "showcase").glob("*/README.md") if p.parent.name != "_template"
    )
    sections = []
    for event in events:
        subset = [e for e in entries if e[2]["event"] == event]
        sections.append(f"## {event}\n\n" + table(subset, root / "showcase"))
        replace(root / "showcase" / event / "README.md", table(subset, root / "showcase" / event))
    replace(root / "showcase/README.md", "\n\n".join(sections))
    replace(root / "README.md", table(entries[:10], root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    build(args.root.resolve())
