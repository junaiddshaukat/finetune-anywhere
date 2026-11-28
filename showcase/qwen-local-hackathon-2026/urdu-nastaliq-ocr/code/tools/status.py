"""One-screen progress report: `make status`. Reads logs only, touches nothing."""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

LOGS = Path(__file__).resolve().parents[1] / "outputs" / "logs"
ROOT = LOGS.parents[1]


def tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return ["(no log yet)"]
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


def progress(path: Path) -> str:
    if not path.exists():
        return "not started"
    hits = re.findall(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+)", path.read_text(encoding="utf-8", errors="replace")[-20000:])
    losses = re.findall(r"'loss': '?([\d.]+)", path.read_text(encoding="utf-8", errors="replace")[-20000:])
    if not hits:
        return "starting"
    done, total, spent, eta = hits[-1]
    return f"step {done}/{total}, elapsed {spent}, ETA {eta}" + (f", last loss {losses[-1]}" if losses else "")


def main() -> None:
    print(time.strftime("== %a %d %b %H:%M =="))
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        print("GPU (temp, util, VRAM):", g)
    except OSError:
        pass
    for chain in ("v2_chain.log", "v3_chain.log"):
        print(f"\n-- {chain} (last 6 lines)")
        print("\n".join("  " + l for l in tail(LOGS / chain, 6)))
    for name in ("train_v2.log", "train_v3.log"):
        print(f"\n{name}: {progress(LOGS / name)}")
    print(f"\ndata render: {tail(LOGS / 'render_noto.log', 1)[0]}")
    for f in ("compare_v2.md", "compare_v3.md", "real_bench.md", "pick_best.md"):
        p = LOGS / f
        if p.exists():
            print(f"\n== {f}\n" + p.read_text(encoding="utf-8", errors="replace")[:2500])


if __name__ == "__main__":
    main()
