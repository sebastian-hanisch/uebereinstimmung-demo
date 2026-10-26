"""Rechnet das Raster der Messreihen vor (E1 bis E5) und schreibt data/agreement_grid.json.

    python tools/make_grid.py            # volles Raster, mehrere Prozesse
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht in data/agreement_grid.json)

Das Raster ist deterministisch (SplitMix64, feste Seeds); tests/test_grid.py rechnet eine feste Probe nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ueb_constants as C  # noqa: E402
import ueb_evaluation as ev  # noqa: E402

JOBS = {"error": ev.error_cell, "spread": ev.spread_cell, "recal": ev.recal_cell, "rank": ev.rank_cell, "ba": ev.ba_cell}


def _run(task):
    kind, args, kwargs = task
    return kind, JOBS[kind](*args, **kwargs)


def main() -> None:
    quick = "--quick" in sys.argv
    tasks = [("error", (k, C.E1_N, C.E1_RUNS), {}) for k in C.CASES]
    tasks += [("error", (k, n, C.E1_SPREAD_RUNS), {}) for k in ("perfekt", "skala", "bias") for n in C.E1_SPREAD_NS]
    tasks += [("spread", (s, b, C.E2_N, C.E2_RUNS), {}) for b in C.E2_BIASES for s in C.E2_SIGMAS]
    tasks += [("recal", (k, C.E3_N, C.E3_RUNS), {}) for k in ("bias", "flach", "skala", "bias_flach", "rauschen")]
    tasks += [("recal", ("skala", n, C.E3_RUNS), {}) for n in C.E3_SMALL_NS if n != C.E3_N]
    tasks += [("rank", (C.E4_N, C.E4_RUNS), {})]
    tasks += [("ba", (k, C.E5_N, C.E5_RUNS), {}) for k in C.E5_CASES]
    # E1 (n = 2 000) und die Streuungszellen (kleine n) tragen dasselbe Format, die Gruppe steht im Namen
    groups = ["e1"] * len(C.CASES) + ["e1_spread"] * (3 * len(C.E1_SPREAD_NS)) + ["e2"] * (len(C.E2_BIASES) * len(C.E2_SIGMAS)) + ["e3"] * 5 + ["e3_small"] * (len(C.E3_SMALL_NS) - 1) \
        + ["e4"] + ["e5"] * len(C.E5_CASES)
    if quick:
        tasks = [("error", ("skala", 100, 1), {"seeds": (7,)}), ("spread", (10, 4, 100, 1), {"seeds": (7,)}), ("recal", ("flach", 100, 1), {"seeds": (7,)}), ("rank", (50, 1), {"seeds": (7,)}),
                 ("ba", ("proportional", 90, 1), {"seeds": (7,)})]
        groups = ["e1", "e2", "e3", "e4", "e5"]
    t0 = time.time()
    out: dict = {g: [] for g in dict.fromkeys(groups)}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for group, (kind, cell) in zip(groups, pool.map(_run, tasks, chunksize=1)):
            out[group].append(cell)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("agreement_grid_quick.json" if quick else "agreement_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f", {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
