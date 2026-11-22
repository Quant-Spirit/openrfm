"""Generate fact-row, same-time slate and event-window prediction tasks."""

from __future__ import annotations

from pathlib import Path

from openrfm.synth.worlds.fact.base import FamilyConfig
from openrfm.synth.worlds.fact.oracle import admitted
from openrfm.synth.worlds.fact.rows import f1_world, f2_world, f3_world
from openrfm.synth.worlds.fact.windows import f4_world
from openrfm.synth.worlds.fact.write import write_autocomplete, write_slate, write_window_pair


def generate(family, slot, seed, raw_root, private_dir, cfg=FamilyConfig(), attempts=6):
    """Write one admitted world (F4: a raw/summary pair); candidates differ only by the draw index."""
    import shutil

    rejected = []
    for a in range(attempts):
        cand = slot + a * 100003
        if family == "F4":
            d = f4_world(seed, cand, cfg)
            if (
                d["kind"] == "binary"
                and min(d["y"].sum(), len(d["y"]) - d["y"].sum()) < cfg.min_class_count
            ):
                rejected.append("rare_class_in_cases")
                continue
            recs = write_window_pair(raw_root, d, cand, seed, private_dir)
            reason = admitted(recs[0], cfg)
        else:
            W = {"F1": f1_world, "F2": f2_world, "F3": f3_world}[family](seed, cand, cfg)
            yc = W["y"][W["cases"]]
            if len(yc) < cfg.min_cases:
                rejected.append("too_few_cases")
                continue
            if (
                W["kind"] in ("binary", "choice")
                and min(yc.sum(), len(yc) - yc.sum()) < cfg.min_class_count
            ):
                rejected.append("rare_class_in_cases")
                continue
            recs = [
                (write_slate if family == "F3" else write_autocomplete)(
                    raw_root, W, cand, seed, private_dir, cfg
                )
            ]
            reason = admitted(recs[0], cfg)
        if reason is None:
            for r in recs:
                r["attempts"], r["rejected"] = a + 1, rejected
            return recs
        rejected.append(reason)
        for r in recs:
            shutil.rmtree(Path(raw_root) / r["db_name"])
            (Path(private_dir) / r["private_file"]).unlink(missing_ok=True)
    raise RuntimeError(f"{family} slot {slot}: exhausted {attempts} candidates {rejected}")


