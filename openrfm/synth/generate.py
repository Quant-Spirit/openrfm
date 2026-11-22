"""Generate a synthetic pretraining corpus; per-world receipts let an interrupted run resume (CPU only).

    PYTHONHASHSEED=0 python -m openrfm.synth.generate --config openrfm/configs/data/backbone_corpus.yaml --out OUT \
        --worldgen BIN --worldgen-materialize BIN [--families NEW.future.count F1 ...] [--limit N]
"""

import argparse
import json
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from openrfm.synth.candidates import selected_targets, world_candidate

FACT_FAMILIES = ("F1", "F2", "F3", "F4")
TABULAR_FAMILIES = {
    "OLD.tabular.tabicl": "tabicl_v2_like_discrete",
    "OLD.tabular.support": "support_required_multiclass",
    "OLD.tabular.operator": "relational_operator_discrete",
}


def world_job(job):
    """Generate world ``index`` of ``family``, trying successive slots until one is admitted."""
    cfg, root, family, index, binary = job
    receipt = root / "receipts" / f"{index:06d}.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    raw = root / "raw"
    dest = raw / f"rel-synthetic-{40000000 + index}"
    shutil.rmtree(dest, ignore_errors=True)
    rejected = []
    for attempt in range(cfg["attempts"]):
        slot = cfg["first_slot"] + 1000 * index + attempt
        reason = world_candidate(cfg, family, index, slot, raw, dest, binary)
        if reason is None and not selected_targets(dest, family):
            reason = "no_target_of_family"
        if reason is not None:
            rejected.append(reason)
            shutil.rmtree(dest, ignore_errors=True)
            continue
        for path in dest.glob("*.json"):  # bind sidecars to the final world name
            path.write_text(
                re.sub(
                    r"rel-synthetic-(?:v2-[0-9a-f]+|[0-9]+)",
                    dest.name,
                    path.read_text(),
                )
            )
        (dest / "unified_private.npz").unlink(missing_ok=True)
        result = dict(
            db=dest.name,
            family=family,
            slot=slot,
            attempts=attempt + 1,
            rejected=rejected,
            targets=selected_targets(dest, family),
        )
        receipt.write_text(json.dumps(result) + "\n")
        return result
    raise RuntimeError(f"{family}: no admitted world in {cfg['attempts']} attempts")


def fact_job(job):
    """One fact-table (F1/F2), slate (F3) or dated-window (F4) world."""
    from openrfm.synth.worlds.fact import FamilyConfig, generate
    from openrfm.synth.worlds.fact.audit import audit_pair, audit_world

    cfg, root, family, index = job
    receipt = root / "receipts" / f"{family}-{index:06d}.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    facts = cfg["fact_rows"]
    params = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in facts["family_config"].items()
    }
    pending = root / "pending" / f"{family}-{index}"
    shutil.rmtree(pending, ignore_errors=True)
    records = generate(
        family,
        facts["first_slot"] + index,
        facts["seed"],
        pending / "raw",
        root / "private",
        FamilyConfig(**params),
        attempts=facts["attempts"],
    )
    if not all(
        audit_world(pending / "raw", r["db_name"], root / "private")["ok"]
        for r in records
    ):
        raise ValueError(f"{family}-{index}: written-world cutoff/oracle audit failed")
    if family == "F4" and not all(
        audit_pair(pending / "raw", records[0], records[1]).values()
    ):
        raise ValueError(f"{family}-{index}: paired renderings differ")
    shipped = [r["db_name"] for r in records if r.get("pair_role") != "raw"]
    for name in shipped:
        (pending / "raw" / name).rename(root / "raw" / name)
    shutil.rmtree(pending)
    result = dict(family=family, index=index, worlds=shipped)
    receipt.write_text(json.dumps(result) + "\n")
    return result


def tabular_worlds(cfg, root, materializer, workers, families, limit=None):
    """Static tabular worlds from the Rust materializer, one manifest per replica."""
    manifest_dir = Path(__file__).resolve().parents[1] / "configs/data/tabular"
    out = root / "tabular" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for replica in range(cfg["tabular"]["replicas"]):
        path = manifest_dir / f"replica{replica}.json"
        manifest_hash = json.loads(path.read_text())["manifest_sha256"]
        for family in families:
            done = root / "receipts" / f"tabular-{replica}-{family}.json"
            if done.exists():
                continue
            subprocess.run(
                [
                    materializer,
                    "--manifest",
                    str(path),
                    "--expected-manifest-sha256",
                    manifest_hash,
                    "--output-root",
                    str(out),
                    "--split",
                    "train",
                    "--family",
                    family,
                    "--workers",
                    str(workers),
                    "--publish-gated-subset",
                ]
                + (["--limit", str(limit)] if limit is not None else []),
                check=True,
            )
            shutil.copy(out / "materialization_audit.json", done)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, default=Path("openrfm/configs/data/backbone_corpus.yaml"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worldgen", default="worldgen")
    p.add_argument("--worldgen-materialize", default="worldgen-materialize")
    p.add_argument("--families", nargs="+", help="generate only these families")
    p.add_argument(
        "--limit", type=int, help="at most this many worlds per family (smoke runs)"
    )
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    wanted = lambda f: not a.families or f in a.families
    for d in ("raw", "receipts", "pending", "private"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    jobs, index = [], 0
    for family, count in cfg["worlds"].items():
        if family in TABULAR_FAMILIES:
            continue
        for i in range(count):
            if wanted(family) and (a.limit is None or i < a.limit):
                jobs.append((cfg, a.out, family, index, a.worldgen))
            index += 1
    facts = [
        (cfg, a.out, f, i)
        for f in FACT_FAMILIES
        if wanted(f)
        for i in range(cfg["fact_rows"]["worlds"][f])
        if a.limit is None or i < a.limit
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        for n, r in enumerate(pool.map(world_job, jobs), 1):
            print("WORLD", n, len(jobs), r["family"], r["db"], flush=True)
        for n, r in enumerate(pool.map(fact_job, facts), 1):
            print("FACT", n, len(facts), r["family"], " ".join(r["worlds"]), flush=True)
    tabular = [name for group, name in TABULAR_FAMILIES.items() if wanted(group)]
    if tabular:
        tabular_worlds(cfg, a.out, a.worldgen_materialize, a.workers, tabular, a.limit)


if __name__ == "__main__":
    main()
