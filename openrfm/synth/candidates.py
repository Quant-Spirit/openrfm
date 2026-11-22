"""Candidate worlds of the corpus families: one draw per slot, written and checked for admission."""

import json
import re
import shutil
import subprocess
from dataclasses import replace

import numpy as np

AUTOCOMPLETE_NAMES = {
    "attribute_category": "category",
    "attribute_numeric": "numeric",
    "event_mark": "mark",
}
NUISANCE_ARGS = [
    "--min-tables",
    "3",
    "--max-tables",
    "20",
    "--min-cols",
    "3",
    "--max-cols",
    "24",
    "--min-entity-rows",
    "128",
    "--max-entity-rows",
    "512",
    "--min-activity-rows",
    "512",
    "--max-activity-rows",
    "4096",
    "--exact-aggregates",
    "--nuisance-text",
]
INT32_CLOCK_LIMIT = 2**31 - 86400


def selected_targets(db, family):
    """Task columns of ``db`` that belong to ``family`` as (table, target, task type)."""
    chosen = []
    for task in json.loads((db / "task_metadata.json").read_text())["tasks"]:
        table = task.get("table", task.get("entity_table"))
        target = task.get("target_column", task.get("target_col"))
        if (
            family.startswith("OLD.nuisance.")
            and task.get("existing_family") != family
        ):
            continue
        if family.startswith(("OLD.A4.", "OLD.long.")) and (
            not target.startswith("v6_y_")
            or task.get("v6_label_kind") != family.rsplit(".", 1)[1]
        ):
            continue
        if family in ("OLD.v6_own_history",) or family.startswith("OLD.frank."):
            if not target.startswith("v6_y_"):
                continue
        kind = "regression" if task["task_type"] == "regression" else "classification"
        chosen.append([table, target, kind])
    return chosen


def declare_nuisance(db, prefixes):
    """Declare the retained aggregate columns of a nuisance world as tasks; no values change."""
    path = db / "task_metadata.json"
    doc = json.loads(path.read_text())
    tasks = {
        (
            t.get("table", t.get("entity_table")),
            t.get("target_column", t.get("target_col")),
        ): t
        for t in doc["tasks"]
    }
    for name in json.loads((db / "column_index.json").read_text()):
        column, table = name.rsplit(" of ", 1)
        rule = prefixes.get(column.split("_", 1)[0])
        if rule is None:
            continue
        classes = re.search(r"_k(\d+)(?:_|$)", column)
        kind = "classification" if classes else "regression"
        if kind not in rule["kinds"]:
            raise ValueError(f"unexpected {kind} column {column}")
        task = tasks.setdefault(
            (table, column),
            dict(
                table=table,
                target_column=column,
                task_type=kind,
                sem_type="number" if kind == "regression" else "discrete",
            ),
        )
        task["existing_family"] = rule["group"]
        if classes:
            task.update(num_classes=int(classes[1]), native_discrete_supported=True)
    keys = sorted(tasks)
    for key, task in tasks.items():
        task["drop_columns"] = [list(k) for k in keys if k != key]
    doc["tasks"] = [tasks[k] for k in keys]
    path.write_text(json.dumps(doc) + "\n")


def unified_candidate(cfg, family, index, slot, raw, dest):
    from openrfm.synth.worlds.unified import PriorConfig, generate_world
    from openrfm.synth.worlds.unified_write import write_world

    cover, seed = cfg["coverage"], cfg["seed"]
    rng = np.random.default_rng([seed, index, 31011])
    shares = [
        cover["rare_future_share"],
        cover["cold_share"],
        cover["long_labels_share"],
    ]
    mode = int(rng.choice(4, p=shares + [1 - sum(shares)]))
    large = bool(rng.random() < cover["large_support_share"])
    prior = PriorConfig(
        entities=cover["support_entities"] if large else 192,
        cutoffs_per_entity=cover["support_cutoffs_per_entity"] if large else 4,
        width_range=tuple(cover["width_range"]),
        relation_count_range=tuple(cover["relation_count_range"]),
        label_history_cap=cover["label_history_cap"],
        text_probability=cover["text_probability"],
    )
    kind = "autocomplete" if family.startswith("NEW.auto.") else "future_window"
    if (
        kind == "future_window"
        and rng.random() < cfg["temporal"]["irregular_clock_share"]
    ):
        jitter = tuple(cfg["temporal"]["label_publication_jitter_fraction_range"])
        prior = replace(prior, label_publication_jitter_fraction_range=jitter)
    if kind == "future_window" and mode == 0:
        prior = replace(
            prior,
            mean_events_range=tuple(cover["rare_mean_events_range"]),
            horizon_fraction_range=tuple(cover["rare_horizon_fraction_range"]),
        )
    elif mode == 1:
        prior = replace(prior, mean_events_range=tuple(cover["cold_mean_events_range"]))
    elif mode == 2:
        prior = replace(prior, mean_events_range=tuple(cover["long_mean_events_range"]))
    data = generate_world(
        seed, slot, config=prior, task_kind=kind, functional=family.rsplit(".", 1)[1]
    )
    actual = (
        "NEW.auto." + AUTOCOMPLETE_NAMES[data["functional"]]
        if kind == "autocomplete"
        else "NEW.future." + data["functional"]
    )
    if actual != family:
        return "other_functional"
    if data["span"] * data["seconds_per_unit"] + 946684800 >= INT32_CLOCK_LIMIT:
        return "event_clock_beyond_int32"
    labels = data["labels"]
    if not np.isfinite(labels).all() or len(np.unique(labels)) < 2:
        return "constant_or_nonfinite"
    record = write_world(raw, seed, slot, data=data)
    (raw / record["db_name"]).rename(dest)


def sql_candidate(cfg, family, slot, raw, dest):
    from openrfm.synth.worlds.sql import TEMPLATES
    from openrfm.synth.worlds.sql_oracle import generate, profile
    from openrfm.synth.worlds.sql_write import export

    if (TEMPLATES.index(family.rsplit(".", 1)[1]) - slot) % 9:
        return "other_template"
    data = generate(cfg["seed"], slot, 128, family.split(".")[2] == "peer")
    try:
        stats = profile(
            data["facts"], data["labels"], data["entities"], data["task_type"]
        )
    except ValueError:
        return "profile_failed"
    oracle, single = stats.get("oracle"), (stats.get("single") or [None] * 15)[8]
    regression = data["task_type"] == "regression"
    counts = np.unique(data["labels"], return_counts=True)[1]
    if (
        oracle is None
        or single is None
        or not (0.05 if regression else 0.60) <= oracle < 0.98
        or oracle - single < 0.02
    ):
        return "oracle_out_of_range"
    if not regression and counts.min() / counts.sum() < 0.05:
        return "rare_class"
    export(data, raw, cfg["seed"], slot, 128, dest.name)


def native_candidate(cfg, family, slot, raw, dest, binary):
    """Temporal-kernel worlds from the Rust generator."""
    tmp = raw.parent / "tmp" / dest.name
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    command = [
        binary,
        "--axis",
        "temporal-kernel",
        "--seed-start",
        str(slot),
        "--num-dbs",
        "1",
        "--output-root",
        str(tmp),
    ]
    if family.startswith("OLD.nuisance."):
        command += NUISANCE_ARGS
    else:
        command += ["--v6-stage", "s2" if family.startswith("OLD.long.") else "s1"]
        if family.startswith(("OLD.long.", "OLD.frank.")):
            command += [
                "--v6-variant",
                "long-history" if family.startswith("OLD.long.") else "f-rank",
            ]
        else:
            command += [
                "--v6-archetype",
                "a4" if family.startswith("OLD.A4.") else "a1",
            ]
    result = subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    if result.returncode == 1:
        shutil.rmtree(tmp)
        return "native_candidate_rejected"
    result.check_returncode()
    generated = next(tmp.glob("rel-synthetic-*"))
    if family.startswith("OLD.nuisance."):
        declare_nuisance(generated, cfg["nuisance_prefixes"])
    generated.rename(dest)
    shutil.rmtree(tmp)


def world_candidate(cfg, family, index, slot, raw, dest, binary):
    """Write one candidate world to ``dest``; return a rejection reason or None."""
    if family == "NEW.derived":
        from openrfm.synth.worlds.derived import world

        world(raw, slot)
        (raw / f"rel-synthetic-{8400000 + slot}").rename(dest)
    elif family.startswith(("NEW.auto.", "NEW.future.")):
        return unified_candidate(cfg, family, index, slot, raw, dest)
    elif family.startswith("OLD.PH."):
        from openrfm.synth.worlds.history_world import write_history_world
        from openrfm.synth.worlds.panel import HISTORY_FAMILIES

        if (HISTORY_FAMILIES.index(family.rsplit(".", 1)[1]) - slot) % 6:
            return "other_panel_family"
        write_history_world(raw, dest.name, slot, cfg["seed"])
    elif family.startswith("OLD.rank."):
        from openrfm.synth.worlds.rank import QS, write_rank_world

        if (QS.index(int(family.rsplit("q", 1)[1])) - slot) % 5:
            return "other_rank_quantile"
        write_rank_world(raw, dest.name, slot, cfg["seed"])
    elif family.startswith("OLD.SQL."):
        return sql_candidate(cfg, family, slot, raw, dest)
    else:
        return native_candidate(cfg, family, slot, raw, dest, binary)

