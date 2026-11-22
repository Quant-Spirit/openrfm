"""Assemble the backbone corpus from ``openrfm generate`` output: one archive per world, the task catalog that groups
the worlds by family and the two frozen TabICL consumers.

    CORPUS/sources/<world>/       v2 archive of each generated world (fp32 datetimes, with the world's sidecars)
    CORPUS/corpus/<alias>         link to its archive; aliases are numbered family by family in ``families`` order
    CORPUS/corpus/compact_task_catalog.json
    CORPUS/consumers/{classifier,regressor}.ckpt
"""
import json
import shutil
from pathlib import Path

from openrfm.hub import sha256
from openrfm.synth.generate import TABULAR_FAMILIES

FIRST_ALIAS = 50000000
TEXT_ENCODER = "all-MiniLM-L12-v2"
TABULAR_MANIFESTS = Path(__file__).resolve().parents[2] / "configs/data/tabular"


def _tasks(world: Path) -> list[list[str]]:
    """``[table, target, kind]`` of every task a world declares."""
    tasks = json.loads((world / "task_metadata.json").read_text())["tasks"]
    return [[t.get("table", t.get("entity_table")), t.get("target_column", t.get("target_col")),
             "regression" if t["task_type"] == "regression" else "classification"] for t in tasks]


def generated_worlds(generated: Path) -> list[tuple[str, Path, list]]:
    """``(family, world directory, targets)`` of every finished world under ``generated``."""
    worlds = []
    for receipt in sorted((generated / "receipts").glob("*.json")):
        record = json.loads(receipt.read_text())
        if "db" in record:
            worlds.append((record["family"], generated / "raw" / record["db"], record["targets"]))
        elif "worlds" in record:
            for name in record["worlds"]:
                world = generated / "raw" / name
                family = json.loads((world / "a5_world.json").read_text())["family_id"]
                worlds.append((family, world, _tasks(world)))
    tabular = generated / "tabular" / "raw"
    if tabular.is_dir():
        names = {name: group for group, name in TABULAR_FAMILIES.items()}
        family_of = {r["db_name"]: names[r["family"]] for path in sorted(TABULAR_MANIFESTS.glob("replica*.json"))
                     for r in json.loads(path.read_text())["records"]}
        for world in sorted(tabular.glob("rel-synthetic-*")):
            worlds.append((family_of[world.name], world, _tasks(world)))
    return worlds


def fetch_consumers(config: dict, out: Path) -> None:
    """Download the pinned TabICL v2 checkpoints into ``out``."""
    from huggingface_hub import hf_hub_download

    for consumer in config["coupling"]["consumers"].values():
        path = out / consumer["file"]
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hf_hub_download(consumer["repo"], consumer["source"]), path)
        if sha256(path) != consumer["sha256"]:
            raise ValueError(f"{path} does not match its pinned sha256")


def assemble(generated: Path, out: Path, config: dict) -> dict:
    """Build the corpus of ``config`` (families, archives, consumers) from ``generated``; returns its counts."""
    import rustler_archive

    by_family = {}
    for family, world, targets in generated_worlds(generated):
        by_family.setdefault(family, []).append((world, targets))
    unknown = set(by_family) - {name for name, _ in config["families"]}
    if unknown:
        raise ValueError(f"generated families missing from the configuration: {sorted(unknown)}")
    sources, union = out / "sources", out / config["archives"]
    for directory in (sources, union):
        directory.mkdir(parents=True, exist_ok=True)
    catalog, alias = [], FIRST_ALIAS
    for family, _ in config["families"]:
        for world, targets in sorted(by_family.get(family, [])):
            if not (sources / world.name).exists():
                rustler_archive.convert(world.name, str(world.parent), str(sources), archive_format="v2",
                                        text_encoder=TEXT_ENCODER, fp32_datetimes=True)
            name = f"rel-synthetic-{alias}"
            link = union / name
            if not link.is_symlink():
                link.symlink_to(Path("..") / "sources" / world.name)
            catalog += [dict(db_name=name, component=family, table=table, target=target, task_type=kind)
                        for table, target, kind in sorted(targets)]
            alias += 1
    (union / "compact_task_catalog.json").write_text(json.dumps(dict(records=catalog)) + "\n")
    fetch_consumers(config, out)
    return dict(worlds=alias - FIRST_ALIAS, tasks=len(catalog), families=sorted(by_family))
