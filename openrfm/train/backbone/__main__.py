"""``python -m openrfm.train.backbone fetch|corpus|train|export``."""
import argparse
import json
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> None:
    """Fetch or assemble the training corpus, train the backbone, or export a training checkpoint as the released file."""
    parser = argparse.ArgumentParser(prog="python -m openrfm.train.backbone")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download the released pretraining corpus")
    fetch.add_argument("--config", type=Path, required=True)
    fetch.add_argument("--out", type=Path, required=True)
    corpus = commands.add_parser("corpus", help="assemble the training corpus from `openrfm generate` output")
    corpus.add_argument("--config", type=Path, required=True)
    corpus.add_argument("--generated", type=Path, required=True, help="the --out directory of `openrfm generate`")
    corpus.add_argument("--out", type=Path, required=True)
    train = commands.add_parser("train", help="pretrain the backbone")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--resume", type=Path, help="training checkpoint to continue from")
    train.add_argument("--steps", type=int, help="stop after this many updates (default: optimizer_steps)")
    export = commands.add_parser("export", help="write the released backbone file of a training checkpoint")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        from .export import export_backbone
        export_backbone(args.checkpoint, args.out)
        return
    config = yaml.safe_load(args.config.read_text())
    if args.command == "fetch":
        from .fetch import fetch
        print(json.dumps(fetch(args.out, config)))
        return
    if args.command == "corpus":
        from .corpus import assemble
        print(json.dumps(assemble(args.generated, args.out, config)))
        return
    from .loop import Trainer
    trainer = Trainer(config, args.corpus, args.out)
    if args.resume is not None:
        trainer.resume(args.resume)
    trainer.train(args.steps or config["optimizer_steps"])


if __name__ == "__main__":
    main()
