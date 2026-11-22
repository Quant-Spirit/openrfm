"""Train the memory module over the frozen backbone.

    python -m openrfm.train.memory --config openrfm/configs/train/memory.yaml --corpus CORPUS --out RUN [--resume] [--steps N]

CORPUS is the backbone corpus of `openrfm train backbone fetch`; the backbone is downloaded unless --checkpoints names a
directory holding openrfm_backbone.pt.

The run writes RUN/latest.pt (resumable, every `checkpoint_every` updates and at the end) and RUN/memory.pt (the
memory module, loadable with `openrfm.models.memory.load_memory`)."""
import argparse
from pathlib import Path

import yaml

from openrfm import hub
from openrfm.train.memory.campaign import train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path('openrfm/configs/train/memory.yaml'))
    parser.add_argument('--corpus', type=Path, required=True, help='the corpus written by `openrfm train backbone fetch`')
    parser.add_argument('--checkpoints', type=Path, help='directory holding openrfm_backbone.pt (default: download)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='continue from OUT/latest.pt')
    parser.add_argument('--steps', type=int, help='stop after this many updates in total (default: `updates`)')
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    inputs = cfg['inputs']
    inputs['backbone'] = hub.checkpoint('backbone', args.checkpoints)
    for key in ('replay_archives', 'replay_selection'):
        inputs[key] = args.corpus/inputs[key]
    train(cfg, args.out, cfg['updates'] if args.steps is None else args.steps, args.resume)


if __name__ == '__main__':
    main()
