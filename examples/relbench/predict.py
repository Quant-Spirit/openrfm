"""Predict RelBench (or 4DBInfer) tasks with a released recipe and print each TEST metric.

    python examples/relbench/predict.py --data DATA                                   # the three rel-f1 tasks
    python examples/relbench/predict.py --data DATA --recipe r1 --tasks rel-event/user-repeat rel-trial/study-outcome
    python examples/relbench/predict.py --data DATA --tasks all                       # all 32 benchmark tasks
"""
import argparse
from pathlib import Path

from openrfm.infer import recipe
from openrfm.infer.run import run

F1 = ['rel-f1/driver-dnf', 'rel-f1/driver-top3', 'rel-f1/driver-position']


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', type=Path, required=True, help='databases are downloaded into DATA/<db>/ when absent')
    p.add_argument('--recipe', choices=('r1', 'r2', 'r3'), default='r2')
    p.add_argument('--tasks', nargs='+', default=F1, help="benchmark tasks, or 'all'")
    p.add_argument('--work', type=Path, default=Path('openrfm-work'))
    a = p.parse_args()
    tasks = list(recipe.load('tasks.yaml')['tasks']) if a.tasks == ['all'] else a.tasks
    rows = []
    for task in tasks:
        r = run(task, a.recipe, a.data, a.work)
        rows.append((task, r['metric'], r['value'], r['SE']))
    print(f"\n{'task':40s} {'metric':6s} {'value':>8s} {'SE':>7s}")
    for task, metric, value, se in rows:
        print(f'{task:40s} {metric:6s} {value:8.4f} {se:7.4f}')


if __name__ == '__main__':
    main()
