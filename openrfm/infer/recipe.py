"""Inference configuration: recipes (openrfm/configs/infer/r1.yaml, r2.yaml, r3.yaml), task protocol and sampler."""
import importlib.resources
import json
from pathlib import Path

import yaml


def configs():
    """The openrfm/configs directory."""
    return Path(str(importlib.resources.files('openrfm')/'configs'))


def load(name):
    """One YAML file of openrfm/configs/infer."""
    return yaml.safe_load((configs()/'infer'/name).read_text())


def task_config(task, recipe, spec=None):
    """(recipe cell, task protocol) of benchmark task `task`, or of the task that YAML file `spec` declares under the
    recipe's default cell."""
    if recipe == 'r3':
        cell, proto = task_config(task, 'r2', spec)
        r3 = load('r3.yaml')
        head = r3['default'][proto['task_type']] if spec is not None else r3['tasks'][task]
        return dict(cell, **head), proto
    if spec is not None:
        proto = yaml.safe_load(Path(spec).read_text())
        return dict(load(f'{recipe}.yaml')['default'][proto['task_type']]), proto
    tasks = load('tasks.yaml')['tasks']
    if task not in tasks:
        raise ValueError(f'{task}: no inference protocol in openrfm/configs/infer/tasks.yaml')
    return load(f'{recipe}.yaml')['tasks'][task], tasks[task]


def program(task):
    """The schema program of `task`."""
    return json.loads((configs()/'rewrite/programs'/(task.replace('/', '__')+'.json')).read_text())

