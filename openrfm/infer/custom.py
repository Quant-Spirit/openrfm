"""`openrfm.predict`: a task of one's own database, from objects to predictions with the original keys."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


def predict(db, task, recipe='r1', work='openrfm-work', name='db', checkpoints=None):
    """Predicts every TEST row of `task` on `db` that the sampler admits, with the recipe's default cell. Returns those
    rows with `id`
    (original entity key), `timestamp`, `prediction` (probability of the positive class, or the value) and the target
    when it is known; the metric is printed when it is."""
    from openrfm.infer.run import run

    work = Path(work)
    data = work/'data'
    db.write(data, name)
    spec = task.write(db, data, name)
    path = work/'specs'/f'{name}__{task.name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    run(None, recipe, data, work, checkpoints=checkpoints, spec=path)
    test = pq.read_table(data/name/'tasks'/task.name/'test.parquet').to_pandas()
    with np.load(work/recipe/f'{name}__{task.name}'/'predictions.npz') as z:
        row, prediction = z['row'], z['p']
    test = test.iloc[row].reset_index(drop=True)
    keys = pq.read_table(data/name/'keys'/f'{task.entity}.parquet').column('id').to_numpy(zero_copy_only=False)
    out = pd.DataFrame({'id': keys[test[db.pkeys[task.entity]].to_numpy(dtype=np.int64)], 'timestamp': test['timestamp']})
    out['prediction'] = prediction
    if task.target in test:
        out[task.target] = test[task.target].to_numpy()
    return out
