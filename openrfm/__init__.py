"""OpenRFM: an open relational foundation model pre-trained on synthetic databases."""
from openrfm.data import Database, Task


def predict(db, task, recipe='r1', work='openrfm-work', name='db', checkpoints=None):
    """Predicts the TEST rows of `task` on `db`; see `openrfm.infer.custom.predict`."""
    from openrfm.infer.custom import predict as _predict
    return _predict(db, task, recipe, work, name, checkpoints)


__all__ = ['Database', 'Task', 'predict']
