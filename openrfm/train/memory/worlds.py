"""Training worlds: a synthetic memory world generated and converted online, or an archive replayed from the backbone
corpus; its history world, task rows and labels; and the frozen backbone columns of a cached set of rows."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
import yaml

from openrfm.archive.history import decode
from openrfm.models.memory import encode_context, scalar_matrix
from openrfm.synth.worlds.memory import write_world
from openrfm.train.memory.panel import PanelSampler
from openrfm.train.memory.tasks import align_targets, execute_bank, load_bank, task_rows
from openrfm.train.memory.world import build_world

REPLAY = 'corpus_replay'
FIRST_DB = 7200000


def needs_raw_keys(archive):
    """True when the archive's schema lacks foreign keys or has parallel foreign keys to one table; such an archive
    needs its source Parquet to name its keys."""
    meta = json.loads((Path(archive)/'table_info.json').read_text())
    return any(info.get('fkey_col_to_pkey_table') is None
               or len(set(info['fkey_col_to_pkey_table'].values())) < len(info['fkey_col_to_pkey_table'])
               for info in meta.values())


def prepare_world(slot, cfg, out, replay, probabilities):
    """World `slot`: its group drawn with `probabilities`, generated and converted (or picked from `replay`)."""
    seed = cfg['seed']+slot
    rng = np.random.default_rng(seed)
    groups = list(yaml.safe_load(Path(cfg['data']).read_text())['mix'])
    group = str(rng.choice(groups, p=probabilities))
    if group == REPLAY:
        record = replay[int(rng.integers(len(replay)))]
        root, db = Path(cfg['inputs']['replay_archives']), record['db']
        selected = {(target[0], target[1]) for target in record['targets']}
        tasks = [task for task in task_rows(root, db) if (task[1], task[2]) in selected]
        task = tasks[int(rng.integers(len(tasks)))]
        skip = 'archive keys need its source Parquet' if needs_raw_keys(root/db) else None
        return dict(root=root, db=db, group=group, slot=slot, directory=None, bank=None, tasks=[task], skip=skip)
    directory = Path(out)/'worlds'/str(slot)
    if directory.exists():
        shutil.rmtree(directory)
    raw, pre, text = (directory/name for name in ('raw', 'pre', 'text'))
    raw.mkdir(parents=True)
    pre.mkdir()
    text.mkdir()
    db = f'rel-synthetic-{FIRST_DB+slot}'
    write_world(raw, db, seed, cfg['data'], prior=yaml.safe_load(Path(cfg['prior']).read_text()),
                signal=group if group.endswith('_signal') else None)
    convert(cfg['inputs']['rustler'], raw, pre, text, db, directory/'pre.log')
    shutil.copy2(raw/db/'task_metadata.json', pre/db/'task_metadata.json')
    metadata = json.loads((raw/db/'task_metadata.json').read_text())['tasks']
    bank = execute_bank(raw/db, metadata, load_bank(cfg['llm_bank']), seed, columns=2*cfg['objective']['llm_programs'])
    return dict(root=pre, db=db, group=group, slot=slot, directory=directory, bank=bank, tasks=task_rows(pre, db),
                skip=None)


def convert(rustler, raw, pre, text, db, log):
    """`rustler pre` of one world into a v1 archive; text is embedded on the CPU."""
    env = dict(os.environ, RELBENCH_DATA_ROOT=str(raw), RELBENCH_PRE_ROOT=str(pre), RUSTLER_TEXT_EMB_CACHE=str(text))
    with open(log, 'w') as stream:
        subprocess.run([rustler, 'pre', db, '--archive-format', 'v1', '--text-encoder', 'all-MiniLM-L12-v2',
                        '--text-encoder-device', 'cpu', '--allow-cpu'], env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def admit(prepared, rt, sampler, length):
    """The history world, roots, cutoffs, labels and LLM bank targets of the world's task for this slot (worlds
    with two tasks alternate between them)."""
    root, db = prepared['root'], prepared['db']
    task = prepared['tasks'][prepared['slot'] % len(prepared['tasks'])]
    meta, = [m for m in json.loads((root/db/'task_metadata.json').read_text())['tasks']
             if (m['table'], m['target_column']) == (task[1], task[2])]
    decoded = decode(root/db)
    panel = PanelSampler(root, task, sampler, length=length)
    roots = torch.as_tensor(panel.roots, dtype=torch.long)
    world = build_world(decoded, rt, panel.graph)
    v = decoded['values']
    ix = torch.where(decoded['cols'] == panel.target)[0]
    sem = v['cell_types'].long()[ix]
    labels = torch.zeros(world.nrows)
    labels[decoded['nodes'].long()[ix]] = torch.where(sem == 4, v['cell_codes'].long()[ix].float(), torch.where(
        sem == 3, v['cell_booleans'].float()[ix], v['cell_numbers'].float()[ix]))
    kind = 'regression' if meta['task_type'] == 'regression' else 'classification'
    return dict(world=world, panel=panel, roots=roots.cuda(), cuts=panel.graph.row_time[roots].long().cuda(),
                labels=labels[roots].cuda(), kind=kind, bank_targets=bank_targets(prepared, meta, world, roots, panel))


def bank_targets(prepared, meta, world, roots, panel):
    """The LLM bank rows of the roots, keyed by the root's entity and clock; None for a replayed world."""
    result = (prepared['bank'] or {}).get(meta['table'])
    if result is None or result['targets'] is None:
        return None
    info = json.loads((prepared['root']/prepared['db']/'table_info.json').read_text())
    entity = info[meta['forecast_entity_table']+':Db']
    parents = world.parents[roots.to(world.parents.device)].cpu()
    offset = entity['node_idx_offset']
    inside = (parents >= offset) & (parents < offset+entity['num_nodes'])
    rows = align_targets(result, (parents[inside]-offset).numpy(), panel.graph.row_time[roots].cpu().numpy())
    if (rows < 0).any():
        print(json.dumps(dict(event='bank_unaligned', db=prepared['db'], missing=int((rows < 0).sum()))), flush=True)
        return None
    return torch.as_tensor(result['targets'][rows]).cuda()


def encode_rows(episode, rt, count, seed, world_seed, depths):
    """`count` random roots in time order (ties in random order) with their frozen backbone columns: the target cell
    at `depths`, then the focal FP32 scalars; and the context states the memory reads."""
    rng = np.random.default_rng([seed, world_seed])
    chosen = torch.as_tensor(rng.choice(len(episode['roots']), min(count, len(episode['roots'])), replace=False),
                             device='cuda')
    cuts = episode['cuts'][chosen]
    order = torch.as_tensor(np.lexsort((rng.permutation(len(chosen)), cuts.cpu().numpy())), device='cuda')
    rows, cuts = episode['roots'][chosen][order], cuts[order]
    states, live, triples, targets = [], [], [], []
    ids = rows.cpu()
    for a in range(0, len(ids), 8):
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in episode['panel'].sample(seed+a, ids[a:a+8]).items()}
        x, m, (r, k, v), z = encode_context(rt, batch)
        states.append(x)
        live.append(m)
        triples.append((r+a, k, v))
        targets.append(z)
    scalars, _ = scalar_matrix(tuple(torch.cat(t) for t in zip(*triples)), len(ids))
    prefix = torch.cat((torch.cat(targets)[:, list(depths)].flatten(1), scalars), 1)
    bank = episode['bank_targets']
    return dict(world=episode['world'], kind=episode['kind'], roots=rows, cuts=cuts,
                labels=episode['labels'][chosen][order], prefix=prefix, states=torch.cat(states),
                live=torch.cat(live), bank_targets=None if bank is None else bank[chosen[order]])
