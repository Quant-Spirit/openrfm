"""The training loop: one world per `updates_per_world` updates, the next world prepared on a CPU thread while the
current one trains, and world groups reallocated by the loss memory can still reduce."""
from concurrent.futures import ThreadPoolExecutor
from itertools import count
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from openrfm.models import load_backbone
from openrfm.models.consumers.kumo import KumoHead
from openrfm.models.memory import MemoryConfig, MemoryModule, save_memory
from openrfm.sampler.episodes import configure_backend
from openrfm.train.memory.allocator import GroupAllocator
from openrfm.train.memory.objective import reducible_loss
from openrfm.train.memory.trainer import Trainer
from openrfm.train.memory.worlds import admit, encode_rows, prepare_world

KINDS = ('classification', 'regression')


def log(**fields):
    print(json.dumps(fields), flush=True)


def replay_records(cfg):
    """Replay archives whose tasks have enough rows for the largest support plus the query."""
    objective = cfg['objective']
    need = max(objective['support'])+objective['query']
    records = json.loads(Path(cfg['inputs']['replay_selection']).read_text())['records']
    return [r for r in records if min(r['target_rows']) >= need]


def train(cfg, out, stop, resume):
    """Train to `stop` updates. A resume continues from `out`/latest.pt with the next world; the RNGs restart
    from the seed."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    configure_backend('native', build=True)
    rt = load_backbone(cfg['inputs']['backbone'], 'cuda').requires_grad_(False)
    allocator = GroupAllocator(yaml.safe_load(Path(cfg['data']).read_text())['mix'])
    state = torch.load(out/'latest.pt', map_location='cpu', weights_only=False) if resume else None
    if state:
        trainer = Trainer.from_payload(state['trainer'], cfg['optimizer'], 'cuda')
        allocator.load(state['allocator'])
    else:
        trainer = Trainer(MemoryModule(MemoryConfig(**dict(cfg['memory'], target_depths=tuple(
            cfg['memory']['target_depths'])))).to('cuda'), cfg['optimizer'])
    memory = trainer.memory.bind(rt)
    objective = cfg['objective']
    consumers = {kind: KumoHead(kind, objective['consumer_size']) for kind in KINDS}
    replay = replay_records(cfg)
    first_slot = state['next_world_slot'] if state else 0
    progress = dict(next_world_slot=first_slot)

    def save():
        payload = dict(trainer=trainer.payload(), allocator=allocator.state(), **progress)
        torch.save(payload, out/'latest.tmp')
        (out/'latest.tmp').replace(out/'latest.pt')

    log(event='start', update=trainer.updates, stop=stop)
    with ThreadPoolExecutor(max_workers=1) as worker:
        def submit(slot):
            return worker.submit(prepare_world, slot, cfg, out, replay, allocator.probabilities())
        future = submit(first_slot)
        for slot in count(first_slot):
            prepared = future.result()
            progress['next_world_slot'] = slot
            world_stop = min(stop, (trainer.updates//cfg['updates_per_world']+1)*cfg['updates_per_world'])
            if world_stop < stop or prepared['skip']:
                future = submit(slot+1)
            if prepared['skip']:
                log(event='world_skipped', slot=slot, db=prepared['db'], reason=prepared['skip'])
                progress['next_world_slot'] = slot+1
                continue
            episode = encode_rows(admit(prepared, rt, cfg['sampler'], cfg['backbone_context']['length']), rt,
                                  cfg['backbone_context']['rows'], cfg['seed'], slot, memory.config.target_depths)
            log(event='world', slot=slot, group=prepared['group'], db=prepared['db'], rows=len(episode['roots']))
            train_world(trainer, episode, prepared, world_stop, stop, cfg, consumers, progress, save)
            if trainer.updates == world_stop:
                measure_world(memory, episode, prepared, cfg, allocator)
            del episode
            if prepared['directory'] is not None:
                shutil.rmtree(prepared['directory'])
            progress['next_world_slot'] = slot+1
            if trainer.updates >= stop:
                break
    save()
    save_memory(memory, out/'memory.pt')
    log(event='done', update=trainer.updates, checkpoint=str(out/'memory.pt'))


def train_world(trainer, episode, prepared, world_stop, stop, cfg, consumers, progress, save):
    """Updates on one cached world until `world_stop`; a FloatingPointError ends the world early."""
    objective, seed = cfg['objective'], cfg['seed']
    n_rows = len(episode['roots'])
    while trainer.updates < world_stop:
        update = trainer.updates
        rng = np.random.default_rng([seed, update])
        # The cache is in time order, so the support rows are earlier than the query rows, as on real tasks.
        k = min(int(rng.choice(objective['support'])), n_rows-objective['query'])
        selected = torch.as_tensor(np.sort(rng.choice(n_rows, k+objective['query'], replace=False)), device='cuda')
        try:
            loss, tasks = trainer.step(episode, selected, k, seed=seed+update, consumers=consumers,
                                       objective=objective,
                                       own_task_only=prepared['group'] in objective['own_task_only'])
        except FloatingPointError as error:
            log(event='world_skipped', slot=prepared['slot'], update=update, reason=str(error))
            return
        if update % cfg['log_every'] == 0 or trainer.updates == stop:
            log(update=trainer.updates, group=prepared['group'], task=episode['kind'], K=k, loss=loss, tasks=tasks)
        if trainer.updates % cfg['checkpoint_every'] == 0:
            progress['next_world_slot'] = prepared['slot']+int(trainer.updates == world_stop)
            save()


def measure_world(memory, episode, prepared, cfg, allocator):
    """Reducible loss of the world just trained, on its cached rows split in half (support = the earlier half)."""
    with torch.no_grad():
        memory.eval()
        columns = memory.columns(episode['world'], episode['roots'], episode['cuts'],
                                 context=(episode['states'], episode['live']))
        memory.train()
    measured = reducible_loss(episode['world'], episode['roots'], episode['cuts'], episode['labels'],
                              episode['prefix'], columns, len(episode['roots'])//2, episode['kind'],
                              cfg['seed']+prepared['slot'], cfg['objective']['program_groups'])
    if measured:
        allocator.update(prepared['group'], measured['reducible'])
        log(event='room', slot=prepared['slot'], group=prepared['group'], **measured)
