"""One memory update and the training checkpoint."""
import torch

from openrfm.models.memory import memory_from_payload, memory_payload
from openrfm.train.memory.objective import kumo_task_losses
from openrfm.train.memory.tasks import program_tasks, scorable


class Trainer:
    """AdamW on the memory module only; the backbone and the Kumo heads stay frozen."""

    def __init__(self, memory, optimizer):
        self.memory = memory
        self.optimizer = torch.optim.AdamW(memory.parameters(), lr=optimizer['lr'],
                                           weight_decay=optimizer['weight_decay'])
        self.updates = 0

    def step(self, episode, selected, k, *, seed, consumers, objective, own_task_only):
        """One update on the rows `selected` of a cached world (support = the first `k`). The tasks are the world's
        own task and, unless `own_task_only`, its program and bank tasks. Raises FloatingPointError, before the
        optimizer steps, when no task is scorable or a loss or gradient is not finite."""
        self.memory.train()
        self.optimizer.zero_grad(set_to_none=True)
        world, rows, cutoffs = episode['world'], episode['roots'][selected], episode['cuts'][selected]
        features = self.memory.columns(world, rows, cutoffs,
                                       context=(episode['states'][selected], episode['live'][selected]))
        tasks = [(episode['kind'], episode['labels'][selected])]
        if not own_task_only:
            bank = episode['bank_targets']
            tasks += program_tasks(world, rows, cutoffs, k, seed, None if bank is None else bank[selected], objective)
        tasks = scorable(tasks, k)
        if not tasks:
            raise FloatingPointError('no scorable task in this episode')
        losses = kumo_task_losses(consumers, episode['prefix'][selected], features, tasks, k, seed)
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.memory.parameters()):
            raise FloatingPointError('nonfinite memory gradient')
        self.optimizer.step()
        self.updates += 1
        return float(torch.tensor(sum(losses)/len(losses))), len(losses)

    def payload(self):
        return dict(memory=memory_payload(self.memory), optimizer=self.optimizer.state_dict(), updates=self.updates)

    @classmethod
    def from_payload(cls, payload, optimizer, device):
        trainer = cls(memory_from_payload(payload['memory']).to(device), optimizer)
        trainer.optimizer.load_state_dict(payload['optimizer'])
        trainer.updates = payload['updates']
        return trainer
