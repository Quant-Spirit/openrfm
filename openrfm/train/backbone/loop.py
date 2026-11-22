"""The training loop: gradient accumulation over micro-batches, the consumer pass, balance, clipping and updates."""
import json
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grads_with_norm_, get_total_norm

from openrfm.sampler.episodes import configure_backend

from .balance import GradientBalance
from .consumer import Consumer
from .coupling import Coupling, preserved_rng
from .data import BatchSource
from .masking import prepare_micro_batch
from .model import build_modules
from .objective import column_slot_loss
from .optim import build_optimizer


class Trainer:
    def __init__(self, config, corpus: Path, out: Path, device="cuda"):
        self.config, self.out, self.device = config, out, torch.device(device)
        configure_backend("native", build=True)
        self.rt, self.head, self.handoff, self.named = build_modules(config, self.device)
        self.source = BatchSource(config, corpus, self.device)
        # TabICL's constructor draws its (immediately overwritten) initial weights from the global RNG.
        with preserved_rng():
            consumers = {task: Consumer(corpus / c["file"], c["sha256"], task == "regression", self.device)
                         for task, c in (("classification", config["coupling"]["consumers"]["classifier"]),
                                         ("regression", config["coupling"]["consumers"]["regressor"]))}
        self.coupling = Coupling(config, self.source.groups, self.source.samplers, self.source.weights,
                                 corpus / config["archives"], consumers, self.handoff)
        self.optimizer, self.scheduler = build_optimizer(self.named, config)
        self.parameters = [p for _, p in self.named]
        self.balance = GradientBalance([p for name, p in self.named if not name.startswith("rt_handoff.")])
        self.step, self.consumed = 0, 0

    def state(self) -> dict:
        return dict(state_dict=self.rt.state_dict(), step=self.step, consumed_train_batches=self.consumed,
                    aux_heads=dict(column_slot_aux_head={k: v.detach().cpu() for k, v in self.head.state_dict().items()}),
                    rt_handoff=self.handoff.export(), optimizer=self.optimizer.state_dict(),
                    scheduler=self.scheduler.state_dict(), coupling=self.coupling.state_dict(),
                    rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))

    def save(self) -> Path:
        path = self.out / f"steps={self.step}.pt"
        self.out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(), path)
        return path

    def resume(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.rt.load_state_dict(state["state_dict"])
        self.head.load_state_dict(state["aux_heads"]["column_slot_aux_head"])
        self.handoff.load_state_dict(state["rt_handoff"]["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.coupling.load_state_dict(state["coupling"])
        torch.set_rng_state(state["rng"]["cpu"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        self.step, self.consumed = state["step"], state["consumed_train_batches"]

    def update(self) -> dict:
        """One optimizer update over ``grad_accum_steps`` micro-batches; returns its loss components."""
        o = self.config["objective"]
        accum = self.config["grad_accum_steps"]
        epoch = self.consumed // self.source.size
        self.optimizer.zero_grad(set_to_none=True)
        self.balance.window_start()
        loss_sum = units_sum = None
        valid = torch.ones((), dtype=torch.bool, device=self.device)
        masked, slots, coupled = [], [], {}
        window = self.source.window(self.consumed, accum, epoch)
        for micro in range(accum):
            self.rt.train()
            batch = window[micro]
            self.consumed += 1
            batch = prepare_micro_batch(batch, regression_tasks=self.source.regression_tasks, step=self.step,
                                        seed=self.config["seed"], p_max=o["multi_cell_mask_p_max"],
                                        support_fraction=o["support_fraction"])
            loss_main, outputs = self.rt(batch)
            loss = loss_main
            loss_slot = column_slot_loss(batch, outputs["hidden"], self.rt.enc_dict["col_name"], self.head)
            if loss_slot is not None:
                loss = loss + o["column_slot_weight"] * loss_slot
            units = batch["masks"].new_tensor(batch["masks"].shape[0], dtype=torch.float32)
            valid &= (units > 0) & torch.isfinite(loss.detach())
            (loss * units.to(loss.dtype)).backward()
            self.balance.before_coupling()
            if micro + 1 == accum:
                coupled = self.coupling.accumulate(self.step, units if units_sum is None else units_sum + units,
                                                   self.rt, self.parameters)
            self.balance.after_coupling()
            numerator = loss.detach().to(torch.float32) * units
            loss_sum = numerator if loss_sum is None else loss_sum + numerator
            units_sum = units if units_sum is None else units_sum + units
            masked.append(loss_main.detach().float())
            slots.append(None if loss_slot is None else loss_slot.detach().float())
        grads = [p.grad for p in self.parameters if p.grad is not None]
        raw_norm = get_total_norm(grads, foreach=True)
        if not bool(torch.isfinite(loss_sum) & (units_sum > 0) & valid & torch.isfinite(raw_norm)):
            raise FloatingPointError(f"non-finite loss or gradients at update {self.step + 1}")
        start = self.coupling.start_step
        self.balance.finish(0.0 if start is None else self.coupling.weight(self.step) / float(self.config["coupling"]["weight"]))
        torch._foreach_mul_(grads, 1.0 / units_sum)
        trainable = [p for p in self.parameters if p.requires_grad]
        grad_norm = get_total_norm([p.grad for p in trainable if p.grad is not None], foreach=True)
        clip_grads_with_norm_(trainable, max_norm=self.config["optimizer"]["max_grad_norm"], total_norm=grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return dict(step=self.step, loss=float(loss_sum / units_sum), grad_norm=float(grad_norm),
                    masked=[float(x) for x in masked], column_slot=[None if x is None else float(x) for x in slots],
                    icl=None if "loss_icl" not in coupled else float(coupled["loss_icl"]))

    def train(self, steps: int) -> None:
        """Run updates until ``steps`` (the schedule always spans ``optimizer_steps``), checkpointing on the way."""
        if self.step == 0:
            self.save()
        with (self.out / "log.jsonl").open("a") as log:
            while self.step < steps:
                started = time.time()
                record = self.update()
                record["seconds"] = time.time() - started
                log.write(json.dumps(record) + "\n")
                log.flush()
                if self.step % self.config["checkpoint_every"] == 0 or self.step == steps:
                    self.save()
