"""World-group allocator: the next world's group is drawn in proportion to the loss memory can still reduce there."""
import numpy as np


class GroupAllocator:
    """Exponential moving average of each group's measured reducible loss; an unseen group counts as the mean of the
    seen ones (1 when none is seen)."""

    def __init__(self, groups, decay=.9):
        self.groups, self.decay = list(groups), decay
        self.reducible, self.seen = dict.fromkeys(self.groups, 0.), dict.fromkeys(self.groups, 0)

    def probabilities(self):
        known = [self.reducible[g] for g in self.groups if self.seen[g]]
        fill = sum(known)/len(known) if known else 1.
        raw = np.array([self.reducible[g] if self.seen[g] else fill for g in self.groups])
        return raw/raw.sum() if raw.sum() > 0 else np.full(len(raw), 1/len(raw))

    def update(self, group, value):
        rate = 1-self.decay if self.seen[group] else 1.
        self.reducible[group] += rate*(max(0., value)-self.reducible[group])
        self.seen[group] += 1

    def state(self):
        return dict(reducible=self.reducible, seen=self.seen)

    def load(self, state):
        self.reducible, self.seen = dict(state['reducible']), dict(state['seen'])
