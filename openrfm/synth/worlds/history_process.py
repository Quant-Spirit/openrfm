"""Latent per-entity processes, events and labels of panel-history worlds."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from openrfm.synth.worlds.panel import (
    HISTORY_ENTITY_RANGE, HISTORY_FAMILIES, HISTORY_HORIZON_DAYS, HISTORY_MAX_EVENTS, HISTORY_ROWS_RANGE, HISTORY_STEPS,
    HISTORY_STEP_DAYS, PARENT_RANGE,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _history_prior(rng: np.random.Generator, family: str, E: int) -> dict[str, Any]:
    """Family parameters: ar = (rho, sd) of the AR(1) deviation, smu = sd of the entity mean around its group."""
    u = rng.uniform
    if family == "drift":
        return dict(rho=u(0.85, 0.99), sa=u(1.1, 1.6), smu=u(0.5, 0.8))
    if family == "meanrev":
        return dict(rho=u(0.0, 0.15), sa=u(1.4, 1.8), smu=u(0.7, 1.0))
    if family == "seasonal":
        return dict(
            rho=0.5,
            sa=0.4,
            smu=u(0.8, 1.2),
            amp=u(1.2, 1.8),
            period=int(rng.choice([8, 12, 16])),
            phi=u(0, 2 * np.pi, E),
        )
    if family == "regime":
        taus = np.sort(rng.integers(int(0.3 * HISTORY_STEPS), int(0.9 * HISTORY_STEPS), (E, 2)), axis=1)
        taus[rng.random(E) < 0.5, 1] = (
            HISTORY_STEPS + 99
        )  # one change point for half of the entities
        return dict(
            rho=0.5,
            sa=0.4,
            smu=u(0.5, 0.8),
            taus=taus,
            levels=u(1.4, 2.0) * rng.normal(size=(E, 3)),
        )
    if family == "window":
        return dict(
            rho=0.93,
            sa=u(1.0, 1.4),
            smu=u(0.5, 0.8),
            W=int(rng.choice([4, 16, 64])),
            kappa=u(2.5, 3.5),
            gamma=u(0.3, 0.6),
        )
    return dict(
        rho=u(0.85, 0.99),
        sa=u(1.0, 1.4),
        smu=u(0.5, 0.8),
        b=rng.choice([-1.0, 1.0], 2) * u(0.8, 1.4, 2),
    )


def _history_static(rng: np.random.Generator, family: str, kind: str) -> dict[str, Any]:
    """Structure fixed before the time loop: groups, entity means, start steps, task rows, tabular nuisance."""
    E, G = int(rng.integers(*HISTORY_ENTITY_RANGE)), HISTORY_STEPS
    ng = max(6, E // int(rng.integers(10, 26)))
    grp = rng.integers(0, ng, E)
    sec = np.where(rng.random(E) < 0.4, rng.integers(0, ng, E), -1)
    sec[sec == grp] = -1
    member = np.zeros((E, ng), np.int64)
    member[np.arange(E), grp] = 1
    member[np.flatnonzero(sec >= 0), sec[sec >= 0]] = 1
    gmean = rng.uniform(0.7, 1.2) * rng.normal(size=ng)
    p = dict(
        E=E,
        G=G,
        ng=ng,
        family=family,
        kind=kind,
        grp=grp,
        member=member,
        heavy=kind == "regression" and rng.random() < 0.5,
    )
    p.update(_history_prior(rng, family, E))
    if family == "window":
        lam_base = {4: 1.2, 16: 1.8, 64: 3.5}[p["W"]]
    else:
        lam_base = rng.uniform(0.9, 1.8)
    p["lam"] = lam_base * np.exp(0.35 * rng.normal(size=E))
    p["base"] = member @ gmean / member.sum(1) + rng.uniform(-0.6, 0.6)
    p["mu"] = p["base"] + p["smu"] * rng.normal(size=E)
    p["c"] = rng.uniform(-0.4, 0.4)
    p["sv"], p["sy"] = rng.uniform(0.6, 1.0), rng.uniform(0.3, 0.7)
    p["start"] = (G * 0.5 * rng.random(E) ** 2).astype(np.int64)
    pairs = []
    for e in range(E):
        k = int(rng.integers(*HISTORY_ROWS_RANGE)) - 1
        later = np.arange(p["start"][e] + 1, G)
        pairs.append((e, int(p["start"][e])))  # the entity's first appearance: a cold-start row
        pairs.extend(
            (e, int(g)) for g in rng.choice(later, size=min(k, len(later)), replace=False)
        )
    p["pairs"] = [pairs[i] for i in rng.permutation(len(pairs))]
    npar = int(rng.integers(*PARENT_RANGE))
    p.update(
        parent_fk=rng.integers(0, npar, E),
        parents=npar,
        parent_x=rng.normal(size=(npar, 2)),
        category=rng.integers(0, 8, E),
        static_x=rng.normal(size=(E, 2)),
        mask=rng.random((E, G, 2)) < 0.10,
        smask=rng.random((E, 2)) < 0.10,
    )
    return p


def _history_noise(rng: np.random.Generator, E: int, G: int) -> dict[str, np.ndarray]:
    """Every time-indexed random input, axis 1 = step; step g's label and events use only entries <= g."""
    K = HISTORY_MAX_EVENTS
    return dict(
        innov=rng.normal(size=(E, G)),
        act=rng.normal(size=(E, G)),
        arr=rng.exponential(size=(E, G, K)),
        off=rng.integers(0, HISTORY_STEP_DAYS, (E, G, K)),
        eps=rng.normal(size=(E, G, K)),
        u=rng.random((E, G, K)),
        nz=rng.normal(size=(E, G, K, 2)),
        ye=rng.normal(size=(E, G)),
        yc=rng.chisquare(3, (E, G)),
        yu=rng.random((E, G)),
        x=rng.normal(size=(E, G, 2)),
    )


def _history_ar(innov: np.ndarray, rho: float, sd: float) -> np.ndarray:
    a = np.empty_like(innov)
    a[:, 0] = sd * innov[:, 0]
    k = np.sqrt(1 - rho**2) * sd
    for g in range(1, innov.shape[1]):
        a[:, g] = rho * a[:, g - 1] + k * innov[:, g]
    return a


def _history_latent(p: Mapping[str, Any], nz: Mapping[str, np.ndarray]) -> np.ndarray:
    """z[e,g]: log-odds / level of entity e at step g (an AR(1) deviation plus the family's structure)."""
    g = np.arange(p["G"])
    z = p["mu"][:, None] + _history_ar(nz["innov"], p["rho"], p["sa"])
    if (
        p["family"] == "regime"
    ):  # the level is redrawn at each change point: old history misleads
        k = (g[None, None] >= p["taus"][:, :, None]).sum(1)
        z = p["base"][:, None] + np.take_along_axis(p["levels"], k, 1) + (z - p["mu"][:, None])
    if p["family"] == "seasonal":
        z = z + p["amp"] * np.sin(2 * np.pi * g[None] / p["period"] + p["phi"][:, None])
    return z


def _history_events(
    p: Mapping[str, Any], nz: Mapping[str, np.ndarray], z: np.ndarray
) -> dict[str, np.ndarray]:
    """Raw outcomes per entity sorted by (entity, day): Poisson arrivals, value = z + noise (window: Bernoulli)."""
    E, G = z.shape
    lam = np.repeat(p["lam"][:, None], G, 1)
    if p["family"] == "window":
        lam = lam * np.exp(0.5 * _history_ar(nz["act"], 0.8, 1.0))  # time-varying exposure
    live = np.arange(G)[None] >= p["start"][:, None]
    exist = (np.cumsum(nz["arr"], 2) / lam[..., None] <= 1) & live[..., None]
    ie, ig, ik = np.nonzero(exist)
    day = HISTORY_STEP_DAYS * ig + nz["off"][ie, ig, ik]
    if p["family"] == "window":
        val = (nz["u"][ie, ig, ik] < _sigmoid(z[ie, ig])).astype(np.float64)
    else:
        val = z[ie, ig] + p["sv"] * nz["eps"][ie, ig, ik]
        val = np.sinh(val) if p["heavy"] else val
    order = np.lexsort((ik, day, ie))
    ie = ie[order]
    return dict(
        ent=ie,
        step=ig[order],
        day=day[order],
        val=val[order],
        f1=nz["nz"][ie, ig[order], ik[order], 0],
        f2=nz["nz"][ie, ig[order], ik[order], 1],
        ptr=np.searchsorted(ie, np.arange(E + 1)),
    )


def _history_score(
    p: Mapping[str, Any],
    nz: Mapping[str, np.ndarray],
    z: np.ndarray,
    ev: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Noiseless label score s[e,g]; window: rate over the last W events up to the label horizon + exposure."""
    E, G = z.shape
    if p["family"] == "nohist":
        return p["b"][0] * nz["x"][..., 0] + p["b"][1] * nz["x"][..., 1] + p["c"]
    if p["family"] != "window":
        return z
    gs, s, expo = HISTORY_STEP_DAYS * np.arange(G), np.zeros((E, G)), np.zeros((E, G))
    for e in range(E):
        d, v = (
            ev["day"][ev["ptr"][e] : ev["ptr"][e + 1]],
            ev["val"][ev["ptr"][e] : ev["ptr"][e + 1]],
        )
        cs = np.concatenate([[0.0], np.cumsum(v)])
        hi = np.searchsorted(d, gs + HISTORY_HORIZON_DAYS)
        lo = np.maximum(0, hi - p["W"])
        r = np.where(hi > lo, (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1), 0.5)
        s[e] = p["kappa"] * (2 * r - 1)
        expo[e] = np.log1p(hi - np.searchsorted(d, gs - 3 * HISTORY_STEP_DAYS))
    centre = np.log1p(
        p["lam"].mean() * (3 + HISTORY_HORIZON_DAYS / HISTORY_STEP_DAYS)
    )  # static: a data-dependent mean would leak the future
    return s + p["gamma"] * (expo - centre) + p["c"]


def _history_labels(p: Mapping[str, Any], nz: Mapping[str, np.ndarray], s: np.ndarray) -> np.ndarray:
    if p["kind"] == "classification":
        return (nz["yu"] < _sigmoid(s)).astype(np.float64)
    noise = nz["ye"] / np.sqrt(nz["yc"] / 3) if p["heavy"] else nz["ye"]
    y = s + p["sy"] * noise
    return np.sinh(y) if p["heavy"] else y


def history_kind(slot: int) -> str:
    """Alternate classification / regression by the slot's index within its family cycle."""
    return "classification" if (slot // len(HISTORY_FAMILIES)) % 2 == 0 else "regression"


def history_family(slot: int) -> str:
    return HISTORY_FAMILIES[slot % len(HISTORY_FAMILIES)]


def history_simulate(p: Mapping[str, Any], nz: Mapping[str, np.ndarray]) -> dict[str, Any]:
    z = _history_latent(p, nz)
    ev = _history_events(p, nz, z)
    s = _history_score(p, nz, z, ev)
    return dict(z=z, score=s, label=_history_labels(p, nz, s), ev=ev)
