"""Oracle, leakage and admission reports of panel-history worlds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from openrfm.synth.io import _require
from openrfm.synth.stats import _r2
from openrfm.synth.worlds.panel import (
    EPOCH_S, HISTORY_ALL_PREDS, HISTORY_BASE_DAY, HISTORY_COLD_SHARE_MIN, HISTORY_EVENT_WINDOW_DAYS, HISTORY_GAIN_MIN,
    HISTORY_HORIZON_DAYS, HISTORY_NOHIST_GAIN_MAX, HISTORY_OWN_EVENT_ROWS, HISTORY_PRED_NAMES, HISTORY_STEPS,
    HISTORY_STEP_DAYS, HISTORY_SUPPLY_MEDIAN_MIN, HISTORY_WINDOW_GAP_MIN, HISTORY_WINDOW_PREDS,
)


def _auc(score: Sequence[float], y: Sequence[int]) -> float:
    s, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    _require(len(pos) and len(neg), "oracle AUROC needs both classes")
    return float(
        (np.sum(pos[:, None] > neg) + 0.5 * np.sum(pos[:, None] == neg)) / (len(pos) * len(neg))
    )


def _ranks(x: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def _auc_fast(score: np.ndarray, y: np.ndarray) -> float:
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((_ranks(score)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc_se(auc: float, n1: int, n0: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    return float(
        np.sqrt(
            (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0)
        )
    )


def _history_read(root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    day = lambda t: (
        t["date"].cast("int64").to_numpy() // (86400 * 10**9) - EPOCH_S // 86400 - HISTORY_BASE_DAY
    )
    t = {
        name: pq.read_table(root / "db" / f"{name}.parquet")
        for name in ("a5_task_0", "case_own_events", "case_own_labels", "case_peer_links")
    }
    kind = json.loads((root / "task_metadata.json").read_text())["tasks"][0]["task_type"]
    n = t["a5_task_0"].num_rows
    ctx = dict(
        n=n,
        kind="regression" if kind == "regression" else "classification",
        y=t["a5_task_0"]["a5_y_0"].to_numpy().astype(np.float64),
        cut=day(t["a5_task_0"]),
    )
    for name, key, cols in (
        ("case_own_events", "ev", ("age_days", "feature_0", "feature_1")),
        ("case_own_labels", "lab", ("label_value",)),
        ("case_peer_links", "peer", ("label_value",)),
    ):
        ctx[key] = dict(
            case=t[name]["case_id"].to_numpy(),
            day=day(t[name]),
            **{c: t[name][c].to_numpy() for c in cols},
        )
        if key == "ev":
            prior = ctx[key]["day"] < ctx["cut"][ctx[key]["case"]]
            ctx[key] = {k: v[prior] for k, v in ctx[key].items()}
        ctx[key]["ptr"] = np.searchsorted(ctx[key]["case"], np.arange(n + 1))
    return ctx


def _history_matrix(ctx: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Per-row predictor matrix (NaN = no history) and the peer-group mean of the peers' latest labels."""
    tf = np.arcsinh if ctx["kind"] == "regression" else (lambda x: x)
    ev, lab, peer = ctx["ev"], ctx["lab"], ctx["peer"]
    X, P = np.full((ctx["n"], len(HISTORY_PRED_NAMES)), np.nan), np.full(ctx["n"], np.nan)
    mean = lambda x: float(x.mean()) if len(x) else np.nan
    for i in range(ctx["n"]):
        L = tf(lab["label_value"][lab["ptr"][i] : lab["ptr"][i + 1]])
        R = tf(ev["feature_0"][ev["ptr"][i] : ev["ptr"][i + 1]])
        age = ev["age_days"][ev["ptr"][i] : ev["ptr"][i + 1]]
        X[i] = [
            mean(L),
            L[0] if len(L) else np.nan,
            mean(L[:4]),
            mean(L[:16]),
            float((age <= HISTORY_EVENT_WINDOW_DAYS).sum()),
            -age[0] if len(age) else np.nan,
            mean(R),
            mean(R[:4]),
            mean(R[:16]),
            mean(R[:64]),
        ]
        P[i] = mean(tf(peer["label_value"][peer["ptr"][i] : peer["ptr"][i + 1]]))
    return X, P


def _history_heldout_score(
    kind: str, x: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray
) -> float:
    """Held-out AUROC (sign chosen on train rows) or R2 of a train-fitted line for one predictor."""
    x = np.where(np.isnan(x), np.nanmean(x[tr]) if np.isfinite(x[tr]).any() else 0.0, x)
    if kind == "classification":
        sign = 1.0 if _auc_fast(x[tr], y[tr]) >= 0.5 else -1.0
        return _auc_fast(sign * x[te], y[te])
    slope, icpt = np.polyfit(x[tr], y[tr], 1)
    return _r2(slope * x[te] + icpt, y[te])


def oracle_report(root: str | Path) -> dict[str, Any]:
    """Admission statistics of a panel-history world computed from its tables only, never from latents."""
    root = Path(root)
    ctx = _history_read(root)
    kind = ctx["kind"]
    tf = np.arcsinh if kind == "regression" else (lambda x: x)
    X, peer = _history_matrix(ctx)
    y = tf(ctx["y"]) if kind == "regression" else ctx["y"]
    step = ctx["cut"] // HISTORY_STEP_DAYS
    tr, te = np.flatnonzero(step < HISTORY_STEPS // 2), np.flatnonzero(step >= HISTORY_STEPS // 2)
    scores = {
        name: _history_heldout_score(kind, X[:, j], y, tr, te)
        for j, name in enumerate(HISTORY_PRED_NAMES)
    }
    base = 0.5 if kind == "classification" else _r2(np.full(len(te), y[tr].mean()), y[te])
    best = max(scores, key=scores.get)
    best_win = max(HISTORY_WINDOW_PREDS, key=scores.get)
    best_all = max(HISTORY_ALL_PREDS, key=scores.get)
    gap64 = max(scores[best_win], scores["raw_last64"]) - scores[best_all]
    peer_cold = None
    matured = np.diff(ctx["lab"]["ptr"])
    cold = np.flatnonzero(
        (matured == 0) & np.isfinite(peer)
    )  # every cold row with peers (train and held-out)
    if len(cold) >= 20 and (kind == "regression" or 0 < ctx["y"][cold].sum() < len(cold)):
        peer_cold = (
            _auc_fast(peer[cold], y[cold])
            if kind == "classification"
            else float(np.corrcoef(peer[cold], y[cold])[0, 1] ** 2)
        )
    n1 = int((ctx["y"][te] == 1).sum())
    n0 = len(te) - n1
    report = dict(
        kind=kind,
        metric="auroc" if kind == "classification" else "r2",
        n_rows=ctx["n"],
        n_heldout=len(te),
        global_value=base,
        scores=scores,
        best_own=best,
        best_own_value=scores[best],
        gain=scores[best] - base,
        best_window=best_win,
        best_all=best_all,
        window_gap=scores[best_win] - scores[best_all],
        window_gap_w64=gap64,
        share_cold=float((matured == 0).mean()),
        share_ge4=float((matured >= 4).mean()),
        share_ge16=float((matured >= 16).mean()),
        median_matured=float(np.median(matured)),
        max_matured=int(matured.max()),
        peer_cold=peer_cold,
        n_cold_rows=len(cold),
        gain_se=_auc_se(scores[best], n1, n0) if kind == "classification" else 0.0,
    )
    if (root / "panel_private.npz").exists():
        report["leakage"] = history_leakage_check(root, ctx, best_own=best)
    return report


def history_leakage_check(
    root: Path, ctx: Mapping[str, Any], *, best_own: str, rows: int = 300
) -> dict[str, Any]:
    """Check that shuffling post-cutoff rows leaves every history unchanged and future-aware predictors beat history."""
    kind = ctx["kind"]
    tf = np.arcsinh if kind == "regression" else (lambda x: x)
    ent = np.load(root / "panel_private.npz")["case_entity"]
    ev = ctx["ev"]
    e_of = ent[ev["case"]]
    key = np.unique(
        np.stack([e_of, ev["day"], ev["feature_0"], ev["feature_1"]], 1), axis=0
    )  # entity stream, deduplicated
    ptr = np.searchsorted(key[:, 0], np.arange(ent.max() + 2))
    rng = np.random.default_rng(0)
    ok, checked = True, 0
    for i in np.flatnonzero(ctx["cut"] // HISTORY_STEP_DAYS >= HISTORY_STEPS // 2)[:rows]:
        d, v = (
            key[ptr[ent[i]] : ptr[ent[i] + 1], 1],
            key[ptr[ent[i]] : ptr[ent[i] + 1], 2].copy(),
        )
        future = d >= ctx["cut"][i]
        v[future] = rng.permutation(v[future])
        sl = slice(ev["ptr"][i], ev["ptr"][i + 1])
        td, tv = ev["day"][sl], ev["feature_0"][sl]
        floor = (
            td.min() + (len(td) == HISTORY_OWN_EVENT_ROWS) if len(td) else 0
        )  # a capped history may cut a day in half
        want = np.flatnonzero(~future & (d >= floor))
        ok &= bool(
            len(td) == len(want) + int((td < floor).sum())
            and np.array_equal(np.sort(v[want]), np.sort(tv[td >= floor]))
        )
        checked += 1
    y = tf(ctx["y"]) if kind == "regression" else ctx["y"]
    hist, fut, keep = [], [], []
    for i in range(ctx["n"]):
        d, v = key[ptr[ent[i]] : ptr[ent[i] + 1], 1], tf(key[ptr[ent[i]] : ptr[ent[i] + 1], 2])
        a, b = np.searchsorted(d, [ctx["cut"][i], ctx["cut"][i] + HISTORY_HORIZON_DAYS])
        if b > a and a > 0:
            keep.append(i)
            fut.append([v[a:b].mean()] + [v[max(0, b - k) : b].mean() for k in (4, 16, 64, b)])
            hist.append([v[max(0, a - k) : a].mean() for k in (4, 16, 64, a)])
    out = dict(history_consistent=bool(ok), rows_checked=checked, future_rows=len(keep))
    if len(keep) >= 60:
        yk = y[keep]
        fit = lambda x: (
            _auc_fast(x, yk)
            if kind == "classification"
            else _r2(np.polyval(np.polyfit(x, yk, 1), x), yk)
        )
        fit = (lambda f: lambda x: max(f(x), f(-x)) if kind == "classification" else f(x))(fit)
        fut, hist = np.asarray(fut), np.asarray(hist)
        out.update(
            history_best=max(fit(hist[:, j]) for j in range(hist.shape[1])),
            future_best=max(fit(fut[:, j]) for j in range(fut.shape[1])),
        )
        out["future_gain"] = out["future_best"] - out["history_best"]
    return out


def history_admission(report: Mapping[str, Any], family: str) -> list[str]:
    """Reasons a world fails its family's thresholds (empty list = admitted)."""
    why, gain, gap = (
        [],
        report["gain"],
        report["window_gap_w64" if family == "window" else "window_gap"],
    )
    if family == "nohist":
        if gain > HISTORY_NOHIST_GAIN_MAX + 2 * report["gain_se"]:
            why.append(f"nohist: own history beats global by {gain:.3f}")
    elif gain < HISTORY_GAIN_MIN:
        why.append(f"{family}: own-history gain {gain:.3f} < {HISTORY_GAIN_MIN}")
    if family in ("regime", "window", "drift") and gap < HISTORY_WINDOW_GAP_MIN:
        why.append(f"{family}: window gap {gap:.3f} < {HISTORY_WINDOW_GAP_MIN}")
    if family == "meanrev" and gap > 0:
        why.append(f"meanrev: window gap {gap:.3f} > 0")
    if (
        report["median_matured"] < HISTORY_SUPPLY_MEDIAN_MIN
        or report["share_cold"] < HISTORY_COLD_SHARE_MIN
        or report["max_matured"] < 64
    ):
        why.append(
            f"label supply median {report['median_matured']} cold {report['share_cold']:.3f} "
            f"max {report['max_matured']}"
        )
    leak = report.get("leakage", {})
    if not leak.get("history_consistent", False):
        why.append("leakage: row history differs from the entity stream before the cutoff")
    if family != "nohist" and leak.get("future_gain", 0.0) <= (
        -0.005 if family == "window" else 0.0
    ):
        why.append(
            f"leakage: future predictor does not beat history ({leak.get('future_gain')})"
        )
    return why
