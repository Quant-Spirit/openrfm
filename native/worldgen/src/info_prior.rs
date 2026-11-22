use serde_json::{json, Value};

use crate::rng::{GenRng, RngExt};

pub const F_RANKS: [usize; 3] = [1, 4, 16];
/// Extra informative latent columns beyond the rank (the map lives in a d-dim subspace of them).
pub const INFORMATIVE_EXTRA: usize = 4;

pub fn f_rank_for_slot(slot: usize) -> usize {
    F_RANKS[slot % F_RANKS.len()]
}

/// Monotone per-column warp of a latent N(0, 1) value (the model must invert it; the oracle does not).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Warp {
    Identity,
    Exp,
    Cube,
    SignedLog,
    Logistic,
}

impl Warp {
    pub const ALL: [Warp; 5] = [
        Warp::Identity,
        Warp::Exp,
        Warp::Cube,
        Warp::SignedLog,
        Warp::Logistic,
    ];
    pub(crate) fn apply(self, z: f64) -> f64 {
        match self {
            Warp::Identity => z,
            Warp::Exp => (0.5 * z).exp(),
            Warp::Cube => z * z * z / 3.0,
            Warp::SignedLog => z.signum() * (1.0 + 3.0 * z.abs()).ln(),
            Warp::Logistic => 1.0 / (1.0 + (-1.5 * z).exp()),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Warp::Identity => "identity",
            Warp::Exp => "exp",
            Warp::Cube => "cube",
            Warp::SignedLog => "signed_log",
            Warp::Logistic => "logistic",
        }
    }
}

/// Random p x d frame with orthonormal columns (Gram-Schmidt on Gaussian columns).
pub fn orthonormal_frame(rng: &mut GenRng, p: usize, d: usize) -> Vec<Vec<f64>> {
    assert!(d >= 1 && d <= p, "rank {d} must be in 1..={p}");
    let mut frame: Vec<Vec<f64>> = Vec::with_capacity(d);
    while frame.len() < d {
        let mut v: Vec<f64> = (0..p).map(|_| rng.standard_normal()).collect();
        for q in &frame {
            let dot: f64 = v.iter().zip(q).map(|(a, b)| a * b).sum();
            for (x, y) in v.iter_mut().zip(q) {
                *x -= dot * y;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-6 {
            frame.push(v.into_iter().map(|x| x / norm).collect());
        }
    }
    frame
}

/// A rank-d own-row label map: s(z) = w2 . tanh(W1 u + b1) + a . u with u = Q^T z.
#[derive(Debug, Clone)]
pub struct RankMap {
    frame: Vec<Vec<f64>>,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    linear: Vec<f64>,
}

impl RankMap {
    pub(crate) fn sample(rng: &mut GenRng, rank: usize) -> Self {
        let p = rank + INFORMATIVE_EXTRA;
        let frame = orthonormal_frame(rng, p, rank);
        let hidden = (4 * rank).max(8);
        let scale = 1.5 / (rank as f64).sqrt();
        let w1 = (0..hidden)
            .map(|_| (0..rank).map(|_| scale * rng.standard_normal()).collect())
            .collect();
        let b1 = (0..hidden).map(|_| rng.normal(0.0, 0.5)).collect();
        let w2 = (0..hidden)
            .map(|_| rng.standard_normal() / (hidden as f64).sqrt())
            .collect();
        let linear = (0..rank)
            .map(|_| 0.3 * rng.standard_normal() / (rank as f64).sqrt())
            .collect();
        RankMap {
            frame,
            w1,
            b1,
            w2,
            linear,
        }
    }

    /// Latent coordinates u = Q^T z of one row's informative latents.
    pub(crate) fn project(&self, z: &[f64]) -> Vec<f64> {
        self.frame
            .iter()
            .map(|q| q.iter().zip(z).map(|(a, b)| a * b).sum())
            .collect()
    }

    pub(crate) fn score(&self, z: &[f64]) -> f64 {
        let u = self.project(z);
        let hidden: f64 = self
            .w1
            .iter()
            .zip(&self.b1)
            .zip(&self.w2)
            .map(|((row, b), w)| {
                w * (row.iter().zip(&u).map(|(a, x)| a * x).sum::<f64>() + b).tanh()
            })
            .sum();
        hidden + self.linear.iter().zip(&u).map(|(a, x)| a * x).sum::<f64>()
    }
}

/// Fresh own-row feature block for `n` entity rows: `p_inf` informative latents and `p_dis`
/// distractors, each published through a random monotone warp. Returns (latents per row, the
/// published columns in order: informative first, then distractors, and each column's warp).
pub fn feature_block(
    rng: &mut GenRng,
    n: usize,
    p_inf: usize,
    p_dis: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Warp>) {
    let columns = p_inf + p_dis;
    let warps: Vec<Warp> = (0..columns)
        .map(|_| Warp::ALL[rng.choose_index(Warp::ALL.len())])
        .collect();
    let latent: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..columns).map(|_| rng.standard_normal()).collect())
        .collect();
    let published: Vec<Vec<f64>> = (0..columns)
        .map(|c| latent.iter().map(|row| warps[c].apply(row[c])).collect())
        .collect();
    let informative = latent
        .into_iter()
        .map(|row| row[..p_inf].to_vec())
        .collect();
    (informative, published, warps)
}

fn variance(v: &[f64]) -> f64 {
    let n = v.len().max(1) as f64;
    let mean = v.iter().sum::<f64>() / n;
    v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

/// Exact channel record of one task: `components` are (channel, weight, standardised component
/// on the task rows); shares are Var(w_c z_c) / Var(sum_c w_c z_c) (they need not sum to 1 when
/// components correlate; the covariance remainder is recorded).
pub fn channel_record(components: &[(&str, f64, &[f64])], extra: Value) -> Value {
    let n = components
        .iter()
        .map(|(_, _, v)| v.len())
        .max()
        .unwrap_or(0);
    let mut total = vec![0.0; n];
    for (_, w, v) in components {
        for (t, x) in total.iter_mut().zip(v.iter()) {
            *t += w * x;
        }
    }
    let var_total = variance(&total).max(1e-12);
    let mut channels = serde_json::Map::new();
    let mut own = 0.0;
    for name in ["H", "F", "P", "S", "R"] {
        let (weight, share) = components.iter().filter(|(c, _, _)| *c == name).fold(
            (0.0, 0.0),
            |(w0, s0), (_, w, v)| {
                let scaled: Vec<f64> = v.iter().map(|x| w * x).collect();
                (w0 + w, s0 + variance(&scaled) / var_total)
            },
        );
        own += share;
        channels.insert(name.into(), json!({"weight": weight, "share": share}));
    }
    json!({"channels": channels, "covariance_share": 1.0 - own, "extra": extra})
}
