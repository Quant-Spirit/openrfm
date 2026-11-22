use rand::distr::uniform::SampleUniform;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Beta, Distribution, StandardNormal};

pub type GenRng = ChaCha20Rng;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn hash_label(label: &str) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in label.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub fn substream(seed: u64, label: &str) -> GenRng {
    let mixed = splitmix64(seed ^ hash_label(label));
    ChaCha20Rng::seed_from_u64(mixed)
}

pub fn child_substream(seed: u64, label: &str, index: usize) -> GenRng {
    let tagged = format!("{label}:{index}");
    substream(seed, &tagged)
}

pub trait RngExt {
    fn usize_inclusive(&mut self, low: usize, high: usize) -> usize;
    fn i64_inclusive(&mut self, low: i64, high: i64) -> i64;
    fn f64_range(&mut self, low: f64, high: f64) -> f64;
    fn normal(&mut self, mean: f64, std: f64) -> f64;
    fn standard_normal(&mut self) -> f64;
    fn beta(&mut self, alpha: f64, beta: f64) -> f64;
    fn bernoulli(&mut self, p: f64) -> bool;
    fn choose_index(&mut self, len: usize) -> usize;
    fn choose_weighted(&mut self, weights: &[f64]) -> usize;
    fn choose_slice<T: Copy>(&mut self, values: &[T]) -> T;
    fn shuffle<T>(&mut self, values: &mut [T]);
}

impl RngExt for GenRng {
    fn usize_inclusive(&mut self, low: usize, high: usize) -> usize {
        if high <= low {
            return low;
        }
        self.random_range(low..=high)
    }

    fn i64_inclusive(&mut self, low: i64, high: i64) -> i64 {
        if high <= low {
            return low;
        }
        self.random_range(low..=high)
    }

    fn f64_range(&mut self, low: f64, high: f64) -> f64 {
        if high <= low {
            return low;
        }
        self.random_range(low..high)
    }

    fn normal(&mut self, mean: f64, std: f64) -> f64 {
        mean + std.max(1e-12) * self.standard_normal()
    }

    fn standard_normal(&mut self) -> f64 {
        StandardNormal.sample(self)
    }

    fn beta(&mut self, alpha: f64, beta: f64) -> f64 {
        match Beta::new(alpha.max(1e-6), beta.max(1e-6)) {
            Ok(dist) => dist.sample(self),
            Err(_) => self.random::<f64>(),
        }
    }

    fn bernoulli(&mut self, p: f64) -> bool {
        self.random::<f64>() < p.clamp(0.0, 1.0)
    }

    fn choose_index(&mut self, len: usize) -> usize {
        assert!(len > 0, "cannot choose from an empty collection");
        self.random_range(0..len)
    }

    fn choose_weighted(&mut self, weights: &[f64]) -> usize {
        assert!(
            !weights.is_empty(),
            "cannot choose from an empty weight vector"
        );
        let total: f64 = weights.iter().copied().filter(|w| *w > 0.0).sum();
        if total <= 0.0 || !total.is_finite() {
            return self.choose_index(weights.len());
        }
        let mut draw = self.random::<f64>() * total;
        for (idx, weight) in weights.iter().enumerate() {
            if *weight <= 0.0 {
                continue;
            }
            if draw <= *weight {
                return idx;
            }
            draw -= *weight;
        }
        weights.len() - 1
    }

    fn choose_slice<T: Copy>(&mut self, values: &[T]) -> T {
        values[self.choose_index(values.len())]
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        if values.len() < 2 {
            return;
        }
        for i in (1..values.len()).rev() {
            let j = self.random_range(0..=i);
            values.swap(i, j);
        }
    }
}

pub fn sample_power_law_int<T>(rng: &mut GenRng, low: T, high: T, exponent: f64) -> T
where
    T: Copy + Into<usize> + TryFrom<usize> + SampleUniform,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let low_usize = low.into();
    let high_usize = high.into();
    if high_usize <= low_usize {
        return low;
    }
    let values: Vec<usize> = (low_usize..=high_usize).collect();
    let weights: Vec<f64> = values
        .iter()
        .map(|v| 1.0 / (*v as f64).powf(exponent))
        .collect();
    T::try_from(values[rng.choose_weighted(&weights)]).expect("sampled value fits")
}

pub fn zscore(values: &[f64]) -> Vec<f64> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return vec![0.0; values.len()];
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / finite.len() as f64;
    let std = var.sqrt().max(1e-8);
    values
        .iter()
        .map(|v| {
            let filled = if v.is_finite() { *v } else { mean };
            (filled - mean) / std
        })
        .collect()
}
