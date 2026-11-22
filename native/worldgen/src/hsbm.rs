use crate::rng::{GenRng, RngExt};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct BipartiteHsbm {
    pub parent_for_child: Vec<usize>,
    pub hierarchy_a: Vec<Vec<usize>>,
    pub hierarchy_b: Vec<Vec<usize>>,
    pub hierarchy_shape_a: Vec<usize>,
    pub hierarchy_shape_b: Vec<usize>,
}

pub fn assign_cluster_at_levels(num_nodes: usize, hierarchy: &[usize]) -> Vec<Vec<usize>> {
    if hierarchy.is_empty() {
        return vec![Vec::new(); num_nodes];
    }
    let num_base_clusters = hierarchy.iter().copied().product::<usize>().max(1);
    let nodes_per_cluster = num_nodes.div_ceil(num_base_clusters);
    let mut base_cluster_offsets = Vec::with_capacity(num_base_clusters);
    for c_idx in 0..num_base_clusters.saturating_sub(1) {
        base_cluster_offsets.push(nodes_per_cluster * (c_idx + 1));
    }
    base_cluster_offsets.push(num_nodes);

    let mut out = vec![vec![0usize; hierarchy.len()]; num_nodes];
    let mut start = 0usize;
    for c_idx in 0..num_base_clusters {
        let end = base_cluster_offsets[c_idx].min(num_nodes);
        for level_idx in 0..hierarchy.len() {
            let fac = if level_idx + 1 < hierarchy.len() {
                hierarchy[level_idx + 1..].iter().product::<usize>().max(1)
            } else {
                1
            };
            let value = (c_idx / fac) % hierarchy[level_idx].max(1);
            for row in out.iter_mut().take(end).skip(start) {
                row[level_idx] = value;
            }
        }
        start = end;
        if start >= num_nodes {
            break;
        }
    }
    out
}

pub fn get_probs_at_levels(
    rng: &mut GenRng,
    hierarchy_a: &[usize],
    hierarchy_b: &[usize],
) -> Vec<Vec<Vec<f64>>> {
    assert_eq!(
        hierarchy_a.len(),
        hierarchy_b.len(),
        "only similar hierarchy levels are supported"
    );
    let mut levels = Vec::with_capacity(hierarchy_a.len());
    for (&a, &b) in hierarchy_a.iter().zip(hierarchy_b.iter()) {
        let rows = a.max(1);
        let cols = b.max(1);
        let mut probs = vec![vec![0.0; cols]; rows];
        for row in probs.iter_mut() {
            for value in row.iter_mut() {
                *value = rng.f64_range(0.001, 0.002);
            }
        }
        for i in 0..rows.max(cols) {
            probs[i % rows][i % cols] = 0.9;
        }
        levels.push(probs);
    }
    levels
}

pub fn fast_get_bipartite_hsbm(
    rng: &mut GenRng,
    size_a: usize,
    size_b: usize,
    hierarchy_a: &[usize],
    hierarchy_b: &[usize],
) -> BipartiteHsbm {
    assert_eq!(
        hierarchy_a.len(),
        hierarchy_b.len(),
        "only similar hierarchy levels are supported"
    );
    assert!(size_a > 0, "HSBM parent side must be non-empty");

    let cluster_a = assign_cluster_at_levels(size_a, hierarchy_a);
    let cluster_b = assign_cluster_at_levels(size_b, hierarchy_b);
    let probs_at_levels = get_probs_at_levels(rng, hierarchy_a, hierarchy_b);

    let mut parent_for_child = Vec::with_capacity(size_b);
    for child_idx in 0..size_b {
        let mut probs = vec![1.0f64; size_a];
        for (level_idx, probs_level) in probs_at_levels.iter().enumerate() {
            let cluster_child = cluster_b[child_idx][level_idx];
            for parent_idx in 0..size_a {
                let cluster_parent = cluster_a[parent_idx][level_idx];
                probs[parent_idx] *= probs_level[cluster_parent][cluster_child];
            }
        }
        let parent_idx = rng.choose_weighted(&probs);
        parent_for_child.push(parent_idx);
    }

    BipartiteHsbm {
        parent_for_child,
        hierarchy_a: cluster_a,
        hierarchy_b: cluster_b,
        hierarchy_shape_a: hierarchy_a.to_vec(),
        hierarchy_shape_b: hierarchy_b.to_vec(),
    }
}
