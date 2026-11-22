use crate::rng::{sample_power_law_int, GenRng, RngExt};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SemType {
    Numerical,
    Boolean,
    Class,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TableKind {
    Entity,
    Activity,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColumnSpec {
    pub name: String,
    pub sem_type: SemType,
    pub num_classes: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ForeignKeySpec {
    pub col_name: String,
    pub parent_table: String,
    pub parent_id: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TableSpec {
    pub id: usize,
    pub name: String,
    pub pkey_col: String,
    pub fkeys: Vec<ForeignKeySpec>,
    pub features: Vec<ColumnSpec>,
    pub kind: TableKind,
    pub num_rows: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SchemaSpec {
    pub tables: Vec<TableSpec>,
    pub topo_order: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SchemaOptions {
    #[serde(default)]
    pub satellite_tables: (usize, usize),
    #[serde(default)]
    pub junction_prob: f64,
    #[serde(default)]
    pub hub_fan_in: bool,
    #[serde(default = "default_max_junction_tables")]
    pub max_junction_tables: usize,
    #[serde(default = "default_max_satellites_per_entity")]
    pub max_satellites_per_entity: usize,
    /// Zipf exponent over eligible hub identities. See sj::hub_weights.
    #[serde(default = "default_hub_fanin_zipf")]
    pub hub_fanin_zipf: f64,
    /// Total S-J tables, including satellites, streams and label sidecars.
    pub num_tables_range: (usize, usize),
    pub num_cols_range: (usize, usize),
    pub rows_entity_range: (usize, usize),
    pub rows_activity_range: (usize, usize),
    /// Fewest foreign keys a table may be given, where the layout allows it.
    pub min_parents: usize,
    pub heavy_tail_fanout: bool,
    /// Hard cap for the typed long-tailed FK fan-out law.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fanout_tail: Option<usize>,
    /// Maximum number of FK edges on any root-to-table chain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_depth: Option<usize>,
}

pub const fn default_max_junction_tables() -> usize {
    4
}
pub const fn default_max_satellites_per_entity() -> usize {
    1
}
pub const fn default_hub_fanin_zipf() -> f64 {
    1.5
}

impl Default for SchemaOptions {
    fn default() -> Self {
        Self {
            max_junction_tables: default_max_junction_tables(),
            max_satellites_per_entity: default_max_satellites_per_entity(),
            hub_fanin_zipf: default_hub_fanin_zipf(),
            satellite_tables: (0, 0),
            junction_prob: 0.0,
            hub_fan_in: false,
            num_tables_range: (3, 15),
            num_cols_range: (5, 20),
            rows_entity_range: (500, 1000),
            rows_activity_range: (2000, 5000),
            min_parents: 1,
            heavy_tail_fanout: false,
            fanout_tail: None,
            max_depth: None,
            min_depth: None,
        }
    }
}

pub fn build_random_schema(
    rng: &mut GenRng,
    options: &SchemaOptions,
) -> anyhow::Result<SchemaSpec> {
    options.validate_scale()?;
    if options.sj_enabled() {
        return crate::sj::schema(rng, options);
    }
    let num_tables = rng.usize_inclusive(options.num_tables_range.0, options.num_tables_range.1);
    let layout = rng.choose_slice(&["BarabasiAlbert", "ReverseRandomTree", "WattsStrogatz"]);
    let min_parents = options.min_parents.max(1);
    let mut edges = match layout {
        "ReverseRandomTree" => random_reverse_tree_edges(rng, num_tables),
        "WattsStrogatz" => random_watts_like_edges(rng, num_tables),
        _ => random_ba_like_edges(rng, num_tables, min_parents),
    };
    // Top up to `min_parents` for every layout.
    if min_parents > 1 {
        for child in 1..num_tables {
            let mut current: Vec<usize> = edges
                .iter()
                .filter(|(_, dst)| *dst == child)
                .map(|(src, _)| *src)
                .collect();
            let target = min_parents.min(child);
            let mut attempts = 0usize;
            while current.len() < target && attempts < child * 8 {
                let candidate = rng.choose_index(child);
                if !current.contains(&candidate) {
                    current.push(candidate);
                    edges.push((candidate, child));
                }
                attempts += 1;
            }
        }
        edges.sort_unstable();
        edges.dedup();
    }
    if let Some(min_depth) = options.min_depth {
        anyhow::ensure!(
            num_tables > min_depth,
            "min-depth requires at least depth+1 tables"
        );
        edges.extend((1..=min_depth).map(|child| (child - 1, child)));
        edges.sort_unstable();
        edges.dedup();
    }
    if let Some(max_depth) = options.max_depth {
        edges = cap_edges_at_depth(&edges, num_tables, max_depth);
    }
    let schema = build_tables_from_edges(rng, num_tables, &edges, options);
    Ok(schema)
}

fn cap_edges_at_depth(
    edges: &[(usize, usize)],
    num_tables: usize,
    max_depth: usize,
) -> Vec<(usize, usize)> {
    let mut depth = vec![0usize; num_tables];
    let mut kept = Vec::with_capacity(edges.len());
    for child in 1..num_tables {
        for &(parent, _) in edges.iter().filter(|(_, candidate)| *candidate == child) {
            if depth[parent] < max_depth {
                kept.push((parent, child));
                depth[child] = depth[child].max(depth[parent] + 1);
            }
        }
    }
    kept
}

fn random_ba_like_edges(
    rng: &mut GenRng,
    num_tables: usize,
    min_parents: usize,
) -> Vec<(usize, usize)> {
    let mut edges = BTreeSet::new();
    for child in 1..num_tables {
        let max_parents = child.min(4);
        let lo = min_parents.clamp(1, max_parents);
        let n_parents = rng.usize_inclusive(lo, max_parents);
        let weights: Vec<f64> = (0..child)
            .map(|parent| {
                let degree = edges
                    .iter()
                    .filter(|(src, dst)| *src == parent || *dst == parent)
                    .count();
                (degree + 1) as f64
            })
            .collect();
        let mut parents = BTreeSet::new();
        let mut attempts = 0usize;
        while parents.len() < n_parents && attempts < child * 4 {
            parents.insert(rng.choose_weighted(&weights));
            attempts += 1;
        }
        if parents.is_empty() {
            parents.insert(rng.choose_index(child));
        }
        for parent in parents {
            edges.insert((parent, child));
        }
    }
    edges.into_iter().collect()
}

fn random_reverse_tree_edges(rng: &mut GenRng, num_tables: usize) -> Vec<(usize, usize)> {
    let mut edges = BTreeSet::new();
    for child in 1..num_tables {
        let parent = rng.choose_index(child);
        edges.insert((parent, child));
    }
    edges.into_iter().collect()
}

fn random_watts_like_edges(rng: &mut GenRng, num_tables: usize) -> Vec<(usize, usize)> {
    let mut edges = BTreeSet::new();
    for child in 1..num_tables {
        edges.insert((child - 1, child));
        if child >= 2 && rng.bernoulli(0.35) {
            edges.insert((child - 2, child));
        }
        if child >= 3 && rng.bernoulli(0.15) {
            edges.insert((rng.choose_index(child), child));
        }
    }
    edges.into_iter().collect()
}

fn build_tables_from_edges(
    rng: &mut GenRng,
    num_tables: usize,
    edges: &[(usize, usize)],
    options: &SchemaOptions,
) -> SchemaSpec {
    let mut parents: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut children: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(parent, child) in edges {
        parents.entry(child).or_default().push(parent);
        children.entry(parent).or_default().push(child);
    }

    let mut tables = Vec::with_capacity(num_tables);
    for table_id in 0..num_tables {
        let name = format!("table_{table_id}");
        let mut parent_ids = parents.remove(&table_id).unwrap_or_default();
        parent_ids.sort_unstable();
        parent_ids.dedup();

        let fkeys = parent_ids
            .iter()
            .enumerate()
            .map(|(idx, parent_id)| ForeignKeySpec {
                col_name: format!("foreign_row_{idx}"),
                parent_table: format!("table_{parent_id}"),
                parent_id: *parent_id,
            })
            .collect::<Vec<_>>();

        let num_cols = sample_power_law_int::<usize>(
            rng,
            options.num_cols_range.0,
            options.num_cols_range.1,
            2.5,
        );
        let mut features = Vec::with_capacity(num_cols);
        for col_idx in 0..num_cols {
            let draw = rng.f64_range(0.0, 1.0);
            let sem_type = if draw < 0.08 {
                SemType::Class
            } else if draw < 0.20 {
                SemType::Boolean
            } else {
                SemType::Numerical
            };
            features.push(ColumnSpec {
                name: format!("feature_{col_idx}"),
                sem_type,
                num_classes: matches!(sem_type, SemType::Class).then(|| rng.usize_inclusive(2, 10)),
            });
        }

        let kind = if children.get(&table_id).map_or(true, Vec::is_empty) {
            TableKind::Activity
        } else {
            TableKind::Entity
        };
        let base_num_rows = match kind {
            TableKind::Entity => {
                rng.usize_inclusive(options.rows_entity_range.0, options.rows_entity_range.1)
            }
            TableKind::Activity => {
                rng.usize_inclusive(options.rows_activity_range.0, options.rows_activity_range.1)
            }
        };
        let num_rows = base_num_rows;

        tables.push(TableSpec {
            id: table_id,
            name,
            pkey_col: "row_idx".to_string(),
            fkeys,
            features,
            kind,
            num_rows,
        });
    }

    SchemaSpec {
        tables,
        topo_order: (0..num_tables).collect(),
    }
}

impl SchemaOptions {
    pub(crate) fn validate_scale(&self) -> anyhow::Result<()> {
        if let Some(depth) = self.min_depth {
            anyhow::ensure!(
                depth > 0 && self.max_depth.map_or(true, |max| depth <= max),
                "min-depth must be positive and no larger than depth"
            );
            anyhow::ensure!(
                !self.sj_enabled(),
                "min-depth is not supported by the S-J layout"
            );
        }
        if let Some(tail) = self.fanout_tail {
            anyhow::ensure!(tail > 0, "fanout_tail must be positive");
        }
        if let Some(depth) = self.max_depth {
            anyhow::ensure!(depth > 0, "depth must be positive");
        }
        Ok(())
    }
    pub(crate) fn sj_enabled(&self) -> bool {
        self.satellite_tables != (0, 0) || self.junction_prob != 0.0 || self.hub_fan_in
    }
    /// Explicit S-J recipe; default() leaves it off.
    pub(crate) fn sj_band() -> Self {
        Self {
            satellite_tables: (1, 1),
            junction_prob: 0.25,
            hub_fan_in: true,
            num_tables_range: (6, 24),
            ..Self::default()
        }
    }
}
