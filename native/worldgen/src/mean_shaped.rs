//! The mean-shaped shortcut: measuring it, and refusing to publish columns that have it.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct MeanShapedDrop {
    pub table: String,
    pub family: String,
    pub arm: String,
    pub reg_column: String,
    pub clf_column: Option<String>,
    /// In-sample R^2 of the per-child-column-mean predictor. The criterion.
    pub mean_only_r2: f64,
    /// The same with the child count added. Reported, never the criterion:
    /// it is the fair baseline against the existing library but a column can
    /// legitimately be recoverable from a count.
    pub mean_and_count_r2: f64,
    pub ceiling: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MeanShapedFilterReport {
    /// Yield per `(family, arm)` for the regression views.
    pub per_family: Vec<FamilyYield>,
    pub drops: Vec<MeanShapedDrop>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FamilyYield {
    pub family: String,
    pub arm: String,
    pub ceiling: f64,
    pub candidate_reg_columns: usize,
    pub dropped_reg_columns: usize,
    pub kept_reg_columns: usize,
    /// Largest mean-only R^2 among the columns that survived, or `None` when nothing survived.
    pub kept_max_mean_only_r2: Option<f64>,
}

impl MeanShapedFilterReport {}
