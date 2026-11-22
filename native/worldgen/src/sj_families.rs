use crate::scm::GeneratedDb;
use serde::Serialize;

pub const CONDREL_RECEIPT_FILE: &str = "condrel_receipt.json";
pub const XREP_MANIFEST_FILE: &str = "xrep_rendering_manifest.json";

pub fn additional_task_drop_columns(
    db: &GeneratedDb,
    target_table: &str,
    target: &str,
) -> Vec<(String, String)> {
    if is_condrel_target(target) {
        let selected = if target.starts_with("condrel_permuted_") {
            "condrel_permuted_u"
        } else {
            "condrel_u"
        };
        return ["condrel_u", "condrel_permuted_u"]
            .into_iter()
            .filter(|column| *column != selected)
            .filter(|column| {
                db.table(target_table)
                    .is_some_and(|table| table.column(column).is_some())
            })
            .map(|column| (target_table.to_owned(), column.to_owned()))
            .collect();
    }
    let Some(rendering) = xrep_rendering(target) else {
        return Vec::new();
    };
    let selected_table = match rendering {
        "own" => target_table,
        "satellite" => "sj_sat_2_0",
        "bag" => "sj_junction_0_2",
        _ => return Vec::new(),
    };
    db.tables
        .iter()
        .filter(|table| table.name != selected_table)
        .flat_map(|table| {
            ["xrep_evidence_0", "xrep_evidence_1", "xrep_donor"]
                .into_iter()
                .filter(move |column| table.column(column).is_some())
                .map(move |column| (table.name.clone(), column.to_owned()))
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct CondrelPair {
    pub query_row: usize,
    pub query_key: i64,
    pub gate: bool,
    pub predicate: f64,
    pub predicate_permuted: f64,
    pub label: f64,
    pub counterfactual_label: f64,
    pub operator_value: f64,
    pub counterfactual_operator_value: f64,
    pub neighbourhood_rows: Vec<i64>,
    pub counterfactual_neighbourhood_rows: Vec<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CondrelReceipt {
    pub schema_version: u32,
    pub family: String,
    pub target_table: String,
    pub target_columns: Vec<String>,
    pub predicate_column: String,
    pub permuted_predicate_column: String,
    pub threshold: f64,
    pub orientation: String,
    pub mechanism: String,
    pub own_weight: f64,
    pub beta: f64,
    pub epsilon_sigma: f64,
    pub gate_true_fraction: f64,
    pub source_sha256: String,
    pub schema_sha256: String,
    pub pairs: Vec<CondrelPair>,
}

#[derive(Debug, Clone, Serialize)]
pub struct XrepRendering {
    pub rendering: String,
    pub table: String,
    pub evidence_columns: Vec<String>,
    pub donor_column: String,
    pub target_column: String,
    pub target_value_sha256: String,
    pub latent_value_sha256: String,
    pub zero_donor_entities: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct XrepManifest {
    pub schema_version: u32,
    pub family: String,
    pub target_table: String,
    pub payload_relevance: String,
    pub admission_density: String,
    pub observation_noise: String,
    pub target_value_sha256: String,
    pub latent_value_sha256: String,
    pub source_sha256: String,
    pub schema_sha256: String,
    pub renderings: Vec<XrepRendering>,
}

fn is_condrel_target(target: &str) -> bool {
    matches!(
        target,
        "condrel_r" | "condrel_k2" | "condrel_permuted_r" | "condrel_permuted_k2"
    )
}

fn xrep_rendering(target: &str) -> Option<&str> {
    target
        .strip_prefix("xrep_y_")
        .and_then(|value| value.rsplit_once('_'))
        .and_then(|(value, readout)| (readout == "r" || readout == "k2").then_some(value))
        .and_then(|value| value.rsplit_once('_'))
        .map(|(_, rendering)| rendering)
}
