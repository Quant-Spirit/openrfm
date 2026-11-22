//! Synthetic relational database generator: schema, structural causal model, label families
//! and parquet writer.

mod hsbm;
mod info_prior;
mod labels;
mod labels_v2;
mod manifest;
mod mean_shaped;
mod parquet;
mod rng;
mod schema;
mod scm;
mod shortcut_gates;
mod sj;
mod sj_families;
mod taxonomy;
mod v6;

pub use manifest::{
    add_manifest_target, corpus_scale_coverage, expected_target_names, integrity_checks,
    print_corpus_scale_coverage, IntegrityCheck, MixtureManifest, MixtureRecord, SelectionContract,
};
pub use parquet::{
    base_value_digest, published_column_index, verify_base_value_digests, write_generated_db,
};
pub use scm::{generate_db, generate_db_skeleton, GenOptions, GeneratedDb};
pub use shortcut_gates::SHORTCUT_CEILING;
pub use taxonomy::manifest_for_record;

/// Forecasting worlds: per-slot world sampling and generation.
pub mod forecast {
    pub use crate::v6::{
        apply_world, assign_variant_slot, generate_world, sample_variant_world, sample_world,
        slot_seed, variant_slot_seed, V6Archetype, V6Stage, V6Variant, V6_MAX_ATTEMPTS,
    };
}

/// A planned record that must not be PUBLISHED, as distinct from a build that is broken.
#[derive(Debug)]
pub struct RecordRefused {
    pub reason: &'static str,
    pub detail: String,
    /// Structured evidence for the refusal, recorded alongside the reason.
    pub evidence: serde_json::Value,
}

impl RecordRefused {
    /// The record's target-blind schema sampling found no admissible skeleton.
    pub const SCHEMA_SAMPLING_EXHAUSTED: &'static str = "schema_sampling_exhausted";
    /// The record materialized, and a shortcut gate refused what it built.
    pub const SHORTCUT_GATE: &'static str = "shortcut_gate";
    pub const NO_ADMISSIBLE_TARGET_TUPLE: &'static str = "no_admissible_target_tuple";

    pub fn shortcut_gate(detail: impl Into<String>) -> anyhow::Error {
        anyhow::Error::new(Self {
            reason: Self::SHORTCUT_GATE,
            detail: detail.into(),
            evidence: serde_json::Value::Null,
        })
    }
}

impl std::fmt::Display for RecordRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.reason, self.detail)
    }
}

impl std::error::Error for RecordRefused {}
