// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
pub mod archive_v1;
mod archive_v2;
mod nodes;
mod outputs;
pub mod pre;
mod stats;
mod tables;
mod text_encoder;
mod text_output;
mod time_cutoffs;
mod vocabulary;

#[cfg(feature = "python")]
mod decoder;

/// `convert(db, data_root, out_root, ...)`: `rustler pre` from Python; the options mirror the command line.
#[cfg(feature = "python")]
#[pyo3::pyfunction]
#[pyo3(signature = (db, data_root, out_root, *, archive_format="v2", text_encoder="all-MiniLM-L12-v2",
    text_encoder_batch=8192, text_encoder_max_length=256, text_encoder_device="auto", allow_cpu=false,
    text_embedding_cache_only=false, time_cutoffs_file=None, fp32_datetimes=false))]
#[allow(clippy::too_many_arguments)]
fn convert(
    py: pyo3::Python<'_>,
    db: String,
    data_root: String,
    out_root: String,
    archive_format: &str,
    text_encoder: &str,
    text_encoder_batch: usize,
    text_encoder_max_length: usize,
    text_encoder_device: &str,
    allow_cpu: bool,
    text_embedding_cache_only: bool,
    time_cutoffs_file: Option<String>,
    fp32_datetimes: bool,
) {
    let cli = pre::Cli {
        archive_format: archive_format.to_string(),
        db_name: db,
        text_encoder: text_encoder.to_string(),
        text_encoder_batch,
        text_encoder_max_length,
        text_encoder_device: text_encoder_device.to_string(),
        allow_cpu,
        text_embedding_cache_only,
        time_cutoffs_file,
        fp32_datetimes,
    };
    // The text encoder re-acquires the interpreter from worker threads.
    py.detach(|| pre::run(&cli, &data_root, &out_root));
}

#[cfg(feature = "python")]
#[pyo3::pymodule]
fn rustler_archive(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    m.add_class::<decoder::ArchiveV1>()?;
    m.add_function(pyo3::wrap_pyfunction!(convert, m)?)
}
