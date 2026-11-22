// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
use crate::archive_v1::{
    ArchivedAdj, ArchivedNode, ArchivedOffsets, ArchivedSemType, ArchivedTableType, Offsets,
    unpack_discrete_label,
};
use half::bf16;
use memmap2::{Mmap, MmapMut};
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rkyv::option::ArchivedOption;
use rkyv::rancor::Error;
use std::collections::HashMap;
use std::fs;
use std::mem::size_of;
use std::path::Path;

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn open_file(path: &Path, what: &str) -> PyResult<fs::File> {
    fs::File::open(path).map_err(|e| {
        PyFileNotFoundError::new_err(format!("Failed to open {what} at {}: {e}", path.display()))
    })
}

/// Memory-map a file; `RUSTLER_MMAP_POPULATE=1` pre-faults the mapping.
fn mmap_file(path: &Path, what: &str) -> PyResult<Mmap> {
    let file = open_file(path, what)?;
    let mut opts = memmap2::MmapOptions::new();
    if std::env::var("RUSTLER_MMAP_POPULATE").unwrap_or_default() == "1" {
        opts.populate();
    }
    unsafe { opts.map(&file) }.map_err(|e| {
        PyRuntimeError::new_err(format!(
            "Failed to memory-map {what} at {}: {e}",
            path.display()
        ))
    })
}

fn embedding_file_tag(model: &str) -> String {
    model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Text embeddings; with `RUSTLER_ALLOW_MISSING_TEXT_EMB` set and no embedding file, zeros sized by
/// the text vocabulary.
fn load_text_embeddings(root: &Path, embedding_model: &str, d_text: usize) -> PyResult<Mmap> {
    let text_path = root.join(format!("text_emb_{}.bin", embedding_file_tag(embedding_model)));
    if text_path.exists() {
        return mmap_file(&text_path, "text embeddings");
    }
    if !env_flag("RUSTLER_ALLOW_MISSING_TEXT_EMB") {
        return Err(PyFileNotFoundError::new_err(format!(
            "Missing required file (text embeddings): {}",
            text_path.display()
        )));
    }
    let text_map_path = root.join("text_map.json");
    let text_map: HashMap<String, i32> =
        serde_json::from_reader(open_file(&text_map_path, "text vocabulary")?).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to parse text vocabulary from {}: {e}",
                text_map_path.display()
            ))
        })?;
    let num_bytes = (text_map.len() * d_text * size_of::<u16>()).max(size_of::<u16>());
    MmapMut::map_anon(num_bytes)
        .and_then(MmapMut::make_read_only)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate zero text embeddings: {e}")))
}

fn archived_option_to_i32(opt: &ArchivedOption<rkyv::rend::i32_le>) -> Option<i32> {
    match opt {
        ArchivedOption::Some(v) => Some((*v).into()),
        ArchivedOption::None => None,
    }
}

struct Dataset {
    nodes: Mmap,
    text_emb: Mmap,
    p2f_adj: Mmap,
    offsets: Vec<i64>,
}

impl Dataset {
    fn load(root: &Path, embedding_model: &str, d_text: usize) -> PyResult<Self> {
        for (what, file) in [
            ("nodes", "nodes.rkyv"),
            ("offsets", "offsets.rkyv"),
            ("p2f adjacency", "p2f_adj.rkyv"),
        ] {
            if !root.join(file).exists() {
                return Err(PyFileNotFoundError::new_err(format!(
                    "Missing required file ({what}): {}",
                    root.join(file).display()
                )));
            }
        }
        let nodes = mmap_file(&root.join("nodes.rkyv"), "nodes")?;
        let text_emb = load_text_embeddings(root, embedding_model, d_text)?;
        let bytes = fs::read(root.join("offsets.rkyv")).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to read offsets from {}: {e}", root.display()))
        })?;
        let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("parse offsets archive: {e}")))?;
        let offsets = rkyv::deserialize::<Offsets, Error>(archived)
            .map_err(|e| PyRuntimeError::new_err(format!("deserialize offsets: {e}")))?
            .offsets;
        let p2f_adj = mmap_file(&root.join("p2f_adj.rkyv"), "p2f adjacency")?;
        Ok(Self {
            nodes,
            text_emb,
            p2f_adj,
            offsets,
        })
    }

    fn num_nodes(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// The bytes of node `idx`, copied into an aligned buffer for checked access.
    fn node_bytes(&self, idx: usize) -> Option<rkyv::util::AlignedVec> {
        let l = usize::try_from(*self.offsets.get(idx)?).ok()?;
        let r = usize::try_from(*self.offsets.get(idx + 1)?).ok()?;
        if l > r || r > self.nodes.len() {
            return None;
        }
        let mut bytes = rkyv::util::AlignedVec::new();
        bytes.extend_from_slice(&self.nodes[l..r]);
        Some(bytes)
    }

    fn text_emb(&self) -> PyResult<&[bf16]> {
        let (pref, slice, suf) = unsafe { self.text_emb.align_to::<bf16>() };
        if !pref.is_empty() || !suf.is_empty() {
            return Err(PyRuntimeError::new_err(
                "text embedding buffer is not aligned for bf16",
            ));
        }
        Ok(slice)
    }
}

fn table_type_to_i32(table_type: &ArchivedTableType) -> i32 {
    match table_type {
        ArchivedTableType::Db => 0,
        ArchivedTableType::Train => 1,
        ArchivedTableType::Val => 2,
        ArchivedTableType::Test => 3,
    }
}

#[derive(Default)]
struct Arrays {
    node_timestamps: Vec<i32>,
    node_table_name_idxs: Vec<i32>,
    f2p_rowptr: Vec<i64>,
    f2p_col_idx: Vec<i32>,
    p2f_rowptr: Vec<i64>,
    p2f_col_idx: Vec<i32>,
    p2f_table_name_idxs: Vec<i32>,
    p2f_table_types: Vec<i32>,
    p2f_timestamps: Vec<i32>,
    cell_nodes: Vec<i32>,
    cell_columns: Vec<i32>,
    cell_types: Vec<i32>,
    cell_numbers: Vec<f32>,
    cell_datetimes: Vec<f32>,
    cell_booleans: Vec<f32>,
    cell_text_ids: Vec<i32>,
    cell_codes: Vec<i32>,
    cell_num_classes: Vec<i32>,
    embedding_bits: Vec<u16>,
}

fn decode(dataset: &Dataset, d_text: usize) -> PyResult<Arrays> {
    let num_nodes = dataset.num_nodes();
    if num_nodes >= i32::MAX as usize {
        return Err(PyValueError::new_err(
            "v1 archive row ids exceed signed int32; refusing lossy export",
        ));
    }
    let mut a = Arrays {
        node_timestamps: vec![i32::MIN; num_nodes],
        node_table_name_idxs: vec![-1; num_nodes],
        f2p_rowptr: vec![0],
        p2f_rowptr: vec![0],
        ..Default::default()
    };

    for index in 0..num_nodes {
        let bytes = dataset
            .node_bytes(index)
            .ok_or_else(|| PyValueError::new_err(format!("invalid archived node {index}")))?;
        let node = rkyv::access::<ArchivedNode, Error>(&bytes)
            .map_err(|e| PyValueError::new_err(format!("invalid archived node {index}: {e}")))?;
        a.node_timestamps[index] = archived_option_to_i32(&node.timestamp).unwrap_or(i32::MIN);
        a.node_table_name_idxs[index] = node.table_name_idx.into();
        a.f2p_col_idx
            .extend(node.f2p_edges.iter().map(|edge| i32::from(edge.node_idx)));
        a.f2p_rowptr.push(a.f2p_col_idx.len() as i64);

        let width = node.col_name_idxs.len();
        if [
            node.sem_types.len(),
            node.number_values.len(),
            node.datetime_values.len(),
            node.boolean_values.len(),
            node.text_values.len(),
            node.class_value_idx.len(),
        ]
        .iter()
        .any(|&len| len != width)
        {
            return Err(PyValueError::new_err(format!(
                "unaligned cell arrays at node {index}"
            )));
        }
        for i in 0..width {
            let typ = match &node.sem_types[i] {
                ArchivedSemType::Number => 0,
                ArchivedSemType::Text => 1,
                ArchivedSemType::DateTime => 2,
                ArchivedSemType::Boolean => 3,
                ArchivedSemType::Discrete => 4,
            };
            let code: i32 = node.class_value_idx[i].into();
            let (num_classes, code) = if typ == 4 {
                unpack_discrete_label(code).map_err(PyValueError::new_err)?
            } else {
                (-1, code)
            };
            a.cell_nodes.push(index as i32);
            a.cell_columns.push(node.col_name_idxs[i].into());
            a.cell_types.push(typ);
            a.cell_numbers.push(node.number_values[i].into());
            a.cell_datetimes.push(node.datetime_values[i].into());
            a.cell_booleans.push(node.boolean_values[i].into());
            a.cell_text_ids.push(node.text_values[i].into());
            a.cell_codes.push(code);
            a.cell_num_classes.push(num_classes);
        }
    }

    let p2f_adj = rkyv::access::<ArchivedAdj, Error>(&dataset.p2f_adj[..])
        .map_err(|e| PyValueError::new_err(format!("invalid archived adjacency: {e}")))?;
    for index in 0..num_nodes {
        if let Some(edges) = p2f_adj.adj.get(index) {
            for edge in edges.iter() {
                a.p2f_col_idx.push(edge.node_idx.into());
                a.p2f_table_name_idxs.push(edge.table_name_idx.into());
                a.p2f_table_types.push(table_type_to_i32(&edge.table_type));
                a.p2f_timestamps
                    .push(archived_option_to_i32(&edge.timestamp).unwrap_or(i32::MIN));
            }
        }
        a.p2f_rowptr.push(a.p2f_col_idx.len() as i64);
    }

    let emb = dataset.text_emb()?;
    if emb.len() % d_text != 0 {
        return Err(PyValueError::new_err("embedding matrix has a partial row"));
    }
    a.embedding_bits = emb.iter().map(|value| value.to_bits()).collect();
    Ok(a)
}

#[pyclass]
pub struct ArchiveV1;

#[pymethods]
impl ArchiveV1 {
    /// Decode a format-v1 archive into named arrays, losslessly; no sampling or masking.
    #[staticmethod]
    fn load_arrays<'a>(
        py: Python<'a>,
        pre_path: &str,
        embedding_model: &str,
        d_text: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        if d_text == 0 || env_flag("RUSTLER_FORCE_ZERO_TEXT_EMB") {
            return Err(PyValueError::new_err(
                "episode loading requires real text embeddings and d_text > 0",
            ));
        }
        let root = Path::new(pre_path);
        if root.join("archive_manifest.json").exists()
            || root.join("v2").exists()
            || root.join("cell_numbers.bin").exists()
        {
            return Err(PyValueError::new_err(
                "format-v2 archive: read it with the v2 reader, not this decoder",
            ));
        }
        let (num_nodes, a) = py.detach(|| -> PyResult<_> {
            let dataset = Dataset::load(root, embedding_model, d_text)?;
            Ok((dataset.num_nodes(), decode(&dataset, d_text)?))
        })?;
        macro_rules! named_arrays {
            ($($name:ident),*) => {
                vec![
                    ("num_nodes", num_nodes).into_py_any(py)?,
                    $((stringify!($name), PyArray1::from_vec(py, a.$name)).into_py_any(py)?,)*
                ]
            };
        }
        Ok(named_arrays!(
            node_timestamps,
            node_table_name_idxs,
            f2p_rowptr,
            f2p_col_idx,
            p2f_rowptr,
            p2f_col_idx,
            p2f_table_name_idxs,
            p2f_table_types,
            p2f_timestamps,
            cell_nodes,
            cell_columns,
            cell_types,
            cell_numbers,
            cell_datetimes,
            cell_booleans,
            cell_text_ids,
            cell_codes,
            cell_num_classes,
            embedding_bits
        ))
    }
}
