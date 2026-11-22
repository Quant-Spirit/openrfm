// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
use anyhow::{Result, anyhow};
use ndarray::Array2;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

fn normalize_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim_matches(|c| matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{b}' | '\u{c}'))
        .to_string()
}

/// Persistent, content-addressed cache for sentence-embedding vectors.
struct EmbeddingCache {
    root: PathBuf,
    tag: String,
}

impl EmbeddingCache {
    fn from_env(model_name: &str, max_length: usize) -> Option<Self> {
        let root = std::env::var("RUSTLER_TEXT_EMB_CACHE").ok()?;
        let root = root.trim();
        if root.is_empty() {
            return None;
        }
        // Anything that changes the produced vector must change the key; embeddings are not
        // normalised (`norm0`).
        let tag = format!(
            "sha256-v1_{:x}__norm0__len{}",
            Sha256::digest(
                model_name
                    .strip_prefix("sentence-transformers/")
                    .unwrap_or(model_name)
                    .as_bytes()
            ),
            max_length
        );
        Some(Self {
            root: PathBuf::from(root),
            tag,
        })
    }

    fn path_for(&self, text: &str) -> PathBuf {
        let hex = format!("{:x}", Sha256::digest(normalize_text(text).as_bytes()));
        self.root.join(&self.tag).join(&hex[..2]).join(hex)
    }

    fn read(&self, text: &str) -> Option<Vec<f32>> {
        let bytes = fs::read(self.path_for(text)).ok()?;
        if bytes.len() < 4 {
            return None;
        }
        let bf16 = bytes.starts_with(b"RTBF16\0\x02");
        let bytes = if bf16 { &bytes[8..] } else { &bytes[..] };
        if bytes.len() < 4 {
            return None;
        }
        let text_len = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
        let dim_at = 4 + text_len;
        if bytes.len() < dim_at + 4 {
            return None;
        }
        // Verify the stored text: a digest collision must never surface as a
        // silently wrong embedding.
        if &bytes[4..dim_at] != text.as_bytes() {
            return None;
        }
        let dim = u32::from_le_bytes(bytes[dim_at..dim_at + 4].try_into().ok()?) as usize;
        let vec_at = dim_at + 4;
        let width = if bf16 { 2 } else { 4 };
        if bytes.len() != vec_at + dim * width {
            return None;
        }
        let mut out = Vec::with_capacity(dim);
        for i in 0..dim {
            let off = vec_at + i * width;
            out.push(if bf16 {
                half::bf16::from_bits(u16::from_le_bytes(bytes[off..off + 2].try_into().ok()?))
                    .to_f32()
            } else {
                f32::from_le_bytes(bytes[off..off + 4].try_into().ok()?)
            });
        }
        Some(out)
    }

    fn write(&self, text: &str, vector: &[f32]) {
        let path = self.path_for(text);
        let Some(parent) = path.parent() else { return };
        if fs::create_dir_all(parent).is_err() {
            return;
        }
        let mut bytes = Vec::with_capacity(16 + text.len() + vector.len() * 2);
        bytes.extend_from_slice(b"RTBF16\0\x02");
        bytes.extend_from_slice(&(text.len() as u32).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
        bytes.extend_from_slice(&(vector.len() as u32).to_le_bytes());
        for value in vector {
            bytes.extend_from_slice(&half::bf16::from_f32(*value).to_bits().to_le_bytes());
        }
        // Write-then-rename so concurrent preprocessing shards never observe a
        // partially written entry.
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        let wrote = fs::File::create(&tmp).and_then(|mut file| file.write_all(&bytes));
        if wrote.is_ok() {
            let _ = fs::rename(&tmp, &path);
        } else {
            let _ = fs::remove_file(&tmp);
        }
    }
}

pub struct TextEncoder {
    model: Mutex<Option<Py<PyAny>>>,
    model_name: String,
    requested_device: Option<String>,
    resolved_device: Mutex<Option<String>>,
    batch_size: usize,
    max_length: usize,
    cache: Option<EmbeddingCache>,
    stats: Mutex<(usize, usize)>,
    allow_cpu: bool,
    cache_only: bool,
}

impl TextEncoder {
    pub fn new(
        model_name: &str,
        device: Option<&str>,
        batch_size: usize,
        max_length: usize,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(anyhow!("text encoder batch size must be positive"));
        }
        Ok(Self {
            model: Mutex::new(None),
            model_name: model_name.to_string(),
            requested_device: device.map(str::to_string),
            resolved_device: Mutex::new(None),
            batch_size,
            max_length,
            cache: EmbeddingCache::from_env(model_name, max_length),
            stats: Mutex::new((0, 0)),
            allow_cpu: false,
            cache_only: false,
        })
    }

    pub fn with_policy(mut self, allow_cpu: bool, cache_only: bool) -> Result<Self> {
        if cache_only && self.cache.is_none() {
            return Err(anyhow!(
                "cache-only preprocessing requires RUSTLER_TEXT_EMB_CACHE"
            ));
        }
        self.allow_cpu = allow_cpu;
        self.cache_only = cache_only;
        Ok(self)
    }

    /// Builds the model on first use, so a fully cached vocabulary never loads it.
    fn ensure_model(&self, py: Python<'_>) -> Result<(Py<PyAny>, String)> {
        let mut guard = self
            .model
            .lock()
            .map_err(|_| anyhow!("text encoder model mutex poisoned"))?;
        let mut device_guard = self
            .resolved_device
            .lock()
            .map_err(|_| anyhow!("text encoder device mutex poisoned"))?;

        if let (Some(model), Some(device)) = (guard.as_ref(), device_guard.as_ref()) {
            return Ok((model.clone_ref(py), device.clone()));
        }

        let torch = py.import("torch")?;

        let cuda_available = torch
            .getattr("cuda")?
            .getattr("is_available")?
            .call0()?
            .is_truthy()?;
        let requested = self.requested_device.as_deref().unwrap_or("auto");
        let resolved_device = match requested {
            "auto" if cuda_available => "cuda".to_string(),
            "auto" | "cpu" if self.allow_cpu => "cpu".to_string(),
            d if (d == "cuda" || d.starts_with("cuda:")) && cuda_available => d.to_string(),
            d if (d == "cuda" || d.starts_with("cuda:")) && self.allow_cpu => "cpu".to_string(),
            _ => {
                return Err(anyhow!(
                    "Text embedding requires available CUDA; CPU encoding requires --allow-cpu (requested {requested})"
                ));
            }
        };

        let st_cls = py
            .import("sentence_transformers")?
            .getattr("SentenceTransformer")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("device", resolved_device.as_str())?;
        let model_obj = st_cls.call((self.model_name.as_str(),), Some(&kwargs))?;
        model_obj.setattr("max_seq_length", self.max_length)?;
        let model_obj = model_obj.unbind();

        *guard = Some(model_obj.clone_ref(py));
        *device_guard = Some(resolved_device.clone());
        Ok((model_obj, resolved_device))
    }

    fn encode_uncached(&self, texts: &[String]) -> Result<Array2<f32>> {
        Python::attach(|py| -> Result<Array2<f32>> {
            let (model, device) = self.ensure_model(py)?;
            let model = model.bind(py);
            let items: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let py_list = PyList::new(py, &items)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("batch_size", self.batch_size)?;
            kwargs.set_item("show_progress_bar", false)?;
            kwargs.set_item("convert_to_numpy", true)?;
            kwargs.set_item("device", device)?;
            kwargs.set_item("normalize_embeddings", false)?;

            let encoded = model.call_method("encode", (py_list.as_any(),), Some(&kwargs))?;
            let out = encoded
                .cast::<PyArray2<f32>>()
                .map_err(|_| anyhow!("encode() did not return float32 ndarray"))?;

            Ok(out.to_owned_array())
        })
    }

    pub fn encode_batch(&self, texts: &[String]) -> Result<Array2<f32>> {
        if texts.is_empty() {
            return Ok(Array2::<f32>::zeros((0, 0)));
        }

        let texts: Vec<String> = texts.iter().map(|text| normalize_text(text)).collect();
        let cache = self.cache.as_ref();

        let resolved: Vec<Option<Vec<f32>>> = texts
            .par_iter()
            .map(|text| cache.and_then(|cache| cache.read(text)))
            .collect();

        let mut missing_order: Vec<String> = Vec::new();
        let mut missing_index: HashMap<String, usize> = HashMap::new();
        for (text, slot) in texts.iter().zip(resolved.iter()) {
            if slot.is_none() {
                let key = format!("{:x}", Sha256::digest(text.as_bytes()));
                if !missing_index.contains_key(&key) {
                    missing_index.insert(key, missing_order.len());
                    missing_order.push(text.clone());
                }
            }
        }

        let hits = resolved.iter().filter(|slot| slot.is_some()).count();
        if let Ok(mut stats) = self.stats.lock() {
            stats.0 += hits;
            stats.1 += texts.len() - hits;
        }

        let mut computed: Vec<Vec<f32>> = Vec::new();
        if !missing_order.is_empty() {
            if self.cache_only {
                let sample: Vec<String> = missing_order
                    .iter()
                    .take(10)
                    .map(|text| format!("{text:?}"))
                    .collect();
                return Err(anyhow!(
                    "Shared text cache has {} misses; warm the corpus vocabulary on GPU first (first {}: {})",
                    missing_order.len(),
                    sample.len(),
                    sample.join(", ")
                ));
            }
            let encoded = self.encode_uncached(&missing_order)?;
            for (row, text) in encoded.rows().into_iter().zip(missing_order.iter()) {
                let vector: Vec<f32> = row.to_vec();
                if let Some(cache) = cache {
                    cache.write(text, &vector);
                }
                computed.push(vector);
            }
        }

        let dim = computed
            .first()
            .map(Vec::len)
            .or_else(|| resolved.iter().flatten().next().map(Vec::len))
            .ok_or_else(|| anyhow!("text encoder produced no embedding dimension"))?;

        let mut out = Array2::<f32>::zeros((texts.len(), dim));
        for (row_idx, (slot, text)) in resolved.iter().zip(texts.iter()).enumerate() {
            let vector = match slot {
                Some(vector) => vector,
                None => {
                    let which = missing_index
                        .get(&format!("{:x}", Sha256::digest(text.as_bytes())))
                        .copied()
                        .ok_or_else(|| anyhow!("missing text index for {text:?}"))?;
                    &computed[which]
                }
            };
            if vector.len() != dim {
                return Err(anyhow!(
                    "cached embedding dimension {} differs from {dim}; clear RUSTLER_TEXT_EMB_CACHE",
                    vector.len()
                ));
            }
            for (col_idx, value) in vector.iter().enumerate() {
                out[[row_idx, col_idx]] = *value;
            }
        }
        Ok(out)
    }

    pub fn cache_stats(&self) -> Option<(usize, usize)> {
        self.cache.as_ref()?;
        self.stats.lock().ok().map(|stats| *stats)
    }

    pub fn cache_root(&self) -> Option<&Path> {
        self.cache.as_ref().map(|cache| cache.root.as_path())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_cache_reads_preserve_order_and_content() {
        let dir = std::env::temp_dir().join(format!(
            "rustler_text_encoder_test_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: this test does not spawn threads that also mutate this
        // process's environment; EmbeddingCache::from_env reads it once,
        // synchronously, right after this call.
        unsafe { std::env::set_var("RUSTLER_TEXT_EMB_CACHE", &dir) };

        let cache = EmbeddingCache::from_env("all-MiniLM-L12-v2", 256).unwrap();
        let texts: Vec<String> = (0..500).map(|i| format!("nuisance-token-{i}")).collect();
        for (i, text) in texts.iter().enumerate() {
            cache.write(text, &[i as f32, (i * 2) as f32]);
        }

        let encoder = TextEncoder::new("all-MiniLM-L12-v2", Some("cpu"), 32, 256)
            .unwrap()
            .with_policy(true, true)
            .unwrap();
        let out = encoder
            .encode_batch(&texts)
            .expect("every text is a pre-written cache hit; no model load should be needed");

        assert_eq!(out.nrows(), texts.len());
        for (i, row) in out.rows().into_iter().enumerate() {
            assert_eq!(
                row[0],
                half::bf16::from_f32(i as f32).to_f32(),
                "row {i} mismatched after parallel resolution"
            );
            assert_eq!(
                row[1],
                half::bf16::from_f32((i * 2) as f32).to_f32(),
                "row {i} mismatched after parallel resolution"
            );
        }
        assert_eq!(encoder.cache_stats(), Some((500, 0)));

        unsafe { std::env::remove_var("RUSTLER_TEXT_EMB_CACHE") };
        std::fs::remove_dir_all(&dir).ok();
    }
}
