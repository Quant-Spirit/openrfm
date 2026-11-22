// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
//! The text vocabulary of an archive and its sentence embeddings (bf16, one row per text id).
use crate::pre::Cli;
use crate::text_encoder;
use anyhow::Result;
use half::bf16;
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub(crate) fn output_path(pre_path: &str, stem: &str, ext: &str) -> String {
    format!("{}/{}.{}", pre_path, stem, ext)
}

pub(crate) fn encoder_file_tag(model: &str) -> String {
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

fn write_text_vocab(pre_path: &str, text_vec: &[String]) -> Result<()> {
    let text_path = output_path(pre_path, "text", "json");
    let mut writer = BufWriter::new(fs::File::create(text_path)?);
    serde_json::to_writer(&mut writer, text_vec)?;

    let mut text_map: HashMap<String, i32> = HashMap::with_capacity(text_vec.len());
    for (idx, s) in text_vec.iter().enumerate() {
        text_map.insert(s.clone(), idx as i32);
    }
    let text_map_path = output_path(pre_path, "text_map", "json");
    let mut writer = BufWriter::new(fs::File::create(text_map_path)?);
    serde_json::to_writer(&mut writer, &serde_json::to_value(&text_map)?)?;
    Ok(())
}

/// The embedding matrix as little-endian bf16, written to a fresh inode that is renamed into place.
fn write_text_embeddings(
    pre_path: &str,
    encoder_tag: &str,
    embeddings: &ndarray::Array2<f32>,
) -> Result<()> {
    let path = Path::new(pre_path).join(format!("text_emb_{}.bin", encoder_tag));
    let pending = pending_path(&path)?;
    {
        let mut writer = BufWriter::new(fs::File::create(&pending)?);
        for value in embeddings.iter() {
            writer.write_all(&bf16::from_f32(*value).to_bits().to_le_bytes())?;
        }
        writer.flush()?;
    }
    fs::rename(&pending, &path)?;
    Ok(())
}

/// A same-directory scratch name for an atomic replace of `path`.
fn pending_path(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("no file name: {}", path.display()))?
        .to_string_lossy();
    let pending = parent.join(format!(".{}.pending.{}", name, std::process::id()));
    if pending.exists() {
        fs::remove_file(&pending)?;
    }
    Ok(pending)
}

pub(crate) fn encode_and_write_text_embeddings(cli: &Cli, pre_path: &str, text_vec: &[String]) -> Result<()> {
    let embeddings = if text_vec.is_empty() {
        println!(
            "[text-cache] {} empty vocabulary; encoder not started",
            cli.db_name
        );
        ndarray::Array2::<f32>::zeros((0, 0))
    } else {
        let encoder = text_encoder::TextEncoder::new(
            &cli.text_encoder,
            Some(&cli.text_encoder_device),
            cli.text_encoder_batch,
            cli.text_encoder_max_length,
        )?
        .with_policy(cli.allow_cpu, cli.text_embedding_cache_only)?;

        let embeddings = encoder.encode_batch(text_vec)?;

        if let (Some((hits, misses)), Some(root)) = (encoder.cache_stats(), encoder.cache_root()) {
            println!(
                "[text-cache] {} hits={} misses={} root={}",
                cli.db_name,
                hits,
                misses,
                root.display()
            );
        }
        embeddings
    };

    let encoder_tag = encoder_file_tag(&cli.text_encoder);
    write_text_embeddings(pre_path, &encoder_tag, &embeddings)
}

pub(crate) fn write_standard_text_outputs<F>(
    pre_path: &str,
    text_to_idx: HashMap<String, i32>,
    encode_and_write: F,
) -> Result<()>
where
    F: FnOnce(&str, &[String]) -> Result<()>,
{
    let mut text_vec = vec![String::new(); text_to_idx.len()];
    for (text, idx) in text_to_idx {
        text_vec[idx as usize] = text;
    }
    write_text_vocab(pre_path, &text_vec)?;
    encode_and_write(pre_path, &text_vec)
}

#[cfg(test)]
mod text_embedding_output_tests {
    use super::{write_standard_text_outputs, write_text_embeddings};
    use half::bf16;
    use ndarray::Array2;
    use std::collections::HashMap;

    #[test]
    fn standard_pass_writes_one_embedding_row_per_standard_vocabulary_entry() {
        let temp = tempfile::tempdir().unwrap();
        let pre_path = temp.path().to_str().unwrap();
        let text_to_idx = HashMap::from([
            ("column of table".to_string(), 1),
            ("table".to_string(), 0),
            ("value".to_string(), 2),
        ]);

        write_standard_text_outputs(pre_path, text_to_idx, |pre_path, text_vec| {
            assert_eq!(
                text_vec.iter().map(String::as_str).collect::<Vec<_>>(),
                vec!["table", "column of table", "value"]
            );
            let embeddings = Array2::from_shape_vec(
                (text_vec.len(), 2),
                vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            )
            .unwrap();
            write_text_embeddings(pre_path, "test-model", &embeddings)
        })
        .unwrap();

        let vocabulary: Vec<String> =
            serde_json::from_slice(&std::fs::read(temp.path().join("text.json")).unwrap()).unwrap();
        let embedding_bytes = std::fs::read(temp.path().join("text_emb_test-model.bin")).unwrap();
        let embedding_dim = 2;
        assert_eq!(
            embedding_bytes.len(),
            vocabulary.len() * embedding_dim * std::mem::size_of::<u16>()
        );
        let row_bytes = embedding_dim * std::mem::size_of::<u16>();
        let row_heads: Vec<f32> = embedding_bytes
            .chunks_exact(row_bytes)
            .map(|row| bf16::from_bits(u16::from_le_bytes([row[0], row[1]])).to_f32())
            .collect();
        assert_eq!(row_heads, vec![10.0, 20.0, 30.0]);
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect()
    }

    fn nlink(path: &std::path::Path) -> u64 {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(path).unwrap().nlink()
    }

    #[test]
    fn rewrite_never_reaches_a_preexisting_hard_link() {
        let temp = tempfile::tempdir().unwrap();
        let db = temp.path().join("db");
        std::fs::create_dir_all(&db).unwrap();
        let emb = Array2::from_shape_vec((1, 2), vec![7.0, 8.0]).unwrap();
        write_text_embeddings(db.to_str().unwrap(), "m", &emb).unwrap();
        std::fs::hard_link(db.join("text_emb_m.bin"), &temp.path().join("alias.bin")).unwrap();
        write_text_embeddings(db.to_str().unwrap(), "m", &Array2::zeros((1, 2))).unwrap();
        assert_eq!(
            std::fs::read(temp.path().join("alias.bin")).unwrap(),
            bf16_bytes(&[7.0, 8.0])
        );
        assert_eq!(nlink(&db.join("text_emb_m.bin")), 1);
    }
}
