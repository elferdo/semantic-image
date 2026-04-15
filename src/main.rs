mod cli;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use clap::Parser;
use directories::ProjectDirs;
use serde::Serialize;
use tokio::fs;

use error_stack::{IntoReport, Report, ResultExt};
use persistent_kv::{Config, PersistentKeyValueStore};
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::embeddings::embedding::ImageEmbeddingModel;
use rig::message::{DocumentSourceKind, Image, ImageMediaType};
use rig::{
    client::{EmbeddingsClient, Nothing},
    providers::ollama,
};
use thiserror::Error;
use tokio::task::JoinHandle;
use tracing::instrument::WithSubscriber;
use tracing::{debug, instrument};
use tracing_subscriber::layer::SubscriberExt;
use walkdir::WalkDir;

use crate::cli::Args;

type KVStore = PersistentKeyValueStore<String, String>;

#[derive(Debug, Error)]
enum AppError {
    #[error("application error")]
    Error,

    #[error("reading image")]
    ImageReadError,

    #[error("LLM agent error")]
    LlmAgentError,

    #[error("extracting file extension from path")]
    FileExtension,

    #[error("accessing application configuration directory")]
    ConfigDir,

    #[error("accessing persistent kv store for image descriptions")]
    KVStoreAccess,
}

struct Agent {
    client: ollama::Client,
}

#[derive(Serialize)]
struct ImageComment {
    path: PathBuf,
    comment: String,
}

impl Agent {
    #[instrument(err, skip(self))]
    async fn describe(&self, image_path: PathBuf) -> Result<ImageComment, Report<AppError>> {
        let comedian_agent = self.client
        .agent("gemma3")
        .preamble("Describe the scene, including mood, places, prominent elements and people. Don't say what you are going to do and don't ask questions.")
        .build();

        let mut image = Image::default();

        let image_bytes = fs::read(&image_path)
            .await
            .change_context(AppError::ImageReadError)?;

        let image_base64 = BASE64_STANDARD.encode(image_bytes);

        image.data = DocumentSourceKind::Base64(image_base64);

        image.media_type = Some(ImageMediaType::JPEG);

        // Prompt the agent and print the response
        let comment = comedian_agent
            .prompt(image)
            .await
            .change_context(AppError::LlmAgentError)?;

        let result = ImageComment {
            path: image_path,
            comment,
        };

        Ok(result)
    }
}

fn is_file_extension_jpg(path: &Path) -> bool {
    matches!(path.extension(), Some(extension) if {
        if extension.to_ascii_lowercase() == "jpg" || extension.to_ascii_lowercase() == "jpeg" {
            true
        } else {
            false
        }
    })
}

fn is_file_already_described(path: &Path, description_store: &KVStore) -> Result<bool, AppError> {
    let description_exists = description_store.get(&path.to_string_lossy().to_string());

    Ok(description_exists.is_some())
}

fn init_data_dir() -> Result<PathBuf, Report<AppError>> {
    let Some(user_dirs) = ProjectDirs::from("com", "fhcarron", env!("CARGO_PKG_NAME")) else {
        return Err(AppError::ConfigDir.into_report());
    };

    let d = user_dirs.data_local_dir().to_owned();

    if let Ok(exists) = d.try_exists() {
        if !exists {
            std::fs::create_dir(&d).change_context(AppError::ConfigDir)?;

            eprintln!("Created data directory {}", d.display());
        }

        Ok(d)
    } else {
        Err(AppError::ConfigDir.into_report())
    }
}

fn open_kv_store(path: &Path) -> Result<KVStore, Report<AppError>> {
    let store = PersistentKeyValueStore::new(path.join("kvstore"), Config::default())
        .map_err(|_| AppError::KVStoreAccess)?;

    Ok(store)
}

#[tokio::main]
async fn main() -> Result<(), Report<AppError>> {
    tracing_subscriber::fmt().pretty().init();

    let args = Args::parse();

    let data_dir = init_data_dir()?;
    let description_store = open_kv_store(&data_dir)?;

    // Ollama must be running locally
    let client: ollama::Client = ollama::Client::new(Nothing).unwrap();

    let agent = Agent { client };

    let entries: Vec<_> = WalkDir::new(args.path)
        .into_iter()
        .filter(|e| {
            e.as_ref()
                .is_ok_and(|dir_entry| is_file_extension_jpg(dir_entry.path()))
        })
        .filter(|e| {
            e.as_ref().is_ok_and(|dir_entry| {
                !is_file_already_described(dir_entry.path(), &description_store).expect("may fail")
            })
        })
        .filter_map(|entry| entry.map(|ed| agent.describe(ed.path().to_path_buf())).ok())
        .collect();

    for e in entries {
        if let Ok(f) = e.await {
            debug!("{:?}", f.path);

            description_store
                .set(f.path.to_string_lossy().to_string(), f.comment)
                .expect("why is this failing?");
        }
    }

    Ok(())
}
