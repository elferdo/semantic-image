use std::path::{Path, PathBuf};

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Serialize;
use tokio::fs;

use error_stack::{Report, ResultExt};
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
use walkdir::WalkDir;

#[derive(Debug, Error)]
enum AppError {
    #[error("application error")]
    Error,
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
    async fn describe(&self, image_path: PathBuf) -> Result<ImageComment, Report<AppError>> {
        let comedian_agent = self.client
        .agent("gemma3")
        .preamble("Describe the scene, including mood, places and people. Don't say what you are going to do and don't ask questions.")
        .build();

        let mut image = Image::default();

        let image_bytes = fs::read(&image_path)
            .await
            .change_context(AppError::Error)?;

        let image_base64 = BASE64_STANDARD.encode(image_bytes);

        image.data = DocumentSourceKind::Base64(image_base64);

        image.media_type = Some(ImageMediaType::JPEG);

        // Prompt the agent and print the response
        let comment = comedian_agent
            .prompt(image)
            .await
            .change_context(AppError::Error)?;

        let result = ImageComment {
            path: image_path,
            comment,
        };

        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), Report<AppError>> {
    tracing_subscriber::fmt().pretty().init();

    tracing::info!("Hola");

    let client: ollama::Client = ollama::Client::new(Nothing).unwrap();

    let agent = Agent { client };

    let entries: Vec<_> = WalkDir::new("/home/fernando/Fotos/JPEG")
        .into_iter()
        .filter_map(|entry| entry.map(|ed| agent.describe(ed.path().to_path_buf())).ok())
        .collect();

    for e in entries {
        if let Ok(f) = e.await {
            println!(
                "{}",
                serde_json::to_string(&f).change_context(AppError::Error)?
            );
        }
    }

    Ok(())
}
