use std::sync::Arc;
use tokio::sync::Mutex;

use env_logger::{Env, init_from_env};
use log::info;

mod audio;
mod model;
mod server;

#[tokio::main]
async fn main() {
    init_from_env(Env::default().default_filter_or("info"));

    let model_path = "./models/ai_voice_detector.onnx";
    let engine = model::InferenceEngine::new(model_path).expect("Failed to load model");

    let shared_engine = Arc::new(Mutex::new(engine));

    info!("Initializing vorsana.");

    // let addr: &str = "127.0.0.1:3000";
    // server::start_server(addr, shared_engine).await;

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{port}");

    server::start_server(&addr, shared_engine).await;
}
