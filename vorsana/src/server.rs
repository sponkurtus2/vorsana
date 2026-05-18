use crate::audio::AudioProcessor;
use crate::model::InferenceEngine;
use axum::body::Bytes;
use axum::extract::DefaultBodyLimit;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Query, State, WebSocketUpgrade};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::response::IntoResponse;
use axum::{
    Json, Router,
    routing::{get, post},
};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

const AI_THRESHOLD: f32 = 0.44;
const UNCERTAIN_MARGIN: f32 = 0.015;

struct AppState {
    engine: Arc<Mutex<InferenceEngine>>,
}

#[derive(Deserialize)]
struct AnalyzeQuery {
    sample_rate: usize,
}

#[derive(Serialize)]
struct AnalyzeResponse {
    label: String,
    probability_ai: f32,
    probability_human: f32,
    threshold: f32,
    confidence_margin: f32,
    segments_analyzed: usize,
    probability_ai_min: f32,
    probability_ai_max: f32,
    probability_ai_preview: Vec<f32>,
    raw_outputs_preview: Vec<Vec<f32>>,
    positive_class_probability_preview: Vec<f32>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// Initializes the Axum router and binds the TCP listener.
pub async fn start_server(addr: &str, engine: Arc<Mutex<InferenceEngine>>) {
    let state = Arc::new(AppState { engine });

    let cors = CorsLayer::new()
        .allow_origin("http://localhost:4321".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(tower_http::cors::Any);

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/audio/analyze", post(analyze_audio_handler))
        .route("/model/health", get(model_health_handler))
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .with_state(state);

    let listener = TcpListener::bind(addr)
        .await
        .expect("Failed to bind TCP listener");

    info!("Server listening on ws://{}", addr);

    axum::serve(listener, app)
        .await
        .expect("Error: server failed to start");
}

async fn model_health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.lock().await;
    let metadata = engine.get_health_metadata();
    Json(metadata)
}

async fn analyze_audio_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<AnalyzeQuery>,
    body: Bytes,
) -> impl IntoResponse {
    if query.sample_rate == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "sample_rate must be greater than zero".to_string(),
            }),
        )
            .into_response();
    }

    if body.len() < 4 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Request body is empty or too small".to_string(),
            }),
        )
            .into_response();
    }

    let samples: Vec<f32> = body
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("Invalid Float32 chunk")))
        .collect();

    if samples.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "No valid Float32 samples found in request body".to_string(),
            }),
        )
            .into_response();
    }

    let mut processor = AudioProcessor::new(query.sample_rate);
    let filterbank = AudioProcessor::create_mel_filterbank();

    let frequency_windows = processor.process_file_samples_to_frequency_domain(&samples);

    let mut mel_history: Vec<Vec<f32>> = Vec::with_capacity(401);
    let mut probabilities: Vec<f32> = Vec::new();
    let mut raw_outputs_preview: Vec<Vec<f32>> = Vec::new();
    let mut positive_class_probability_preview: Vec<f32> = Vec::new();
    let mut engine = state.engine.lock().await;

    for magnitudes in frequency_windows {
        let mel_bins = AudioProcessor::apply_mel_filters(&magnitudes, &filterbank);
        mel_history.push(mel_bins);

        if mel_history.len() == 401 {
            match engine.predict_debug(mel_history.clone()) {
                Ok(prediction) => {
                    probabilities.push(prediction.probability_ai);

                    if raw_outputs_preview.len() < 5 {
                        raw_outputs_preview.push(prediction.raw_outputs);
                        positive_class_probability_preview
                            .push(prediction.probability_positive_class);
                    }
                }
                Err(e) => {
                    error!("Inference error while analyzing uploaded audio: {:?}", e);

                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: "Inference failed while analyzing uploaded audio".to_string(),
                        }),
                    )
                        .into_response();
                }
            }
            mel_history.drain(0..40);
        }
    }

    if probabilities.is_empty() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: "Audio is too short or did not produce enough mel frames for inference"
                    .to_string(),
            }),
        )
            .into_response();
    }

    let probability_ai = probabilities.iter().sum::<f32>() / probabilities.len() as f32;

    let probability_ai_min = probabilities.iter().copied().fold(f32::INFINITY, f32::min);

    let probability_ai_max = probabilities
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let probability_ai_preview = probabilities.iter().copied().take(10).collect::<Vec<f32>>();

    let probability_human = 1.0 - probability_ai;

    let confidence_margin = (probability_ai - AI_THRESHOLD).abs();

    let label = if probability_ai >= AI_THRESHOLD {
        "ai".to_string()
    } else {
        "human".to_string()
    };

    let confidence_margin = (probability_ai - AI_THRESHOLD).abs();

    let response = AnalyzeResponse {
        label,
        probability_ai,
        probability_human,
        threshold: AI_THRESHOLD,
        confidence_margin,
        segments_analyzed: probabilities.len(),
        probability_ai_min,
        probability_ai_max,
        probability_ai_preview,
        raw_outputs_preview,
        positive_class_probability_preview,
    };

    (StatusCode::OK, Json(response)).into_response()
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    info!("Websocket client connected");

    let mut processor = AudioProcessor::new(48000);

    let mut mel_history: Vec<Vec<f32>> = Vec::with_capacity(401);

    let filterbank = AudioProcessor::create_mel_filterbank();

    while let Some(result) = socket.recv().await {
        match result {
            Ok(Message::Binary(bytes)) => {
                if bytes.len() == 16384 {
                    let frequency_windows = processor.process_to_frequency_domain(&bytes);

                    for magnitudes in frequency_windows {
                        let mel_bins = AudioProcessor::apply_mel_filters(&magnitudes, &filterbank);
                        mel_history.push(mel_bins);

                        if mel_history.len() == 401 {
                            let mut engine = state.engine.lock().await;

                            match engine.predict(mel_history.clone()) {
                                Ok(probability) => {
                                    let is_ai = probability > 0.5;
                                    let response = if is_ai {
                                        format!("DETECTED: AI Voice ({:.2}%)", probability * 100.0)
                                    } else {
                                        format!(
                                            "DETECTED: Human Voice ({:.2}%)",
                                            (1.0 - probability) * 100.0
                                        )
                                    };

                                    if let Err(e) =
                                        socket.send(Message::Text(response.into())).await
                                    {
                                        error!("Failed to send WS message: {}", e);
                                    }
                                }
                                Err(e) => error!("Inference error: {:?}", e),
                            }

                            mel_history.drain(0..40);
                        }
                    }
                } else {
                    warn!("Received malformed chunk: {} bytes", bytes.len());
                }
            }
            Ok(Message::Close(_)) => break,
            Err(e) => {
                error!("Websocket connection error: {}", e);
                break;
            }
            _ => {}
        }
    }
    info!("Client disconnected");
}
