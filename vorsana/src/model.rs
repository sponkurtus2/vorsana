use ndarray::Array4;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::path::Path;

use serde::Serialize;

/// Defines what the single ONNX output represents after sigmoid.
///
/// true  => sigmoid(output) means probability of AI voice.
/// false => sigmoid(output) means probability of human voice.
const POSITIVE_CLASS_IS_AI: bool = true;

/// Structs to check the health of an onnx file.
#[derive(Serialize)]
pub struct TensorDetails {
    pub name: String,
    pub data_type: String,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub status: String,
    pub inputs: Vec<TensorDetails>,
    pub outputs: Vec<TensorDetails>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictionDebug {
    pub probability_ai: f32,
    pub probability_positive_class: f32,
    pub raw_outputs: Vec<f32>,
}

/// InferenceEngine handles the loading and execution of the ONNX model.
pub struct InferenceEngine {
    session: Session,
}

impl InferenceEngine {
    /// Loads the ONNX model and automatically associates the .onnx.data file.
    ///
    /// # Arguments
    /// * `model_path` - Path to the .onnx file.
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    /// Prints model metadata to identify required input shapes and names.
    pub fn inspect_model(&self) {
        println!("--- Model Metadata ---");

        for (i, input) in self.session.inputs().iter().enumerate() {
            println!(
                "Input [{}]: name='{}', type={:?}",
                i,
                input.name(),
                input.dtype()
            );
        }

        for (i, output) in self.session.outputs().iter().enumerate() {
            println!(
                "Output [{}]: name='{}', type={:?}",
                i,
                output.name(),
                output.dtype()
            );
        }

        println!("----------------------");
    }

    pub fn predict(&mut self, mel_spectrogram: Vec<Vec<f32>>) -> Result<f32, ort::Error> {
        let debug = self.predict_debug(mel_spectrogram)?;
        Ok(debug.probability_ai)
    }

    /// Executes inference on a given spectrogram segment.
    /// Maps and normalizes the [401, 128] mel history into the model's expected [1, 3, 128, 401] shape.
    // pub fn predict(&mut self, mel_spectrogram: Vec<Vec<f32>>) -> Result<f32, ort::Error> {
    //     let rows = 128;
    //     let cols = 401;
    //
    //     let mut min_val = f32::INFINITY;
    //     let mut max_val = f32::NEG_INFINITY;
    //
    //     for frame in mel_spectrogram.iter() {
    //         for &val in frame.iter() {
    //             if val < min_val {
    //                 min_val = val;
    //             }
    //             if val > max_val {
    //                 max_val = val;
    //             }
    //         }
    //     }
    //
    //     let range = if (max_val - min_val).abs() < 1e-5 {
    //         1.0
    //     } else {
    //         max_val - min_val
    //     };
    //
    //     let mut input_4d = Array4::<f32>::zeros((1, 3, rows, cols));
    //
    //     for (t, frame) in mel_spectrogram.iter().enumerate() {
    //         for (m, &val) in frame.iter().enumerate() {
    //             let normalized_val = (val - min_val) / range;
    //             for c in 0..3 {
    //                 input_4d[[0, c, m, t]] = normalized_val;
    //             }
    //         }
    //     }
    //
    //     let shape = input_4d.shape().to_vec();
    //     let data = input_4d.into_raw_vec();
    //     let input_tensor = Value::from_array((shape, data))?;
    //
    //     let outputs = self
    //         .session
    //         .run(ort::inputs!["mel_spectrogram" => input_tensor])?;
    //
    //     let (_shape, slice) = outputs[0].try_extract_tensor::<f32>()?;
    //     let logit = slice[0];
    //     let probability = 1.0 / (1.0 + (-logit).exp());
    //
    //     Ok(probability)
    // }

    pub fn get_health_metadata(&self) -> ModelInfo {
        let inputs = self
            .session
            .inputs()
            .iter()
            .map(|input| TensorDetails {
                name: input.name().to_string(),
                data_type: format!("{:?}", input.dtype()),
            })
            .collect();

        let outputs = self
            .session
            .outputs()
            .iter()
            .map(|output| TensorDetails {
                name: output.name().to_string(),
                data_type: format!("{:?}", output.dtype()),
            })
            .collect();

        ModelInfo {
            status: "Happy".to_string(),
            inputs,
            outputs,
        }
    }

    pub fn predict_debug(
        &mut self,
        mel_spectrogram: Vec<Vec<f32>>,
    ) -> Result<PredictionDebug, ort::Error> {
        let rows = 128;
        let cols = 401;

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for frame in mel_spectrogram.iter() {
            for &val in frame.iter() {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
        }

        let range = if (max_val - min_val).abs() < 1e-5 {
            1.0
        } else {
            max_val - min_val
        };

        let mut input_4d = Array4::<f32>::zeros((1, 3, rows, cols));

        for (t, frame) in mel_spectrogram.iter().enumerate() {
            for (m, &val) in frame.iter().enumerate() {
                let normalized_val = (val - min_val) / range;

                for c in 0..3 {
                    input_4d[[0, c, m, t]] = normalized_val;
                }
            }
        }

        let shape = input_4d.shape().to_vec();
        let data = input_4d.into_raw_vec();
        let input_tensor = Value::from_array((shape, data))?;

        let outputs = self
            .session
            .run(ort::inputs!["mel_spectrogram" => input_tensor])?;

        let (_shape, slice) = outputs[0].try_extract_tensor::<f32>()?;

        let raw_outputs = slice.to_vec();

        let probability_positive_class = if raw_outputs.len() == 1 {
            let value = raw_outputs[0];

            if (0.0..=1.0).contains(&value) {
                value
            } else {
                1.0 / (1.0 + (-value).exp())
            }
        } else if raw_outputs.len() >= 2 {
            let a = raw_outputs[0];
            let b = raw_outputs[1];

            let max = a.max(b);
            let exp_a = (a - max).exp();
            let exp_b = (b - max).exp();

            let softmax_0 = exp_a / (exp_a + exp_b);
            let softmax_1 = exp_b / (exp_a + exp_b);

            // For two-class outputs, this assumes class index 1 is the positive class.
            // If your exported model uses the opposite order, this must be changed.
            softmax_1
        } else {
            0.0
        };

        let probability_ai = if POSITIVE_CLASS_IS_AI {
            probability_positive_class
        } else {
            1.0 - probability_positive_class
        };

        Ok(PredictionDebug {
            probability_ai,
            probability_positive_class,
            raw_outputs,
        })
    }
}
