use num_complex::Complex;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Target sample rate for the AI model (16kHz).
const TARGET_HZ: usize = 16000;
/// Number of audio channels (Mono).
const CHANNELS: usize = 1;
/// Fixed input chunk size from the frontend (4096 floats).
const INPUT_CHUNK_SIZE: usize = 4096;
/// Spectrogram window size in samples.
pub const WINDOW_SIZE: usize = 400;
/// Hop length (stride) between successive windows.
const HOP_LENGTH: usize = 160;

const MEL_BINS: usize = 128;
const SAMPLE_RATE: f32 = 16000.0;
const MIN_FREQ: f32 = 0.0;
const MAX_FREQ: f32 = 8000.0;

/// AudioProcessor handles PCM conversion, resampling, and spectral analysis.
pub struct AudioProcessor {
    resampler: SincFixedIn<f32>,
    accumulation_buffer: Vec<f32>,
    fft_handle: Arc<dyn Fft<f32>>,
    hann_window: Vec<f32>,
}

impl AudioProcessor {
    /// Initializes the processor, resampler, and pre-calculates the Hann window.
    ///
    /// # Arguments
    /// * `source_hz` - Incoming sample rate from the client.
    pub fn new(source_hz: usize) -> Self {
        let resample_ratio = TARGET_HZ as f64 / source_hz as f64;

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };

        let resampler =
            SincFixedIn::<f32>::new(resample_ratio, 1.0, params, INPUT_CHUNK_SIZE, CHANNELS)
                .expect("Failed to initialize Rubato resampler");

        let mut planner = FftPlanner::new();
        let fft_handler = planner.plan_fft_forward(WINDOW_SIZE);

        // Pre-calculate Hann Window coefficients to reduce runtime overhead.
        let hann_window: Vec<f32> = (0..WINDOW_SIZE)
            .map(|i| {
                0.5 * (1.0
                    - f32::cos(2.0 * std::f32::consts::PI * i as f32 / (WINDOW_SIZE - 1) as f32))
            })
            .collect();

        Self {
            resampler,
            accumulation_buffer: Vec::with_capacity(TARGET_HZ * 2),
            fft_handle: fft_handler,
            hann_window,
        }
    }

    /// Processes binary audio data and returns frequency domain magnitudes.
    ///
    /// # Arguments
    /// * `bytes` - Raw byte buffer from WebSocket.
    /// Processes binary microphone audio data and returns frequency domain magnitudes.
    /// This path keeps a silence gate because live microphone input can contain long empty sections.
    pub fn process_to_frequency_domain(&mut self, bytes: &[u8]) -> Vec<Vec<f32>> {
        let pcm_data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("Invalid byte chunk")))
            .collect();

        self.process_samples_internal(&pcm_data, true)
    }

    /// Processes raw bytes into time-domain sliding windows.
    pub fn process_raw_bytes(&mut self, bytes: &[u8]) -> Vec<Vec<f32>> {
        let pcm_data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("Invalid byte chunk")))
            .collect();

        let input_view = vec![pcm_data];
        let resampled_buffer = self
            .resampler
            .process(&input_view, None)
            .expect("Resampling failed");

        self.accumulation_buffer.extend(&resampled_buffer[0]);

        let mut windows = Vec::new();

        while self.accumulation_buffer.len() >= WINDOW_SIZE {
            let window = self.accumulation_buffer[0..WINDOW_SIZE].to_vec();
            windows.push(window);
            self.accumulation_buffer.drain(0..HOP_LENGTH);
        }

        windows
    }

    pub fn process_file_samples_to_frequency_domain(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut all_windows = Vec::new();

        for chunk in samples.chunks(INPUT_CHUNK_SIZE) {
            let mut fixed_chunk = vec![0.0_f32; INPUT_CHUNK_SIZE];
            fixed_chunk[..chunk.len()].copy_from_slice(chunk);

            let windows = self.process_samples_internal(&fixed_chunk, false);
            all_windows.extend(windows);
        }

        all_windows
    }

    pub fn create_mel_filterbank() -> Vec<Vec<f32>> {
        let mut filters = vec![vec![0.0; WINDOW_SIZE / 2 + 1]; MEL_BINS];

        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);

        let min_mel = hz_to_mel(MIN_FREQ);
        let max_mel = hz_to_mel(MAX_FREQ);

        let mut mel_points = vec![0.0; MEL_BINS + 2];
        for i in 0..MEL_BINS + 2 {
            let mel = min_mel + i as f32 * (max_mel - min_mel) / (MEL_BINS + 1) as f32;
            mel_points[i] = mel_to_hz(mel);
        }

        let bins: Vec<usize> = mel_points
            .iter()
            .map(|&hz| (hz * (WINDOW_SIZE as f32 + 1.0) / SAMPLE_RATE).floor() as usize)
            .collect();

        for i in 0..MEL_BINS {
            for j in bins[i]..bins[i + 1] {
                filters[i][j] = (j - bins[i]) as f32 / (bins[i + 1] - bins[i]) as f32;
            }
            for j in bins[i + 1]..bins[i + 2] {
                filters[i][j] = (bins[i + 2] - j) as f32 / (bins[i + 2] - bins[i + 1]) as f32;
            }
        }
        filters
    }

    /// Applies the Mel-filterbank to FFT power magnitudes and converts to DB scale (Log-Mel).
    pub fn apply_mel_filters(magnitudes: &[f32], filterbank: &[Vec<f32>]) -> Vec<f32> {
        filterbank
            .iter()
            .map(|filter| {
                let power_energy = magnitudes
                    .iter()
                    .zip(filter)
                    .map(|(m, f)| m * f)
                    .sum::<f32>();

                let db_scaled = 10.0 * (power_energy.max(1e-10)).log10();

                // Clamping (Piso dinámico a -80 dB, estándar de PyTorch/Librosa)
                if db_scaled < -80.0 { -80.0 } else { db_scaled }
            })
            .collect()
    }

    fn process_samples_internal(
        &mut self,
        samples: &[f32],
        apply_silence_gate: bool,
    ) -> Vec<Vec<f32>> {
        if samples.is_empty() {
            return Vec::new();
        }

        if apply_silence_gate {
            let square_sum: f32 = samples.iter().map(|&x| x * x).sum();
            let rms = (square_sum / samples.len() as f32).sqrt();

            if rms < 0.01 {
                self.accumulation_buffer.clear();
                return Vec::new();
            }
        }

        let input_view = vec![samples.to_vec()];
        let resampled_buffer = self
            .resampler
            .process(&input_view, None)
            .expect("Resampling operation failed");

        self.accumulation_buffer.extend(&resampled_buffer[0]);

        let mut frequency_windows = Vec::new();

        while self.accumulation_buffer.len() >= WINDOW_SIZE {
            let mut complex_frame: Vec<Complex<f32>> = self.accumulation_buffer[0..WINDOW_SIZE]
                .iter()
                .zip(self.hann_window.iter())
                .map(|(sample, window_coeff)| Complex {
                    re: sample * window_coeff,
                    im: 0.0,
                })
                .collect();

            self.fft_handle.process(&mut complex_frame);

            let magnitudes: Vec<f32> = complex_frame[0..(WINDOW_SIZE / 2 + 1)]
                .iter()
                .map(|c: &Complex<f32>| c.norm_sqr())
                .collect();

            frequency_windows.push(magnitudes);
            self.accumulation_buffer.drain(0..HOP_LENGTH);
        }

        frequency_windows
    }
}
