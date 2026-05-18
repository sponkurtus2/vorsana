# UI


## Implementaciones

-----------------------------------------------------------------------------

# LOGIC

    :   Subir proyecto a github

    :   Presentacion en faces de la historia del proyecto

    :   Video presentacion del funcionamiento del proyecto
    


## Implementaciones

    : Metodo de transporte WebSocket (Velocidad) {
        Implementación de comunicación bidireccional full-duplex y asíncrona 
        para la transmisión de chunks de audio crudo en tiempo real, minimizando el overhead de HTTP.
    }

    : Pipeline de Procesamiento de Audio y DSP {
        Conversión de flujos de bytes a arreglos flotantes (f32),
        remuestreo a la tasa nativa del modelo y transformación matemática de la señal 
        (coincidencia exacta con el preprocesamiento de PyTorch).
    }

    : Sesión de Inferencia Concurrente (ort) {
        Carga del modelo ONNX en memoria compartida (Arc) para 
        ejecutar pasadas hacia adelante sin bloquear el hilo de red.
    }

    : Espectrograma de Mel con una tasa de muestreo de 16 kHz, un tamaño de ventana FFT de 400 y un salto de 160

## Librerias

    : ndarray   - Construir tensores de entrada-salida para la IA
    : tokio     - Runtime asincrono
    : axum      - Web server
    : ort       - Procesar archivo onnx
    : rubato    - remuestreo asincrono de audio digital
    : rustfft   - Generar espectogramas
    : serde     - Deserializacion de tipos para estructuras
