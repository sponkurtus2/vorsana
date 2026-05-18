/**
 * Interface defining callback hooks for UI updates.
 */
export interface AudioClientCallbacks {
	onLog: (message: string) => void;
	onStateChange: (connected: boolean, recording: boolean) => void;
}

/**
 * Manages WebSocket communication and microphone audio streaming.
 */
export class AudioStreamClient {
	private ws: WebSocket | null = null;
	private audioContext: AudioContext | null = null;
	private scriptProcessor: ScriptProcessorNode | null = null;
	private mediaStreamSource: MediaStreamAudioSourceNode | null = null;
	private audioStream: MediaStream | null = null;
	private isRecording: boolean = false;

	// Buffer size of 4096 frames (Aprox 85-92ms of latency)
	private readonly bufferSize = 4096;

	constructor(
		private readonly wsUrl: string,
		private readonly callbacks: AudioClientCallbacks
	) { }

	/**
	 * Initializes the WebSocket connection to the backend server.
	 */
	public connect(): void {
		try {
			this.ws = new WebSocket(this.wsUrl);
			this.ws.binaryType = "arraybuffer";

			this.ws.onopen = () => {
				this.callbacks.onLog('WebSocket connected successfully.');
				this.updateState();
			};

			this.ws.onclose = () => {
				this.callbacks.onLog('WebSocket connection closed.');
				this.stopRecording();
				this.updateState();
			};

			this.ws.onerror = (error) => {
				this.callbacks.onLog('WebSocket encountered an error.');
				console.error('[WS Error]', error);
			};

			this.ws.onmessage = (event) => {
				this.callbacks.onLog(`Server: ${event.data}`);
			};
		} catch (error) {
			this.callbacks.onLog('Failed to instantiate WebSocket connection.');
			console.error('[Connection Error]', error);
		}
	}

	/**
	 * Requests microphone access and begins streaming binary chunks via WebSocket.
	 */
	public async startRecording(): Promise<void> {
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
			this.callbacks.onLog('Error: WebSocket is not connected.');
			return;
		}

		try {
			this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
			this.callbacks.onLog('Microphone access granted.');

			// Initializes web audio API
			this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
			this.mediaStreamSource = this.audioContext.createMediaStreamSource(this.audioStream);

			// Create a processing node (1 input channel, 1 output channel)
			this.scriptProcessor = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);

			this.scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
				if (!this.isRecording || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

				// Extract raw Float32 mono channel data
				const inputBuffer = audioProcessingEvent.inputBuffer;
				const inputData = inputBuffer.getChannelData(0); // Float32Array

				// Send the raw byte buffer down the socket channel
				this.ws.send(inputData.buffer);
			};

			// Connect nodes
			this.mediaStreamSource.connect(this.scriptProcessor);
			this.scriptProcessor.connect(this.audioContext.destination);

			this.isRecording = true;
			this.callbacks.onLog('Audio streaming started.');
			this.updateState();

		} catch (error) {
			this.callbacks.onLog('Microphone access denied or initialization failed.');
			console.error('[Media Error]', error);
		}
	}

	/**
	 * Stops the audio capture and terminates the media tracks.
	 */
	public stopRecording(): void {
		if (this.scriptProcessor && this.mediaStreamSource) {
			this.scriptProcessor.disconnect();
			this.mediaStreamSource.disconnect();
			this.scriptProcessor = null;
			this.mediaStreamSource = null;
		}

		if (this.audioContext && this.audioContext.state !== 'closed') {
			this.audioContext.close();
			this.audioContext = null;
		}


		if (this.audioStream) {
			this.audioStream.getTracks().forEach(track => track.stop());
			this.audioStream = null;
		}

		if (this.isRecording) {
			this.isRecording = false;
			this.callbacks.onLog('Audio streaming stopped.');
			this.updateState();
		}
	}

	/**
	 * Closes the WebSocket connection gracefully.
	 */
	public disconnect(): void {
		this.stopRecording();
		if (this.ws) {
			this.ws.close();
			this.ws = null;
		}
	}

	/**
	 * Triggers the state change callback to update UI elements.
	 */
	private updateState(): void {
		const isConnected = this.ws !== null && this.ws.readyState === WebSocket.OPEN;
		this.callbacks.onStateChange(isConnected, this.isRecording);
	}
}
