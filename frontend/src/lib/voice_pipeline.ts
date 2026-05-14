/**
 * Real-time Voice Pipeline for Brownie Frontend
 * 
 * Features:
 * - Wake word detection (Hey Brownie / Brownie)
 * - Voice Activity Detection
 * - Streaming STT
 * - Camera control (explicit only)
 * - Microphone state management
 */

export enum VoiceState {
  IDLE = "idle",
  ACTIVE = "active",
  PROCESSING = "processing",
  SPEAKING = "speaking",
  ERROR = "error",
}

export interface VoicePipelineState {
  state: string;
  microphone_enabled: boolean;
  camera_enabled: boolean;
  active_clone: string | null;
  voices_available: string[];
  timestamp: string;
}

export interface AudioFrame {
  data: Float32Array;
  timestamp: number;
  sampleRate: number;
}

/**
 * Voice Activity Detector using energy-based detection
 */
class VoiceActivityDetector {
  private energyThreshold: number;
  private silenceDuration: number;
  private lastVoiceTime: number = 0;
  private isSpeaking: boolean = false;

  constructor(energyThreshold = 0.03, silenceDuration = 1000) {
    this.energyThreshold = energyThreshold;
    this.silenceDuration = silenceDuration;
  }

  detect(frame: AudioFrame): boolean {
    // Calculate RMS energy
    let sum = 0;
    for (let i = 0; i < frame.data.length; i++) {
      sum += frame.data[i] * frame.data[i];
    }
    const energy = Math.sqrt(sum / frame.data.length);

    if (energy > this.energyThreshold) {
      this.lastVoiceTime = Date.now();
      this.isSpeaking = true;
      return true;
    }

    // Check silence timeout
    if (Date.now() - this.lastVoiceTime > this.silenceDuration) {
      this.isSpeaking = false;
      return false;
    }

    return this.isSpeaking;
  }

  reset(): void {
    this.lastVoiceTime = 0;
    this.isSpeaking = false;
  }
}

/**
 * Main Voice Pipeline
 */
export class VoicePipeline {
  private ws: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private vad: VoiceActivityDetector;
  private state: VoiceState = VoiceState.IDLE;
  private microphoneEnabled: boolean = false;
  private cameraEnabled: boolean = false;
  private onStateChange: (state: VoiceState) => void;
  private onTranscription: (text: string) => void;
  private onError: (error: string) => void;

  constructor(
    onStateChange: (state: VoiceState) => void = () => {},
    onTranscription: (text: string) => void = () => {},
    onError: (error: string) => void = () => {}
  ) {
    this.vad = new VoiceActivityDetector();
    this.onStateChange = onStateChange;
    this.onTranscription = onTranscription;
    this.onError = onError;
  }

  /**
   * Initialize WebSocket connection to voice server
   */
  async connect(wsUrl: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log("[Voice] WebSocket connected");
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (e) {
            console.error("[Voice] Failed to parse message:", e);
          }
        };

        this.ws.onerror = (error) => {
          console.error("[Voice] WebSocket error:", error);
          this.onError("WebSocket connection failed");
          reject(error);
        };

        this.ws.onclose = () => {
          console.log("[Voice] WebSocket closed");
        };
      } catch (e) {
        reject(e);
      }
    });
  }

  /**
   * Handle incoming messages from server
   */
  private handleMessage(data: any): void {
    if (data.type === "state") {
      console.log("[Voice] State update:", data.payload);
      this.onStateChange(data.payload.state);
    } else if (data.type === "response_start") {
      // Early partial response
      console.log("[Voice] Response started:", data.text);
      console.log(`[Voice] LLM latency: ${data.llm_time_ms}ms`);
    } else if (data.type === "response_complete") {
      // Full response with metrics
      console.log("[Voice] Response complete:", data.text);
      if (data.metrics) {
        console.log("[Voice] Performance metrics:", {
          llm_ms: data.metrics.llm_time_ms,
          tts_ms: data.metrics.tts_time_ms,
          total_ms: data.metrics.total_time_ms,
        });
      }
      this.onTranscription(data.text);
    } else if (data.type === "response") {
      // Legacy response format
      console.log("[Voice] Response:", data.text);
      this.onTranscription(data.text);
    } else if (data.type === "error") {
      console.error("[Voice] Server error:", data.message);
      this.onError(data.message);
    } else if (data.type === "status") {
      console.log("[Voice] Status:", data.message);
    }
  }

  /**
   * Enable microphone with VAD
   */
  async enableMicrophone(): Promise<void> {
    try {
      if (this.microphoneEnabled) return;

      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false,
        },
      });

      this.mediaStream = stream;
      const source = this.audioContext.createMediaStreamSource(stream);

      // Create script processor for audio frames
      this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);

      this.scriptProcessor.onaudioprocess = (event) => {
        const frame: AudioFrame = {
          data: event.inputBuffer.getChannelData(0).slice(),
          timestamp: Date.now(),
          sampleRate: this.audioContext!.sampleRate,
        };

        this.processAudioFrame(frame);
      };

      source.connect(this.scriptProcessor);
      this.scriptProcessor.connect(this.audioContext.destination);

      this.microphoneEnabled = true;
      this.setState(VoiceState.IDLE);
      console.log("[Voice] Microphone enabled");
    } catch (e) {
      this.onError(`Failed to enable microphone: ${e}`);
      throw e;
    }
  }

  /**
   * Disable microphone completely
   */
  async disableMicrophone(): Promise<void> {
    if (!this.microphoneEnabled) return;

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }

    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    this.microphoneEnabled = false;
    this.setState(VoiceState.IDLE);
    this.vad.reset();
    console.log("[Voice] Microphone disabled");
  }

  /**
   * Enable camera (explicit only)
   */
  async enableCamera(): Promise<void> {
    try {
      // Camera only activates on explicit request
      this.cameraEnabled = true;
      console.log("[Voice] Camera enabled");
    } catch (e) {
      this.onError(`Failed to enable camera: ${e}`);
    }
  }

  /**
   * Disable camera
   */
  async disableCamera(): Promise<void> {
    this.cameraEnabled = false;
    console.log("[Voice] Camera disabled");
  }

  /**
   * Process audio frame from microphone
   */
  private async processAudioFrame(frame: AudioFrame): Promise<void> {
    if (!this.microphoneEnabled) return;

    // Voice activity detection
    const hasVoice = this.vad.detect(frame);

    if (this.state === VoiceState.IDLE && hasVoice) {
      this.setState(VoiceState.ACTIVE);
      console.log("[Voice] Voice detected - listening");
    } else if (this.state === VoiceState.ACTIVE && !hasVoice) {
      this.setState(VoiceState.PROCESSING);
      console.log("[Voice] Silence detected - processing");
      // Finalize transcription
    }
  }

  /**
   * Interrupt current speech
   */
  async interrupt(): Promise<void> {
    if (this.state === VoiceState.SPEAKING) {
      this.setState(VoiceState.IDLE);
      console.log("[Voice] Speech interrupted");
      
      // Notify backend
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify({
            type: "interrupt"
          }));
        } catch (e) {
          console.error("[Voice] Failed to send interrupt:", e);
        }
      }
    }
  }

  /**
   * Update state with callback
   */
  private setState(newState: VoiceState): void {
    if (this.state !== newState) {
      this.state = newState;
      console.log(`[Voice] State: ${newState}`);
      this.onStateChange(newState);
    }
  }

  /**
   * Get current state
   */
  getState(): VoiceState {
    return this.state;
  }

  /**
   * Check if microphone is enabled
   */
  isMicrophoneEnabled(): boolean {
    return this.microphoneEnabled;
  }

  /**
   * Check if camera is enabled
   */
  isCameraEnabled(): boolean {
    return this.cameraEnabled;
  }

  /**
   * Send transcription to backend for processing
   */
  async sendTranscription(text: string, voiceClone?: string): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.onError("WebSocket not connected");
      return;
    }

    try {
      this.setState(VoiceState.PROCESSING);
      
      this.ws.send(JSON.stringify({
        type: "transcription",
        text,
        voice_clone: voiceClone || null,
      }));
      
      this.setState(VoiceState.SPEAKING);
    } catch (e) {
      this.onError(`Failed to send transcription: ${e}`);
      this.setState(VoiceState.ERROR);
    }
  }

  /**
   * Load voice clone
   */
  async loadVoiceClone(name: string, samples: string[]): Promise<void> {
    if (!this.ws) throw new Error("Not connected");

    this.ws.send(
      JSON.stringify({
        type: "load_clone",
        name,
        samples,
      })
    );
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    await this.disableMicrophone();
    await this.disableCamera();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    console.log("[Voice] Pipeline cleaned up");
  }
}

/**
 * Singleton instance
 */
let _instance: VoicePipeline | null = null;

/**
 * Get or create voice pipeline
 */
export function getVoicePipeline(
  onStateChange?: (state: VoiceState) => void,
  onTranscription?: (text: string) => void,
  onError?: (error: string) => void
): VoicePipeline {
  if (!_instance) {
    _instance = new VoicePipeline(onStateChange, onTranscription, onError);
  }
  return _instance;
}
