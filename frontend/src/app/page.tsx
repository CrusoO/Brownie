"use client";

import {
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Activity,
  BookOpen,
  Camera,
  CheckCircle2,
  Circle,
  Mic,
  MicOff,
  Play,
  Plus,
  RotateCcw,
  Send,
  ShieldCheck,
  ShieldOff,
  Sparkles,
  Trash2,
  Volume2,
  VolumeX,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { VoicePipeline, VoiceState, getVoicePipeline } from "@/lib/voice_pipeline";

type TraceEvent = {
  id: string;
  type: string;
  message: string;
  data?: Record<string, unknown>;
  timestamp: string;
};

type ChatResponse = {
  run_id: string;
  session_id: string;
  response: string;
  route: "talk" | "tool" | string;
  trace: TraceEvent[];
  memory_ids: string[];
};

type Message = {
  id: string;
  role: "user" | "brownie";
  content: string;
  timestamp: string;
};

type Workflow = {
  id: string;
  session_id: string;
  name: string;
  trigger: string;
  steps: string[];
  created_at: string;
  updated_at: string;
  run_count: number;
};

type SocketMessage =
  | { type: "trace"; event: TraceEvent }
  | { type: "final"; response: ChatResponse };

type BrowserSpeechRecognition = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start: () => void;
  stop: () => void;
  onresult: ((event: SpeechRecognitionResultEvent) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
};

type SpeechRecognitionResultEvent = {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
};

type SpeechRecognitionWindow = Window & {
  SpeechRecognition?: new () => BrowserSpeechRecognition;
  webkitSpeechRecognition?: new () => BrowserSpeechRecognition;
};

type FaceProfile = {
  vector: number[];
  createdAt: string;
};

type FaceState = "not_enrolled" | "camera" | "locked" | "verified" | "error";

const configuredWsUrl = process.env.NEXT_PUBLIC_BACKEND_WS_URL;
const configuredHttpUrl = process.env.NEXT_PUBLIC_BACKEND_HTTP_URL;
const sessionId = "brownie-web";
const faceStorageKey = "brownie.face.profile";

function getWsUrl() {
  if (configuredWsUrl) {
    return configuredWsUrl;
  }

  const host =
    typeof window === "undefined" ? "localhost" : window.location.hostname || "localhost";
  const frontendPort = typeof window === "undefined" ? "" : window.location.port;
  const backendPort = frontendPort === "3002" ? "8010" : "8000";

  return `ws://${host}:${backendPort}/ws/chat`;
}

function getHttpUrl() {
  if (configuredHttpUrl) {
    return configuredHttpUrl;
  }

  const host =
    typeof window === "undefined" ? "localhost" : window.location.hostname || "localhost";
  const frontendPort = typeof window === "undefined" ? "" : window.location.port;
  const backendPort = frontendPort === "3002" ? "8010" : "8000";

  return `http://${host}:${backendPort}`;
}

function getVoiceWsUrl() {
  const host =
    typeof window === "undefined" ? "localhost" : window.location.hostname || "localhost";
  const frontendPort = typeof window === "undefined" ? "" : window.location.port;
  const backendPort = frontendPort === "3002" ? "8010" : "8000";

  return `ws://${host}:${backendPort}/ws/voice`;
}

function nowLabel(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function uniqueEvents(events: TraceEvent[]) {
  const seen = new Set<string>();
  return events.filter((event) => {
    if (seen.has(event.id)) {
      return false;
    }
    seen.add(event.id);
    return true;
  });
}

function cosineSimilarity(left: number[], right: number[]) {
  const length = Math.min(left.length, right.length);
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;

  for (let index = 0; index < length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }

  if (!leftNorm || !rightNorm) {
    return 0;
  }

  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

export default function Home() {
  const httpUrl = useMemo(() => getHttpUrl(), []);
  const socketRef = useRef<WebSocket | null>(null);
  const activeRequestRef = useRef<{ startedAt: number } | null>(null);
  const pendingMessageRef = useRef<string | null>(null);
  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const voicePipelineRef = useRef<VoicePipeline | null>(null);

  const [input, setInput] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState<"ready" | "connecting" | "online" | "error">(
    "ready",
  );
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "initial",
      role: "brownie",
      content: "Ready.",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [voiceReplies, setVoiceReplies] = useState(true);
  const [isListening, setIsListening] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("Idle");
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  const [faceState, setFaceState] = useState<FaceState>(() =>
    typeof window !== "undefined" && window.localStorage.getItem(faceStorageKey)
      ? "locked"
      : "not_enrolled",
  );
  const [cameraOn, setCameraOn] = useState(false);
  const [faceScore, setFaceScore] = useState<number | null>(null);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [workflowName, setWorkflowName] = useState("");
  const [workflowTrigger, setWorkflowTrigger] = useState("");
  const [workflowSteps, setWorkflowSteps] = useState("");
  const [showWorkflows, setShowWorkflows] = useState(false);
  const [responseTimeMs, setResponseTimeMs] = useState(0);
  const [lastVoiceTime, setLastVoiceTime] = useState(0);
  const [lastFaceVerify, setLastFaceVerify] = useState(0);
  const [hasGreeted, setHasGreeted] = useState(false);
  const voiceRepliesRef = useRef(true);
  const [voicePipelineState, setVoicePipelineState] = useState<string>("idle");

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    setIsListening(false);
    setVoiceStatus("Idle");
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraOn(false);
  }, []);

  const appendUserMessage = useCallback((content: string) => {
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        content,
        timestamp: new Date().toISOString(),
      },
    ]);
  }, []);

  const appendAssistantMessage = useCallback((content: string) => {
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "brownie",
        content,
        timestamp: new Date().toISOString(),
      },
    ]);
  }, []);

  const loadWorkflows = useCallback(async () => {
    const response = await fetch(
      `${httpUrl}/workflows?session_id=${encodeURIComponent(sessionId)}`,
    );
    if (!response.ok) {
      return;
    }
    const payload = (await response.json()) as Workflow[];
    setWorkflows(payload);
  }, [httpUrl]);

  // Stop listening when voice is manually toggled off
  useEffect(() => {
    voiceRepliesRef.current = voiceReplies;
    if (!voiceReplies) {
      stopListening();
    }
  }, [voiceReplies, stopListening]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadWorkflows();
      // Auto-start face verification for greeting
      if (typeof window !== "undefined" && window.localStorage.getItem(faceStorageKey)) {
        void verifyFace();
      }
    }, 0);

    return () => {
      window.clearTimeout(timer);
      stopCamera();
      stopListening();
      window.speechSynthesis?.cancel();
    };
  }, [loadWorkflows, stopCamera, stopListening]);

  // Initialize voice pipeline once
  useEffect(() => {
    const initVoicePipeline = async () => {
      try {
        const pipeline = getVoicePipeline(
          (state) => setVoicePipelineState(state.toString()),
          (text) => {
            // Only reflect voice pipeline output in the UI input.
            // Sending it as a new query creates duplicate round-trips.
            setInput(text);
          },
          (error) => {
            console.error("[Voice Error]", error);
            setVoiceStatus(`Error: ${error}`);
          }
        );

        voicePipelineRef.current = pipeline;
        const wsUrl = getVoiceWsUrl();
        await pipeline.connect(wsUrl);
        if (voiceRepliesRef.current) {
          await pipeline.enableMicrophone();
        }
      } catch (e) {
        console.error("[Voice Pipeline] Failed to initialize:", e);
      }
    };

    initVoicePipeline();

    return () => {
      // Cleanup on unmount
      if (voicePipelineRef.current) {
        void voicePipelineRef.current.cleanup();
        voicePipelineRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const syncMicrophone = async () => {
      if (!voicePipelineRef.current) {
        return;
      }
      if (voiceReplies) {
        await voicePipelineRef.current.enableMicrophone();
      } else {
        await voicePipelineRef.current.disableMicrophone();
      }
    };
    void syncMicrophone();
  }, [voiceReplies]);

  useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) {
      return;
    }

    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      if (!voices.length) {
        return;
      }

      const preferredVoice = voices.find((voice) =>
        /en-US/i.test(voice.lang) &&
        /(Google|Microsoft|Alex|Samantha|Alloy|Olivia|Aria)/i.test(voice.name),
      );

      setSelectedVoice(preferredVoice ?? voices[0]);
    };

    loadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", loadVoices);
    };
  }, []);

  // Auto-start voice listening when page loads
  useEffect(() => {
    const voiceTimer = window.setTimeout(() => {
      if (voiceRepliesRef.current && !isListening && !isRunning) {
        startListening();
      }
    }, 3000);

    return () => {
      window.clearTimeout(voiceTimer);
    };
  }, [isListening, isRunning]);

  const statusText = useMemo(() => {
    if (isRunning) {
      return "Running";
    }
    if (status === "online") {
      return "Online";
    }
    if (status === "connecting") {
      return "Connecting";
    }
    if (status === "error") {
      return "Offline";
    }
    return "Ready";
  }, [isRunning, status]);

  const faceText = useMemo(() => {
    if (faceState === "verified") {
      return "Verified";
    }
    if (faceState === "locked") {
      return "Locked";
    }
    if (faceState === "camera") {
      return "Camera";
    }
    if (faceState === "error") {
      return "Error";
    }
    return "Not Enrolled";
  }, [faceState]);

  function speak(text: string) {
    if (!voiceReplies || typeof window === "undefined" || !window.speechSynthesis) {
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    if (selectedVoice) {
      utterance.voice = selectedVoice;
      utterance.lang = selectedVoice.lang;
    }
    utterance.rate = 0.96;
    utterance.pitch = 1.05;
    utterance.volume = 1;
    window.speechSynthesis.speak(utterance);
  }

  function appendFinal(response: ChatResponse) {
    setTrace((current) => uniqueEvents([...current, ...response.trace]));
    setMessages((current) => [
      ...current,
      {
        id: response.run_id,
        role: "brownie",
        content: response.response,
        timestamp: new Date().toISOString(),
      },
    ]);
    speak(response.response);
  }

  function connectChatSocket() {
    const current = socketRef.current;
    if (
      current &&
      (current.readyState === WebSocket.OPEN || current.readyState === WebSocket.CONNECTING)
    ) {
      return current;
    }

    setStatus("connecting");
    const socket = new WebSocket(getWsUrl());
    socketRef.current = socket;

    socket.onopen = () => {
      setStatus("online");
      if (pendingMessageRef.current) {
        socket.send(JSON.stringify({ message: pendingMessageRef.current, session_id: sessionId }));
        pendingMessageRef.current = null;
      }
    };

    socket.onmessage = (payload) => {
      const body = JSON.parse(payload.data) as SocketMessage;
      if (body.type === "trace") {
        setTrace((currentTrace) => uniqueEvents([...currentTrace, body.event]));
      }

      if (body.type === "final") {
        if (activeRequestRef.current) {
          setResponseTimeMs(Date.now() - activeRequestRef.current.startedAt);
          activeRequestRef.current = null;
        }
        appendFinal(body.response);
        setIsRunning(false);
        setStatus("online");
        if (voiceRepliesRef.current) {
          setTimeout(() => startListening(), 500);
        }
      }
    };

    socket.onerror = () => {
      setIsRunning(false);
      setStatus("error");
      appendAssistantMessage("Backend connection failed.");
    };

    socket.onclose = () => {
      socketRef.current = null;
    };

    return socket;
  }

  async function controlVoiceSensor(action: "enable" | "disable", sensor: "microphone" | "camera") {
    try {
      await fetch(`${httpUrl}/voice/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, sensor }),
      });
    } catch {
      // Keep local controls responsive even if backend control endpoint is unavailable.
    }
  }

  async function handleLocalCommand(rawMessage: string): Promise<boolean> {
    const transcript = rawMessage.toLowerCase().trim();
    if (!transcript) {
      return true;
    }

    const mentionsVoice = /(voice assistant|voice|microphone|mic)/.test(transcript);
    const mentionsCamera = /(webcam|camera|cam)/.test(transcript);
    const wantsDisable = /\b(disable|turn off|stop)\b/.test(transcript);
    const wantsEnable = /\b(enable|turn on|start)\b/.test(transcript);
    const mentionsBoth = /\b(both|them|all)\b/.test(transcript);

    if (wantsDisable && ((mentionsVoice && mentionsCamera) || mentionsBoth)) {
      setVoiceReplies(false);
      stopListening();
      stopCamera();
      setFaceState("locked");
      if (voicePipelineRef.current) {
        await voicePipelineRef.current.disableMicrophone();
        await voicePipelineRef.current.disableCamera();
      }
      await Promise.all([
        controlVoiceSensor("disable", "microphone"),
        controlVoiceSensor("disable", "camera"),
      ]);
      appendAssistantMessage("Voice assistant and webcam are both disabled.");
      return true;
    }

    if (wantsEnable && ((mentionsVoice && mentionsCamera) || mentionsBoth)) {
      setVoiceReplies(true);
      await startCamera();
      if (voicePipelineRef.current) {
        await voicePipelineRef.current.enableMicrophone();
        await voicePipelineRef.current.enableCamera();
      }
      await Promise.all([
        controlVoiceSensor("enable", "microphone"),
        controlVoiceSensor("enable", "camera"),
      ]);
      appendAssistantMessage("Voice assistant and webcam are both enabled.");
      return true;
    }

    if (wantsDisable && mentionsVoice) {
      setVoiceReplies(false);
      stopListening();
      if (voicePipelineRef.current) {
        await voicePipelineRef.current.disableMicrophone();
      }
      await controlVoiceSensor("disable", "microphone");
      appendAssistantMessage("Voice assistant disabled. I will only respond in text.");
      return true;
    }

    if (wantsEnable && mentionsVoice) {
      setVoiceReplies(true);
      await controlVoiceSensor("enable", "microphone");
      if (voicePipelineRef.current) {
        await voicePipelineRef.current.enableMicrophone();
      }
      appendAssistantMessage("Voice assistant enabled.");
      return true;
    }

    if (wantsDisable && mentionsCamera) {
      stopCamera();
      setFaceState("locked");
      if (voicePipelineRef.current) {
        await voicePipelineRef.current.disableCamera();
      }
      await controlVoiceSensor("disable", "camera");
      appendAssistantMessage("Webcam disabled.");
      return true;
    }

    if (wantsEnable && mentionsCamera) {
      await startCamera();
      if (voicePipelineRef.current) {
        await voicePipelineRef.current.enableCamera();
      }
      await controlVoiceSensor("enable", "camera");
      appendAssistantMessage("Webcam enabled.");
      return true;
    }

    return false;
  }

  useEffect(() => {
    const socket = connectChatSocket();
    return () => {
      if (
        socket.readyState === WebSocket.OPEN ||
        socket.readyState === WebSocket.CONNECTING
      ) {
        socket.close();
      }
    };
  }, []);

  function sendMessage(message: string) {
    const cleaned = message.trim();
    if (!cleaned || isRunning) {
      return;
    }

    setInput("");
    setTrace([]);
    setIsRunning(true);
    setStatus("connecting");
    setResponseTimeMs(0);
    activeRequestRef.current = { startedAt: Date.now() };
    appendUserMessage(cleaned);
    pendingMessageRef.current = cleaned;

    const socket = connectChatSocket();
    if (socket.readyState === WebSocket.OPEN && pendingMessageRef.current) {
      socket.send(JSON.stringify({ message: pendingMessageRef.current, session_id: sessionId }));
      pendingMessageRef.current = null;
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleaned = input.trim();
    if (!cleaned || isRunning) {
      return;
    }
    setInput("");
    const handled = await handleLocalCommand(cleaned);
    if (handled) {
      appendUserMessage(cleaned);
      return;
    }
    sendMessage(cleaned);
  }

  function reset() {
    socketRef.current?.close();
    socketRef.current = null;
    activeRequestRef.current = null;
    pendingMessageRef.current = null;
    setInput("");
    setIsRunning(false);
    setStatus("ready");
    setTrace([]);
    window.speechSynthesis?.cancel();
    setMessages([
      {
        id: "initial",
        role: "brownie",
        content: "Ready.",
        timestamp: new Date().toISOString(),
      },
    ]);
  }

  function startListening() {
    // Debounce voice recognition (minimum 300ms between commands)
    const now = Date.now();
    if (now - lastVoiceTime < 300) {
      return;
    }

    const speechWindow = window as SpeechRecognitionWindow;
    const SpeechRecognition =
      speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setVoiceStatus("Unavailable");
      return;
    }

    recognitionRef.current?.stop();
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.onresult = (event) => {
      const transcript = event.results[0]?.[0]?.transcript?.trim().toLowerCase() ?? "";
      setLastVoiceTime(Date.now());
      setVoiceStatus(transcript ? "Captured" : "Idle");
      if (transcript) {
        // Voice commands for UI interaction
        if (transcript.includes("show") && transcript.includes("workflow")) {
          setShowWorkflows(true);
          speak("Showing workflows");
          return;
        }
        if (transcript.includes("hide") && transcript.includes("workflow")) {
          setShowWorkflows(false);
          speak("Hiding workflows");
          return;
        }
        void handleLocalCommand(transcript).then((handled) => {
          if (handled) {
            appendUserMessage(transcript);
            return;
          }
          setInput(transcript);
          sendMessage(transcript);
        });
      }
    };
    recognition.onerror = () => {
      setVoiceStatus("Error");
      setIsListening(false);
    };
    recognition.onend = () => {
      setIsListening(false);
      // Auto-restart listening only if voice is still enabled
      setTimeout(() => {
        if (voiceRepliesRef.current && !isRunning) {
          startListening();
        }
      }, 500);
    };
    recognitionRef.current = recognition;
    setVoiceStatus("Listening");
    setIsListening(true);
    recognition.start();
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setFaceState("error");
      return false;
    }

    if (!streamRef.current) {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    }

    setCameraOn(true);
    if (faceState !== "verified") {
      setFaceState("camera");
    }
    return true;
  }

  function captureFaceVector() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      return null;
    }

    const sourceSize = Math.min(video.videoWidth, video.videoHeight);
    if (!sourceSize) {
      return null;
    }

    const sourceX = Math.floor((video.videoWidth - sourceSize) / 2);
    const sourceY = Math.floor((video.videoHeight - sourceSize) / 2);
    const size = 32;
    canvas.width = size;
    canvas.height = size;

    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      return null;
    }

    context.drawImage(video, sourceX, sourceY, sourceSize, sourceSize, 0, 0, size, size);
    const pixels = context.getImageData(0, 0, size, size).data;
    const buckets: number[] = [];
    const cells = 8;
    const cellSize = size / cells;

    for (let cellY = 0; cellY < cells; cellY += 1) {
      for (let cellX = 0; cellX < cells; cellX += 1) {
        let total = 0;
        for (let y = 0; y < cellSize; y += 1) {
          for (let x = 0; x < cellSize; x += 1) {
            const px = Math.floor(cellX * cellSize + x);
            const py = Math.floor(cellY * cellSize + y);
            const offset = (py * size + px) * 4;
            total += pixels[offset] * 0.299 + pixels[offset + 1] * 0.587 + pixels[offset + 2] * 0.114;
          }
        }
        buckets.push(total / (cellSize * cellSize * 255));
      }
    }

    const mean = buckets.reduce((sum, value) => sum + value, 0) / buckets.length;
    const variance =
      buckets.reduce((sum, value) => sum + (value - mean) ** 2, 0) / buckets.length;
    const deviation = Math.sqrt(variance) || 1;
    return buckets.map((value) => (value - mean) / deviation);
  }

  async function enrollFace() {
    try {
      const ready = await startCamera();
      if (!ready) {
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 250));
      const vector = captureFaceVector();
      if (!vector) {
        setFaceState("error");
        return;
      }
      const profile: FaceProfile = {
        vector,
        createdAt: new Date().toISOString(),
      };
      window.localStorage.setItem(faceStorageKey, JSON.stringify(profile));
      setFaceScore(1);
      setFaceState("verified");
    } catch {
      setFaceState("error");
    }
  }

  async function verifyFace() {
    try {
      const rawProfile = window.localStorage.getItem(faceStorageKey);
      if (!rawProfile) {
        await enrollFace();
        return;
      }

      // Debounce face verification (max once per 30 seconds)
      const now = Date.now();
      if (now - lastFaceVerify < 30000) {
        return;
      }
      setLastFaceVerify(now);

      const ready = await startCamera();
      if (!ready) {
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 250));
      const profile = JSON.parse(rawProfile) as FaceProfile;
      const vector = captureFaceVector();
      if (!vector) {
        setFaceState("error");
        return;
      }

      const score = cosineSimilarity(profile.vector, vector);
      setFaceScore(score);
      const isVerified = score >= 0.78;
      setFaceState(isVerified ? "verified" : "locked");

      // Friendly greeting on first verification
      if (isVerified && !hasGreeted) {
        setHasGreeted(true);
        speak("Hi there! How can I help you today?");
        setMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            role: "brownie",
            content: "👋 Hi there! How can I help you today?",
            timestamp: new Date().toISOString(),
          },
        ]);
        // Auto-start listening after greeting
        setTimeout(() => {
          if (voiceReplies) {
            startListening();
          }
        }, 2000);
      }
    } catch {
      setFaceState("error");
    }
  }

  function clearFace() {
    window.localStorage.removeItem(faceStorageKey);
    setFaceScore(null);
    setFaceState("not_enrolled");
  }

  async function createWorkflow(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const steps = workflowSteps
      .split(/\r?\n/)
      .map((step) => step.trim())
      .filter(Boolean);

    if (!workflowName.trim() || !workflowTrigger.trim() || steps.length === 0) {
      return;
    }

    const response = await fetch(`${httpUrl}/workflows`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        name: workflowName.trim(),
        trigger: workflowTrigger.trim(),
        steps,
      }),
    });

    if (!response.ok) {
      return;
    }

    setWorkflowName("");
    setWorkflowTrigger("");
    setWorkflowSteps("");
    await loadWorkflows();
  }

  async function deleteWorkflow(workflow: Workflow) {
    await fetch(
      `${httpUrl}/workflows/${workflow.id}?session_id=${encodeURIComponent(sessionId)}`,
      { method: "DELETE" },
    );
    await loadWorkflows();
  }

  async function runWorkflow(workflow: Workflow) {
    await fetch(
      `${httpUrl}/workflows/${workflow.id}/run?session_id=${encodeURIComponent(sessionId)}`,
      { method: "POST" },
    );
    await loadWorkflows();
    sendMessage(workflow.trigger);
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="flex h-12 items-center justify-between border-b border-gray-700 bg-gray-900 px-4 shadow-sm">
        <div className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-md bg-blue-600 text-white text-xs font-semibold">
            B
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight">Brownie</h1>
            <p className="text-xs text-gray-400">AI Assistant</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => setShowWorkflows(!showWorkflows)}
            title="Toggle workflows"
            aria-label="Toggle workflows"
          >
            <BookOpen className="h-3 w-3" />
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => setVoiceReplies((current) => !current)}
            title={voiceReplies ? "Mute voice" : "Enable voice"}
            aria-label={voiceReplies ? "Mute voice" : "Enable voice"}
          >
            {voiceReplies ? <Volume2 className="h-3 w-3" /> : <VolumeX className="h-3 w-3" />}
          </Button>
          <Button
            type="button"
            variant={faceState === "verified" ? "default" : "secondary"}
            size="sm"
            onClick={verifyFace}
            title="Verify face"
            aria-label="Verify face"
          >
            {faceState === "verified" ? <ShieldCheck className="h-3 w-3" /> : <ShieldOff className="h-3 w-3" />}
          </Button>
        </div>
      </header>

      <main className={cn(
        "mx-auto grid max-w-7xl min-h-[calc(100vh-3rem)] gap-3 p-3",
        showWorkflows ? "xl:grid-cols-[minmax(0,1fr)_320px]" : "grid-cols-1"
      )}>
        <Card className="flex min-h-[600px] flex-col overflow-hidden">
          <ScrollArea className="min-h-0 flex-1">
            <div className="flex flex-col gap-2 p-3">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex flex-col gap-1",
                    message.role === "user" ? "items-end" : "items-start",
                  )}
                >
                  <div
                    className={cn(
                      "max-w-[min(720px,85%)] rounded-md px-3 py-2 text-xs leading-5",
                      message.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-gray-800 text-gray-100 border border-gray-700",
                    )}
                  >
                    <p className="whitespace-pre-wrap break-words">{message.content}</p>
                  </div>
                  <span className="px-1 text-[10px] text-gray-500">
                    {nowLabel(message.timestamp)}
                  </span>
                </div>
              ))}
            </div>
          </ScrollArea>

          <form
            onSubmit={submit}
            className="flex items-center gap-2 border-t border-gray-700 bg-gray-900 p-3"
          >
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={reset}
              title="Reset"
              aria-label="Reset"
            >
              <RotateCcw className="h-3 w-3" />
            </Button>
            <Button
              type="button"
              variant={isListening ? "default" : "secondary"}
              size="sm"
              onClick={startListening}
              title="Listen"
              aria-label="Listen"
              disabled={isRunning}
            >
              {isListening ? <MicOff className="h-3 w-3" /> : <Mic className="h-3 w-3" />}
            </Button>
            <Input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Message Brownie"
              disabled={isRunning}
              className="flex-1"
            />
            <Button
              type="submit"
              size="sm"
              title="Send"
              aria-label="Send"
              disabled={!input.trim() || isRunning}
            >
              {isRunning ? (
                <div className="h-3 w-3 animate-spin rounded-full border border-white/30 border-t-white" />
              ) : (
                <Send className="h-3 w-3" />
              )}
            </Button>
          </form>
        </Card>

        {showWorkflows && (
        <aside className="grid min-h-[600px] grid-rows-[auto_auto_auto_minmax(200px,1fr)] gap-3">
          <div className="grid grid-cols-2 gap-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Status</CardTitle>
                <Circle
                  className={cn(
                    "size-3 fill-current",
                    status === "error" ? "text-red-500" : "text-green-500",
                  )}
                />
              </CardHeader>
              <CardContent className="flex items-center justify-between gap-3 pt-0">
                <div>
                  <p className="text-sm font-semibold tracking-normal">{statusText}</p>
                  <p className="text-xs text-gray-500">
                    {responseTimeMs > 0 ? `${responseTimeMs}ms` : "Backend connection"}
                  </p>
                </div>
                <Activity className="size-4 text-gray-500" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Voice</CardTitle>
                {isListening ? (
                  <Mic className="size-3 text-gray-100" />
                ) : (
                  <Volume2 className="size-3 text-gray-500" />
                )}
              </CardHeader>
              <CardContent className="space-y-2 pt-0">
                <div>
                  <p className="text-sm font-semibold tracking-normal">{voiceStatus}</p>
                  <p className="text-xs text-gray-500">
                    {selectedVoice?.name ?? "System voice"}
                  </p>
                </div>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={startListening}
                  title="Listen"
                  aria-label="Listen"
                  disabled={isRunning}
                >
                  {isListening ? <MicOff className="h-3 w-3" /> : <Mic className="h-3 w-3" />}
                </Button>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Face</CardTitle>
              {faceState === "verified" ? (
                <CheckCircle2 className="size-3 text-green-500" />
              ) : (
                <Camera className="size-3 text-gray-500" />
              )}
            </CardHeader>
            <CardContent className="grid gap-2 pt-0">
              <video
                ref={videoRef}
                className={cn(
                  "h-20 w-full rounded-md border border-gray-600 bg-gray-800 object-cover",
                  !cameraOn && "hidden",
                )}
                playsInline
                muted
              />
              <canvas ref={canvasRef} className="hidden" />
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-semibold tracking-normal">{faceText}</span>
                <span className="text-xs text-gray-500">
                  {faceScore === null ? "" : `${Math.round(faceScore * 100)}%`}
                </span>
              </div>
              <div className="grid grid-cols-4 gap-1">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => void startCamera()}
                  title="Camera"
                  aria-label="Camera"
                >
                  <Camera className="h-3 w-3" />
                </Button>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => void enrollFace()}
                  title="Enroll"
                  aria-label="Enroll"
                >
                  <ShieldCheck className="h-3 w-3" />
                </Button>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => void verifyFace()}
                  title="Verify"
                  aria-label="Verify"
                >
                  <CheckCircle2 className="h-3 w-3" />
                </Button>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={clearFace}
                  title="Clear"
                  aria-label="Clear"
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Teach</CardTitle>
              <BookOpen className="size-3 text-gray-500" />
            </CardHeader>
            <CardContent className="grid gap-2 pt-0">
              <form onSubmit={createWorkflow} className="grid gap-2">
                <Input
                  value={workflowName}
                  onChange={(event) => setWorkflowName(event.target.value)}
                  placeholder="Workflow name"
                />
                <Input
                  value={workflowTrigger}
                  onChange={(event) => setWorkflowTrigger(event.target.value)}
                  placeholder="Trigger phrase"
                />
                <textarea
                  value={workflowSteps}
                  onChange={(event) => setWorkflowSteps(event.target.value)}
                  placeholder="One step per line"
                  className="min-h-20 resize-none rounded-md border border-gray-600 bg-gray-800 px-3 py-2 text-xs text-gray-100 outline-none transition focus:border-blue-500 focus:ring-1 focus:ring-blue-500 placeholder:text-gray-500"
                />
                <Button
                  type="submit"
                  size="sm"
                  disabled={!workflowName || !workflowTrigger || !workflowSteps}
                >
                  <Plus className="h-3 w-3" />
                  Save
                </Button>
              </form>
              <div className="grid gap-2">
                {workflows.slice(0, 4).map((workflow) => (
                  <div
                    key={workflow.id}
                    className="grid gap-2 rounded-md border border-gray-700 bg-gray-800 p-3"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate text-xs font-semibold">{workflow.name}</p>
                        <p className="truncate text-xs text-gray-500">
                          {workflow.trigger}
                        </p>
                      </div>
                      <div className="flex shrink-0 gap-1">
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => void runWorkflow(workflow)}
                          title="Run"
                          aria-label="Run"
                          disabled={isRunning}
                        >
                          <Play className="h-3 w-3" />
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => void deleteWorkflow(workflow)}
                          title="Delete"
                          aria-label="Delete"
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="flex min-h-0 flex-col">
            <CardHeader>
              <CardTitle>Reasoning</CardTitle>
              <Sparkles className="size-3 text-gray-500" />
            </CardHeader>
            <ScrollArea className="min-h-0 flex-1">
              <div className="flex flex-col gap-2 px-3 pb-3">
                {trace.length === 0 ? (
                  <div className="rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-xs text-gray-500">
                    Idle.
                  </div>
                ) : (
                  trace.map((event) => (
                    <div
                      key={event.id}
                      className="rounded-md border border-gray-700 bg-gray-800 px-3 py-2"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="truncate font-mono text-[10px] uppercase text-gray-500">
                          {event.type}
                        </span>
                        <span className="shrink-0 text-[10px] text-gray-500">
                          {nowLabel(event.timestamp)}
                        </span>
                      </div>
                      <p className="mt-1 text-xs leading-4 text-gray-200">{event.message}</p>
                    </div>
                  ))
                )}
              </div>
            </ScrollArea>
          </Card>
        </aside>
        )}
      </main>
    </div>
  );
}
