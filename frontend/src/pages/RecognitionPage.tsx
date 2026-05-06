import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { RecognitionResponse } from "../api/types";
import { blobToWavFile, preferredMimeType } from "../audio/recording";
import { StatusMessage } from "../components/StatusMessage";
import { TrackAudioPlayer } from "../components/TrackAudioPlayer";

const allowedExtensions = [".mp3", ".wav"];
const maxRecordingSeconds = 20;

function isAllowedAudio(file: File): boolean {
  const name = file.name.toLowerCase();
  return allowedExtensions.some((extension) => name.endsWith(extension));
}

function formatConfidence(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0.0000";
  }
  return value.toFixed(4);
}

export function RecognitionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [recordingFile, setRecordingFile] = useState<File | null>(null);
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<number | null>(null);
  const autoStopRef = useRef<number | null>(null);

  const fileHint = useMemo(() => {
    if (!file) {
      return "MP3 or WAV file";
    }
    return `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  }, [file]);

  useEffect(() => {
    return () => {
      clearRecordingTimers();
      stopStream();
      if (recordingUrl) {
        URL.revokeObjectURL(recordingUrl);
      }
    };
  }, [recordingUrl]);

  function clearRecordingTimers() {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (autoStopRef.current) {
      window.clearTimeout(autoStopRef.current);
      autoStopRef.current = null;
    }
  }

  function stopStream() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  function clearRecording() {
    setRecordingFile(null);
    setRecordingSeconds(0);
    setResult(null);
    setError("");
    setRecordingUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return null;
    });
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null;
    setResult(null);
    setError("");

    if (!nextFile) {
      setFile(null);
      return;
    }

    if (!isAllowedAudio(nextFile)) {
      setFile(null);
      setError("Only mp3 and wav files are supported");
      event.target.value = "";
      return;
    }

    setFile(nextFile);
  }

  async function submitAudio(audioFile: File, source: "upload" | "microphone") {
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const response = await api.recognize(audioFile, source);
      setResult(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Recognition service is unavailable");
    } finally {
      setLoading(false);
    }
  }

  async function handleUploadSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!file) {
      setError("Choose an audio file first");
      return;
    }

    await submitAudio(file, "upload");
  }

  async function startRecording() {
    setError("");
    setResult(null);
    clearRecording();

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone recording is not supported in this browser");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = preferredMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

      streamRef.current = stream;
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        void finalizeRecording(recorder.mimeType || "audio/webm");
      };

      recorder.start();
      setRecording(true);
      setRecordingSeconds(0);

      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((value) => Math.min(value + 1, maxRecordingSeconds));
      }, 1000);

      autoStopRef.current = window.setTimeout(() => {
        stopRecording();
      }, maxRecordingSeconds * 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Microphone access was denied");
      stopStream();
      clearRecordingTimers();
      setRecording(false);
    }
  }

  function stopRecording() {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    clearRecordingTimers();
    setRecording(false);
    stopStream();
  }

  async function finalizeRecording(mimeType: string) {
    const blob = new Blob(chunksRef.current, { type: mimeType });
    chunksRef.current = [];

    if (!blob.size) {
      setError("Recording is empty");
      return;
    }

    try {
      const wavFile = await blobToWavFile(blob);
      setRecordingFile(wavFile);
      setRecordingUrl((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous);
        }
        return URL.createObjectURL(wavFile);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to prepare WAV recording");
    }
  }

  async function sendRecording() {
    if (!recordingFile) {
      setError("Record audio before sending it");
      return;
    }

    await submitAudio(recordingFile, "microphone");
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Recognition</p>
          <h1>Recognize audio</h1>
        </div>
      </header>

      <div className="workspace-grid">
        <div className="input-stack">
          <form className="panel upload-panel" onSubmit={handleUploadSubmit}>
            <div className="panel-heading">
              <h2>File upload</h2>
            </div>

            <label className="file-drop">
              <input accept=".mp3,.wav,audio/mpeg,audio/wav" type="file" onChange={handleFileChange} />
              <span>Choose file</span>
              <strong>{fileHint}</strong>
            </label>

            <button className="primary-button" disabled={loading || !file}>
              {loading ? "Recognizing..." : "Recognize file"}
            </button>
          </form>

          <section className="panel recorder-panel">
            <div className="panel-heading">
              <h2>Microphone</h2>
            </div>

            <div className="recording-meter">
              <span className={recording ? "record-dot active" : "record-dot"} />
              <strong>{recording ? "Recording" : "Ready"}</strong>
              <span>
                {recordingSeconds}s / {maxRecordingSeconds}s
              </span>
            </div>

            <div className="button-row">
              {!recording ? (
                <button className="primary-button" disabled={loading} onClick={() => void startRecording()}>
                  Start recording
                </button>
              ) : (
                <button className="danger-button" onClick={stopRecording}>
                  Stop
                </button>
              )}
              <button className="ghost-button" disabled={recording || !recordingFile} onClick={clearRecording}>
                Clear
              </button>
            </div>

            {recordingUrl && (
              <div className="recording-preview">
                <audio controls src={recordingUrl} />
                <button className="primary-button" disabled={loading} onClick={() => void sendRecording()}>
                  {loading ? "Recognizing..." : "Send recording"}
                </button>
              </div>
            )}
          </section>
        </div>

        <section className="panel result-panel" aria-live="polite">
          <div className="panel-heading">
            <h2>Result</h2>
          </div>

          {error && <StatusMessage tone="error">{error}</StatusMessage>}

          {!result && !loading && !error && (
            <div className="empty-state">Result will appear after sending audio.</div>
          )}

          {loading && <div className="loading-state">Analyzing audio...</div>}

          {result && (
            <div className="recognition-result">
              <div className={`result-badge ${result.match ? "found" : "not-found"}`}>
                {result.match ? "FOUND" : "NOT FOUND"}
              </div>

              {result.match ? (
                <>
                  <dl className="detail-list">
                    <div>
                      <dt>Title</dt>
                      <dd>{result.title || "UNKNOWN"}</dd>
                    </div>
                    <div>
                      <dt>Artist</dt>
                      <dd>{result.artist || "UNKNOWN"}</dd>
                    </div>
                    <div>
                      <dt>Confidence</dt>
                      <dd>{formatConfidence(result.confidence)}</dd>
                    </div>
                    <div>
                      <dt>Source</dt>
                      <dd>{result.source || "unknown"}</dd>
                    </div>
                    <div>
                      <dt>Track ID</dt>
                      <dd>{result.trackId || "n/a"}</dd>
                    </div>
                  </dl>
                  <TrackAudioPlayer trackId={result.trackId} />
                </>
              ) : (
                <StatusMessage tone="info">Track was not found in the catalog.</StatusMessage>
              )}
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
