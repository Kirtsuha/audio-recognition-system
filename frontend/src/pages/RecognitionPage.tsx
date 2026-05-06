import { ChangeEvent, FormEvent, useMemo, useState } from "react";
import { api, ApiError } from "../api/client";
import type { RecognitionResponse } from "../api/types";
import { StatusMessage } from "../components/StatusMessage";

const allowedExtensions = [".mp3", ".wav"];

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
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const fileHint = useMemo(() => {
    if (!file) {
      return "MP3 или WAV файл";
    }
    return `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  }, [file]);

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
      setError("Поддерживаются только mp3 и wav файлы");
      event.target.value = "";
      return;
    }

    setFile(nextFile);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!file) {
      setError("Выберите аудиофайл для распознавания");
      return;
    }

    setLoading(true);
    try {
      const response = await api.recognize(file, "upload");
      setResult(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Сервис распознавания недоступен");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Recognition</p>
          <h1>Распознавание аудиофайла</h1>
        </div>
      </header>

      <div className="workspace-grid">
        <form className="panel upload-panel" onSubmit={handleSubmit}>
          <label className="file-drop">
            <input accept=".mp3,.wav,audio/mpeg,audio/wav" type="file" onChange={handleFileChange} />
            <span>Выбрать файл</span>
            <strong>{fileHint}</strong>
          </label>

          {error && <StatusMessage tone="error">{error}</StatusMessage>}

          <button className="primary-button" disabled={loading || !file}>
            {loading ? "Recognizing..." : "Recognize"}
          </button>
        </form>

        <section className="panel result-panel" aria-live="polite">
          <div className="panel-heading">
            <h2>Result</h2>
          </div>

          {!result && !loading && (
            <div className="empty-state">Результат появится после отправки файла.</div>
          )}

          {loading && <div className="loading-state">Анализируем аудио...</div>}

          {result && (
            <div className="recognition-result">
              <div className={`result-badge ${result.match ? "found" : "not-found"}`}>
                {result.match ? "FOUND" : "NOT FOUND"}
              </div>

              {result.match ? (
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
              ) : (
                <StatusMessage tone="info">Трек не найден в каталоге.</StatusMessage>
              )}
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
