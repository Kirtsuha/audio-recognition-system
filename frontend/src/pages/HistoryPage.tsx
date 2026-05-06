import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { RecognitionHistoryItem } from "../api/types";
import { StatusMessage } from "../components/StatusMessage";

function formatDate(value: string | null): string {
  if (!value) {
    return "n/a";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function formatConfidence(value: number): string {
  return Number.isFinite(value) ? value.toFixed(4) : "0.0000";
}

export function HistoryPage() {
  const [items, setItems] = useState<RecognitionHistoryItem[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [mutatingId, setMutatingId] = useState<number | null>(null);
  const [clearing, setClearing] = useState(false);

  const loadHistory = useCallback(async () => {
    setError("");
    setLoading(true);
    try {
      setItems(await api.getHistory());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось загрузить историю");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  async function deleteItem(id: number) {
    setError("");
    setMutatingId(id);
    try {
      await api.deleteHistoryItem(id);
      await loadHistory();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось удалить запись");
    } finally {
      setMutatingId(null);
    }
  }

  async function clearHistory() {
    if (!window.confirm("Очистить всю историю распознаваний?")) {
      return;
    }

    setError("");
    setClearing(true);
    try {
      await api.clearHistory();
      await loadHistory();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось очистить историю");
    } finally {
      setClearing(false);
    }
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">History</p>
          <h1>История распознаваний</h1>
        </div>
        <button className="danger-button" disabled={clearing || items.length === 0} onClick={clearHistory}>
          {clearing ? "Clearing..." : "Clear all"}
        </button>
      </header>

      {error && <StatusMessage tone="error">{error}</StatusMessage>}

      <section className="panel">
        {loading ? (
          <div className="loading-state">Загружаем историю...</div>
        ) : items.length === 0 ? (
          <div className="empty-state">История пока пустая.</div>
        ) : (
          <div className="history-list">
            {items.map((item) => (
              <article className="history-row" key={item.id}>
                <div className="history-main">
                  <div className="history-title">
                    <span>#{item.id}</span>
                    <strong>{item.title || "NOT FOUND"}</strong>
                  </div>
                  <div className="history-meta">
                    <span>{item.filename || "unknown file"}</span>
                    <span>{item.artist || "UNKNOWN"}</span>
                    <span>trackId: {item.trackId || "n/a"}</span>
                    <span>confidence: {formatConfidence(item.confidence)}</span>
                    <span>source: {item.source || "unknown"}</span>
                    <span>{formatDate(item.createdAt)}</span>
                  </div>
                </div>

                <button
                  className="ghost-button"
                  disabled={mutatingId === item.id}
                  onClick={() => void deleteItem(item.id)}
                >
                  {mutatingId === item.id ? "Deleting..." : "Delete"}
                </button>
              </article>
            ))}
          </div>
        )}
      </section>
    </section>
  );
}
