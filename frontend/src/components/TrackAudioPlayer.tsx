import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import { StatusMessage } from "./StatusMessage";

type TrackAudioPlayerProps = {
  trackId: number | string | null | undefined;
};

export function TrackAudioPlayer({ trackId }: TrackAudioPlayerProps) {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  async function loadAudio() {
    if (!trackId) {
      return;
    }

    setError("");
    setLoading(true);

    try {
      const blob = await api.getTrackAudio(trackId);
      const nextUrl = URL.createObjectURL(blob);
      setAudioUrl((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous);
        }
        return nextUrl;
      });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Audio is unavailable");
    } finally {
      setLoading(false);
    }
  }

  if (!trackId) {
    return null;
  }

  return (
    <div className="audio-player-box">
      {!audioUrl ? (
        <button className="ghost-button" disabled={loading} onClick={loadAudio}>
          {loading ? "Loading audio..." : "Play original"}
        </button>
      ) : (
        <audio controls src={audioUrl} />
      )}
      {error && <StatusMessage tone="error">{error}</StatusMessage>}
    </div>
  );
}
