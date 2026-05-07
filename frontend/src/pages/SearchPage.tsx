import { FormEvent, useState } from "react";
import { api, ApiError } from "../api/client";
import type { Track } from "../api/types";
import { StatusMessage } from "../components/StatusMessage";
import { TrackAudioPlayer } from "../components/TrackAudioPlayer";

function durationLabel(value: number | null | undefined): string {
  if (!value || !Number.isFinite(value)) {
    return "n/a";
  }

  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [artist, setArtist] = useState("");
  const [tracks, setTracks] = useState<Track[]>([]);
  const [searched, setSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSearched(false);

    if (!query.trim() && !artist.trim()) {
      setError("Enter title or artist");
      return;
    }

    setLoading(true);

    try {
      const response = await api.searchTracks(query, artist, 30);
      setTracks(response.items ?? []);
      setSearched(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Search is unavailable");
      setTracks([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Search</p>
          <h1>Track catalog</h1>
        </div>
      </header>

      <form className="panel search-form" onSubmit={handleSubmit}>
        <label>
          Title
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Song title"
          />
        </label>

        <label>
          Artist
          <input
            value={artist}
            onChange={(event) => setArtist(event.target.value)}
            placeholder="Artist"
          />
        </label>

        <button className="primary-button" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && <StatusMessage tone="error">{error}</StatusMessage>}

      <section className="panel">
        {loading ? (
          <div className="loading-state">Searching catalog...</div>
        ) : searched && tracks.length === 0 ? (
          <div className="empty-state">Nothing found.</div>
        ) : !searched ? (
          <div className="empty-state">Enter a title or artist to search the catalog.</div>
        ) : (
          <div className="track-grid">
            {tracks.map((track) => (
              <article className="track-card" key={track.trackId}>
                <div className="track-card-main">
                  <h2>{track.title || "UNKNOWN"}</h2>
                  <p>{track.artist || "UNKNOWN"}</p>
                </div>

                <dl className="compact-details">
                  <div>
                    <dt>Album</dt>
                    <dd>{track.album || "n/a"}</dd>
                  </div>
                  <div>
                    <dt>Duration</dt>
                    <dd>{durationLabel(track.durationSec)}</dd>
                  </div>
                  <div>
                    <dt>Track ID</dt>
                    <dd>{track.trackId}</dd>
                  </div>
                </dl>

                <TrackAudioPlayer trackId={track.trackId} />
              </article>
            ))}
          </div>
        )}
      </section>
    </section>
  );
}
