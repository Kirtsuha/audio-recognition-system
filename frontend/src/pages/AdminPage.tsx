import { ChangeEvent, FormEvent, useState } from "react";
import { api, ApiError } from "../api/client";
import type {
  AdminBulkUploadResponse,
  AdminTrackUploadResponse,
  CatalogJobResponse,
  CatalogJobStartedResponse,
  CatalogSyncResponse,
} from "../api/types";
import { StatusMessage } from "../components/StatusMessage";

type Result = {
  title: string;
  payload: unknown;
};

function fileName(file: File | null, fallback: string) {
  return file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB` : fallback;
}

function errorText(error: unknown, fallback: string) {
  return error instanceof ApiError ? error.message : fallback;
}

function formatPayload(payload: unknown) {
  return JSON.stringify(payload, null, 2);
}

export function AdminPage() {
  const [trackFile, setTrackFile] = useState<File | null>(null);
  const [trackTitle, setTrackTitle] = useState("");
  const [trackArtist, setTrackArtist] = useState("");
  const [trackAlbum, setTrackAlbum] = useState("");

  const [archiveFile, setArchiveFile] = useState<File | null>(null);
  const [manifestFile, setManifestFile] = useState<File | null>(null);
  const [bucket, setBucket] = useState("");
  const [prefix, setPrefix] = useState("");
  const [maxFiles, setMaxFiles] = useState("");

  const [jobId, setJobId] = useState("");
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);

  function chooseFile(setter: (file: File | null) => void) {
    return (event: ChangeEvent<HTMLInputElement>) => {
      setter(event.target.files?.[0] ?? null);
      setError("");
      setResult(null);
    };
  }

  async function run<T>(key: string, title: string, action: () => Promise<T>) {
    setError("");
    setResult(null);
    setLoading(key);

    try {
      const payload = await action();
      setResult({ title, payload });
      return payload;
    } catch (err) {
      setError(errorText(err, "Admin request failed"));
      return null;
    } finally {
      setLoading("");
    }
  }

  async function uploadTrack(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!trackFile) {
      setError("Choose a track file");
      return;
    }
    if (!trackTitle.trim() || !trackArtist.trim()) {
      setError("Title and artist are required");
      return;
    }

    await run<AdminTrackUploadResponse>("track", "Single track upload", () =>
      api.adminUploadTrack(trackFile, trackTitle.trim(), trackArtist.trim(), trackAlbum.trim()),
    );
  }

  async function uploadBulk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!archiveFile) {
      setError("Choose a ZIP archive");
      return;
    }
    if (!manifestFile) {
      setError("Choose a CSV manifest");
      return;
    }

    await run<AdminBulkUploadResponse>("bulk", "Bulk upload", () =>
      api.adminUploadBulk(archiveFile, manifestFile, bucket, prefix, maxFiles),
    );
  }

  async function syncCatalog() {
    const payload = await run<CatalogSyncResponse>("sync", "Catalog sync", api.adminSyncCatalog);
    if (payload?.jobId) {
      setJobId(payload.jobId);
    }
  }

  async function fullRebuild() {
    const payload = await run<CatalogJobStartedResponse>("rebuild", "Full rebuild", api.adminFullRebuild);
    if (payload?.jobId) {
      setJobId(payload.jobId);
    }
  }

  async function getJob(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!jobId.trim()) {
      setError("Enter job ID");
      return;
    }

    await run<CatalogJobResponse>("job", "Catalog job", () => api.adminGetJob(jobId.trim()));
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>Catalog administration</h1>
        </div>
      </header>

      {error && <StatusMessage tone="error">{error}</StatusMessage>}

      <div className="admin-grid">
        <form className="panel admin-form" onSubmit={uploadTrack}>
          <div className="panel-heading">
            <h2>Single track</h2>
          </div>

          <label className="file-drop compact-file-drop">
            <input accept=".mp3,.wav,.flac,.ogg,.m4a,.aac,audio/*" type="file" onChange={chooseFile(setTrackFile)} />
            <span>Choose audio</span>
            <strong>{fileName(trackFile, "Audio file")}</strong>
          </label>

          <label>
            Title
            <input value={trackTitle} onChange={(event) => setTrackTitle(event.target.value)} />
          </label>
          <label>
            Artist
            <input value={trackArtist} onChange={(event) => setTrackArtist(event.target.value)} />
          </label>
          <label>
            Album
            <input value={trackAlbum} onChange={(event) => setTrackAlbum(event.target.value)} />
          </label>

          <button className="primary-button" disabled={loading === "track"}>
            {loading === "track" ? "Uploading..." : "Upload track"}
          </button>
        </form>

        <form className="panel admin-form" onSubmit={uploadBulk}>
          <div className="panel-heading">
            <h2>Bulk archive</h2>
          </div>

          <label className="file-drop compact-file-drop">
            <input accept=".zip,application/zip" type="file" onChange={chooseFile(setArchiveFile)} />
            <span>Choose ZIP</span>
            <strong>{fileName(archiveFile, "Archive file")}</strong>
          </label>

          <label className="file-drop compact-file-drop">
            <input accept=".csv,text/csv" type="file" onChange={chooseFile(setManifestFile)} />
            <span>Choose CSV</span>
            <strong>{fileName(manifestFile, "Manifest file")}</strong>
          </label>

          <label>
            Bucket
            <input value={bucket} placeholder="default full-tracks" onChange={(event) => setBucket(event.target.value)} />
          </label>
          <label>
            Prefix
            <input value={prefix} placeholder="optional" onChange={(event) => setPrefix(event.target.value)} />
          </label>
          <label>
            Max files
            <input inputMode="numeric" value={maxFiles} placeholder="optional" onChange={(event) => setMaxFiles(event.target.value)} />
          </label>

          <button className="primary-button" disabled={loading === "bulk"}>
            {loading === "bulk" ? "Uploading..." : "Upload archive"}
          </button>
        </form>

        <section className="panel admin-form">
          <div className="panel-heading">
            <h2>Catalog jobs</h2>
          </div>

          <div className="button-row">
            <button className="primary-button" disabled={loading === "sync"} onClick={() => void syncCatalog()}>
              {loading === "sync" ? "Running..." : "Incremental sync"}
            </button>
            <button className="ghost-button" disabled={loading === "rebuild"} onClick={() => void fullRebuild()}>
              {loading === "rebuild" ? "Starting..." : "Full rebuild"}
            </button>
          </div>

          <form className="job-query" onSubmit={getJob}>
            <label>
              Job ID
              <input value={jobId} onChange={(event) => setJobId(event.target.value)} />
            </label>
            <button className="primary-button" disabled={loading === "job"}>
              {loading === "job" ? "Loading..." : "Get job"}
            </button>
          </form>
        </section>

        <section className="panel admin-result">
          <div className="panel-heading">
            <h2>Result</h2>
          </div>

          {!result ? (
            <div className="empty-state">Admin response will appear here.</div>
          ) : (
            <div className="result-json">
              <strong>{result.title}</strong>
              <pre>{formatPayload(result.payload)}</pre>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
