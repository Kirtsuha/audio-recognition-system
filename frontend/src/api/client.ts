import { clearStoredToken, getStoredToken } from "../auth/tokenStore";
import type {
  AuthResponse,
  RecognitionHistoryItem,
  RecognitionResponse,
  Track,
  TrackSearchResponse,
  UserProfile,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiError extends Error {
  status: number;
  details: unknown;

  constructor(message: string, status: number, details: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

async function parseResponse(response: Response): Promise<unknown> {
  if (response.status === 204) {
    return null;
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response.text();
}

function errorMessage(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
  }

  if (typeof payload === "string" && payload.trim()) {
    return payload;
  }

  return fallback;
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  protectedRequest = true,
): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getStoredToken();
  const isFormData = init.body instanceof FormData;

  if (!isFormData && init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  if (protectedRequest && token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
  });

  const payload = await parseResponse(response);

  if (response.status === 401) {
    clearStoredToken();
    window.dispatchEvent(new Event("auth:expired"));
  }

  if (!response.ok) {
    throw new ApiError(
      errorMessage(payload, `Request failed with status ${response.status}`),
      response.status,
      payload,
    );
  }

  return payload as T;
}

async function requestBlob(path: string): Promise<Blob> {
  const headers = new Headers();
  const token = getStoredToken();

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, { headers });

  if (response.status === 401) {
    clearStoredToken();
    window.dispatchEvent(new Event("auth:expired"));
  }

  if (!response.ok) {
    const payload = await parseResponse(response);
    throw new ApiError(
      errorMessage(payload, `Request failed with status ${response.status}`),
      response.status,
      payload,
    );
  }

  return response.blob();
}

export const api = {
  login(username: string, password: string) {
    return request<AuthResponse>(
      "/api/auth/login",
      {
        method: "POST",
        body: JSON.stringify({ username, password }),
      },
      false,
    );
  },

  register(username: string, password: string) {
    return request<AuthResponse>(
      "/api/auth/register",
      {
        method: "POST",
        body: JSON.stringify({ username, password }),
      },
      false,
    );
  },

  recognize(file: File, source = "upload") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("source", source);

    return request<RecognitionResponse>("/api/recognition", {
      method: "POST",
      body: formData,
    });
  },

  getHistory() {
    return request<RecognitionHistoryItem[]>("/api/history");
  },

  deleteHistoryItem(id: number) {
    return request<void>(`/api/history/${id}`, {
      method: "DELETE",
    });
  },

  clearHistory() {
    return request<void>("/api/history", {
      method: "DELETE",
    });
  },

  getProfile() {
    return request<UserProfile>("/api/users/me");
  },

  deleteProfile() {
    return request<void>("/api/users/me", {
      method: "DELETE",
    });
  },

  searchTracks(query: string, artist = "", limit = 20) {
    const params = new URLSearchParams({ limit: String(limit) });
    if (query.trim()) {
      params.set("query", query.trim());
    }
    if (artist.trim()) {
      params.set("artist", artist.trim());
    }
    return request<TrackSearchResponse>(`/api/tracks/search?${params}`);
  },

  getTrack(id: number | string) {
    return request<Track>(`/api/tracks/${id}`);
  },

  getTrackAudio(id: number | string) {
    return requestBlob(`/api/tracks/${id}/audio`);
  },
};
