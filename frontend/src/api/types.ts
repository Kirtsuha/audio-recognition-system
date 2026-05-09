export type AuthResponse = {
  accessToken: string;
  tokenType: string;
};

export type RecognitionResponse = {
  match: boolean;
  trackId: string | null;
  title: string | null;
  artist: string | null;
  confidence: number;
  source: string;
};

export type RecognitionHistoryItem = {
  id: number;
  filename: string | null;
  trackId: string | null;
  title: string | null;
  artist: string | null;
  confidence: number;
  source: string | null;
  createdAt: string | null;
};

export type UserProfile = {
  id: number;
  username: string;
  role: string;
};

export type Track = {
  trackId: number;
  title: string | null;
  artist: string | null;
  album: string | null;
  durationSec?: number | null;
  s3Key: string | null;
};

export type TrackSearchResponse = {
  items: Track[];
  count: number;
};

export type AdminTrackUploadResponse = {
  trackId: number;
  title: string | null;
  artist: string | null;
  album: string | null;
  status: string;
};

export type AdminBulkUploadResponse = {
  status: string;
  summary: Record<string, unknown>;
};

export type CatalogJobStep = {
  name: string;
  status: string;
  errorMessage: string | null;
};

export type CatalogJobResponse = {
  jobId: string;
  type: string;
  status: string;
  startedAt: string | null;
  finishedAt: string | null;
  errorMessage: string | null;
  steps: CatalogJobStep[];
};

export type CatalogJobStartedResponse = {
  jobId: string;
  status: string;
};

export type CatalogSyncResponse = {
  jobId: string;
  status: string;
  steps: CatalogJobStep[];
};
