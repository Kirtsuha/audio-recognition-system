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
  s3Key: string | null;
};

export type TrackSearchResponse = {
  items: Track[];
  count: number;
};
