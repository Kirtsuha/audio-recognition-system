from pydantic import BaseModel, Field


class FingerprintParams(BaseModel):
    sample_rate: int = 11025
    n_fft: int = 2048
    hop_length: int = 512

    peak_neighborhood_size: int = 12
    amp_min_db: float = -40.0
    max_peaks_per_frame: int = 4

    fan_value: int = 8
    min_delta_t: int = 2
    max_delta_t: int = 60

    freq_bin_size: int = 2
    delta_t_bin_size: int = 2

    offset_bin: int = 3
    offset_tolerance_bins: int = 3

    min_aligned_matches: int = 15
    min_query_coverage: float = 0.002
    min_score_gap: float = 1.2


class FingerprintExperimentRequest(BaseModel):
    experiment_name: str = Field(default="experiment", min_length=1, max_length=100)

    bucket: str = "full-tracks"
    prefix: str = ""

    track_limit: int = Field(default=1000, ge=1)
    query_limit: int | None = Field(default=500, ge=1)

    clean_queries_per_track: int = Field(default=1, ge=0)
    noisy_queries_per_track: int = Field(default=1, ge=0)
    negative_queries: int = Field(default=150, ge=0)

    durations_sec: list[float] = Field(default_factory=lambda: [5.0, 10.0, 15.0])

    noisy_corruptions: list[str] = Field(
        default_factory=lambda: [
            "noise_snr20",
            "noise_snr10",
            "noise_snr5",
            "gain",
            "clipping",
            "reverb",
            "mixed_noise_reverb",
            "strong_noisy",
        ]
    )

    index_duration_sec: float | None = None

    random_seed: int = 42

    fixed_eval_set_name: str = Field(default="default", min_length=1, max_length=100)
    recreate_eval_set: bool = False

    params: FingerprintParams


class AsyncJobResponse(BaseModel):
    accepted: bool
    status: str
    run_dir: str | None = None