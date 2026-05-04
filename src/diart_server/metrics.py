"""Prometheus metric definitions for the diart standalone server.

These module-level metric instances are imported and updated by other
modules (sessions, routes, model loading) to record runtime telemetry.
The metrics themselves are defined here so they are registered exactly
once with the default Prometheus registry.

See STANDALONE_SERVER.md section 5.9 for the metric specification.
"""

from prometheus_client import Counter, Gauge, Histogram

active_sessions = Gauge(
    "diart_active_sessions",
    "Number of active WebSocket sessions currently being served",
)

predictions_total = Counter(
    "diart_predictions_total",
    "Total number of speaker diarization predictions emitted",
    ["pipeline", "model"],
)

prediction_latency = Histogram(
    "diart_prediction_latency_seconds",
    "End-to-end latency from push_pcm to prediction emission, in seconds",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0),
)

audio_bytes_received = Counter(
    "diart_audio_bytes_received_total",
    "Total audio bytes received from clients",
)

session_duration = Histogram(
    "diart_session_duration_seconds",
    "Session duration from open to close, in seconds",
    buckets=(10, 30, 60, 300, 900, 1800, 3600, 7200, 14400, 28800),
)

model_load_seconds = Gauge(
    "diart_model_load_seconds",
    "Time taken to load each model at warm-up, in seconds",
    ["model"],
)

session_errors_total = Counter(
    "diart_session_errors_total",
    "Total session errors emitted, by error code",
    ["code"],
)

queue_overflow_total = Counter(
    "diart_queue_overflow_total",
    "Total times the audio queue overflowed and dropped a frame",
)
