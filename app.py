"""
KRM Air Monitor — Industrial AI Leak Warning System
====================================================
FastAPI backend with Isolation Forest anomaly detection and
real-time telemetry ingestion from ESP32 sensor clusters.
"""

from __future__ import annotations

import time
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("krm_air_monitor")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="KRM Air Monitor",
    description="Industrial AI Leak Warning System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class TelemetryPayload(BaseModel):
    """Schema for the ESP32 sensor payload."""
    mq2: int = Field(..., description="MQ-2 combustible gas ADC reading")
    mq135: int = Field(..., description="MQ-135 air quality ADC reading")
    temperature: float = Field(..., description="Ambient temperature in °C")
    humidity: float = Field(..., description="Relative humidity in %")
    pressure: float = Field(..., description="Barometric pressure in hPa")
    voc_gas_kohm: float = Field(..., description="VOC gas sensor resistance in kΩ")


class TelemetryRecord(BaseModel):
    """An enriched telemetry record stored in the rolling buffer."""
    timestamp: str
    epoch: float
    mq2: int
    mq135: int
    temperature: float
    humidity: float
    pressure: float
    voc_gas_kohm: float
    delta_mq2: float
    delta_mq135: float
    delta_voc: float
    ai_status: Literal["NORMAL", "PREDICTED LEAK", "CRITICAL LEAK"]
    ai_score: float
    buzzer_triggered: bool = Field(default=False, description="Whether the alarm buzzer is active")
    buzzer_mode: Literal["OFF", "WARNING", "CRITICAL"] = Field(default="OFF", description="Buzzer alarm pattern mode")
    spike_detected: bool = Field(default=False, description="Whether a sudden gas spike rate-of-change was triggered")


class LatestResponse(BaseModel):
    """Response model for GET /api/telemetry/latest."""
    connected: bool
    last_seen_seconds_ago: float | None
    record: TelemetryRecord | None


# ---------------------------------------------------------------------------
# In-memory rolling buffer & AI engine
# ---------------------------------------------------------------------------
BUFFER_SIZE = 60
buffer: deque[TelemetryRecord] = deque(maxlen=BUFFER_SIZE)

# Isolation Forest — retrained every ingestion on the rolling window
# contamination="auto" lets it adapt; low n_estimators for speed
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42,
)

# Configurable thresholds
MQ2_CRITICAL = 2800
MQ135_CRITICAL = 2200
MQ2_WARNING = 1800
MQ135_WARNING = 1400
DELTA_SPIKE = 300           # rapid rise per interval
VOC_DROP_THRESHOLD = -5.0   # kΩ drop per interval (rising concentration)


def _classify(
    mq2: int,
    mq135: int,
    delta_mq2: float,
    delta_mq135: float,
    delta_voc: float,
    anomaly_label: int,
    anomaly_score: float,
) -> tuple[Literal["NORMAL", "PREDICTED LEAK", "CRITICAL LEAK"], bool, Literal["OFF", "WARNING", "CRITICAL"], bool]:
    """
    Multi-signal classification combining rule-based thresholds with
    the Isolation Forest anomaly label.

    Returns:
        (ai_status, buzzer_triggered, buzzer_mode, spike_detected)
    """
    is_spike = (delta_mq2 >= DELTA_SPIKE) or (delta_mq135 >= DELTA_SPIKE)
    is_violent_spike = (delta_mq2 >= DELTA_SPIKE * 2) or (delta_mq135 >= DELTA_SPIKE * 2)

    # ---- CRITICAL: hard sensor thresholds or extreme spikes ----
    if mq2 >= MQ2_CRITICAL or mq135 >= MQ135_CRITICAL or is_violent_spike:
        return "CRITICAL LEAK", True, "CRITICAL", (is_spike or is_violent_spike)

    # ---- PREDICTED LEAK: early micro-leak indicators / moderate spikes ----
    # Rule 1 — moderate sensor levels
    if mq2 >= MQ2_WARNING or mq135 >= MQ135_WARNING:
        return "PREDICTED LEAK", True, "WARNING", is_spike

    # Rule 2 — significant rate-of-change spike
    if is_spike:
        return "PREDICTED LEAK", True, "WARNING", True

    # Rule 3 — VOC resistance dropping fast (gas concentration rising)
    if delta_voc <= VOC_DROP_THRESHOLD:
        return "PREDICTED LEAK", True, "WARNING", False

    # Rule 4 — Isolation Forest flags anomaly
    if anomaly_label == -1:
        return "PREDICTED LEAK", True, "WARNING", False

    return "NORMAL", False, "OFF", False


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/telemetry", status_code=201)
async def ingest_telemetry(payload: TelemetryPayload):
    """
    Ingest a single telemetry frame from the ESP32.
    Computes derivatives, runs the Isolation Forest, classifies status & buzzer trigger,
    and stores the enriched record in the rolling buffer.
    """
    now = datetime.now(timezone.utc)

    # --- compute deltas ---
    if buffer:
        prev = buffer[-1]
        delta_mq2 = float(payload.mq2 - prev.mq2)
        delta_mq135 = float(payload.mq135 - prev.mq135)
        delta_voc = float(payload.voc_gas_kohm - prev.voc_gas_kohm)
    else:
        delta_mq2 = 0.0
        delta_mq135 = 0.0
        delta_voc = 0.0

    # --- build feature vector ---
    features = np.array([[
        payload.mq2,
        payload.mq135,
        payload.temperature,
        payload.humidity,
        payload.voc_gas_kohm,
        delta_mq2,
        delta_mq135,
    ]])

    # --- Isolation Forest inference ---
    anomaly_label = 1    # default: normal
    anomaly_score = 0.0

    if len(buffer) >= 10:
        # Retrain on rolling window for adaptive behaviour
        history_features = np.array([
            [
                r.mq2, r.mq135, r.temperature, r.humidity,
                r.voc_gas_kohm, r.delta_mq2, r.delta_mq135,
            ]
            for r in buffer
        ])
        try:
            iso_forest.fit(history_features)
            anomaly_label = int(iso_forest.predict(features)[0])
            anomaly_score = float(iso_forest.decision_function(features)[0])
        except Exception as exc:
            logger.warning("Isolation Forest error: %s", exc)

    # --- classify & determine buzzer trigger ---
    status, buzzer_triggered, buzzer_mode, spike_detected = _classify(
        payload.mq2, payload.mq135,
        delta_mq2, delta_mq135, delta_voc,
        anomaly_label, anomaly_score,
    )

    record = TelemetryRecord(
        timestamp=now.isoformat(),
        epoch=now.timestamp(),
        mq2=payload.mq2,
        mq135=payload.mq135,
        temperature=payload.temperature,
        humidity=payload.humidity,
        pressure=payload.pressure,
        voc_gas_kohm=payload.voc_gas_kohm,
        delta_mq2=delta_mq2,
        delta_mq135=delta_mq135,
        delta_voc=delta_voc,
        ai_status=status,
        ai_score=anomaly_score,
        buzzer_triggered=buzzer_triggered,
        buzzer_mode=buzzer_mode,
        spike_detected=spike_detected,
    )
    buffer.append(record)

    if buzzer_triggered:
        logger.warning(
            "🚨 BUZZER TRIGGERED [%s] | %s | Spike=%s | MQ2=%d MQ135=%d ΔMQ2=%.1f ΔMQ135=%.1f",
            buzzer_mode, status, spike_detected, payload.mq2, payload.mq135, delta_mq2, delta_mq135
        )
    else:
        logger.info("✓  NORMAL | Buzzer: OFF | MQ2=%d MQ135=%d", payload.mq2, payload.mq135)

    return {
        "status": "ok",
        "ai_status": status,
        "buzzer_triggered": buzzer_triggered,
        "buzzer_mode": buzzer_mode,
        "spike_detected": spike_detected,
        "records_buffered": len(buffer),
    }


@app.get("/api/telemetry/latest", response_model=LatestResponse)
async def get_latest():
    """Return the most recent telemetry record and connection status."""
    if not buffer:
        return LatestResponse(connected=False, last_seen_seconds_ago=None, record=None)

    last = buffer[-1]
    age = time.time() - last.epoch
    return LatestResponse(
        connected=age < 30,       # consider disconnected after 30 s
        last_seen_seconds_ago=round(age, 1),
        record=last,
    )


@app.get("/api/telemetry/history")
async def get_history(limit: int = 60):
    """Return the rolling buffer (most-recent-last)."""
    records = list(buffer)[-limit:]
    return {
        "count": len(records),
        "records": [r.model_dump() for r in records],
    }


@app.post("/api/simulate")
async def simulate_reading(scenario: str = "normal"):
    """
    Generate and ingest a realistic simulated reading.
    Scenarios: 'normal', 'leak_early', 'critical_leak', 'recovery'
    """
    import random
    prev_mq2 = buffer[-1].mq2 if buffer else 420
    prev_mq135 = buffer[-1].mq135 if buffer else 360
    prev_temp = buffer[-1].temperature if buffer else 28.0
    prev_hum = buffer[-1].humidity if buffer else 60.0
    prev_press = buffer[-1].pressure if buffer else 1013.2
    prev_voc = buffer[-1].voc_gas_kohm if buffer else 50.0

    if scenario == "critical_leak":
        mq2 = min(3800, max(2900, prev_mq2 + random.randint(400, 800)))
        mq135 = min(3200, max(2300, prev_mq135 + random.randint(300, 600)))
        voc = max(5.0, prev_voc - random.uniform(8.0, 15.0))
    elif scenario == "leak_early":
        mq2 = min(2600, max(1900, prev_mq2 + random.randint(150, 350)))
        mq135 = min(2000, max(1500, prev_mq135 + random.randint(120, 250)))
        voc = max(15.0, prev_voc - random.uniform(3.0, 7.0))
    elif scenario == "recovery":
        mq2 = max(400, prev_mq2 - random.randint(200, 400))
        mq135 = max(2000, prev_mq135 - random.randint(150, 300))
        voc = min(52.0, prev_voc + random.uniform(3.0, 6.0))
    else:  # normal
        mq2 = max(380, min(650, prev_mq2 + random.randint(-15, 18)))
        mq135 = max(320, min(580, prev_mq135 + random.randint(-12, 15)))
        voc = max(42.0, min(55.0, prev_voc + random.uniform(-0.8, 0.8)))

    temp = round(max(20.0, min(40.0, prev_temp + random.uniform(-0.1, 0.1))), 1)
    hum = round(max(30.0, min(80.0, prev_hum + random.uniform(-0.2, 0.2))), 1)
    press = round(max(980.0, min(1030.0, prev_press + random.uniform(-0.05, 0.05))), 1)
    voc = round(voc, 1)

    payload = TelemetryPayload(
        mq2=mq2,
        mq135=mq135,
        temperature=temp,
        humidity=hum,
        pressure=press,
        voc_gas_kohm=voc
    )
    return await ingest_telemetry(payload)


# ---------------------------------------------------------------------------
# Serve the SPA
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_dashboard():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 KRM Air Monitor starting on http://0.0.0.0:5000")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info",
    )
