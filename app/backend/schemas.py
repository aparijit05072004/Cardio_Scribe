"""The API contract.

These models are the single source of truth shared by the FastAPI backend, the
SPA and the Gradio app. DEMO and LIVE return the *same* shape, so the UI never
branches on mode except to show its provenance banner.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal["demo", "live"]
JobState = Literal["queued", "processing", "completed", "failed"]


class QCInfo(BaseModel):
    sampling_rate: int
    n_leads: int
    n_samples: int
    duration_s: float
    units: str = "mV"
    clipped: bool = Field(False, description="any |sample| >= 4.0 mV, i.e. likely ADC saturation")
    clipped_leads: list[str] = []
    flat_leads: list[str] = []
    warnings: list[str] = []


class Diagnosis(BaseModel):
    label: str
    probability: float


class Retrieved(BaseModel):
    similarity: float
    report_text: str
    ecg_id: int


class Provenance(BaseModel):
    """Where each part of the result came from. The UI renders this verbatim.

    In DEMO mode `report_is_model_output` is False and the UI must say so; a
    placeholder report presented as a clinical draft is the exact failure the
    model card forbids.
    """
    mode: Mode
    report_is_model_output: bool
    probabilities_are_model_output: bool
    measurement_source: str = Field(..., description="e.g. 'neurokit2', 'demo-detector'")
    notes: list[str] = []


class AnalysisResult(BaseModel):
    record_name: str
    provenance: Provenance
    qc: QCInfo

    # --- the drafted report ---
    report: str
    report_fields: dict[str, str] = Field(
        default_factory=dict,
        description="the 8-field schema parsed out of `report`, where present",
    )

    # --- evidence ---
    diagnoses: list[Diagnosis]
    probabilities: dict[str, float]
    threshold: float
    measurements: dict[str, Optional[float]]
    intervals: Optional[dict[str, Any]] = None

    # --- explainability ---
    signal: list[list[float]] = Field(..., description="[n_samples][12] standardized mV")
    signal_decimation: int = 1
    lead_names: list[str]
    sampling_rate: int
    attention: Optional[list[float]] = Field(
        None, description="length n_samples, sums to 1, or null when unavailable"
    )
    lead_importance: dict[str, float] = {}
    retrieved: list[Retrieved] = []

    # --- guardrails ---
    confident: bool
    issues: list[str] = []
    flagged_for_review: bool
    repair_actions: list[str] = []

    elapsed_s: float = 0.0


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    stage: str
    stage_index: int
    n_stages: int
    message: str
    queue_position: Optional[int] = None
    error: Optional[str] = None
    created_at: str
    result: Optional[AnalysisResult] = None


class SubmitResponse(BaseModel):
    job_id: str
    status: JobState
    queue_position: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    mode: Mode
    runtime_dir: Optional[str] = None
    runtime_ready: bool
    missing: list[str] = Field(
        default_factory=list,
        description="symbols the live runtime module does not export",
    )
    threshold: float
    sampling_rate: int
    queue_depth: int
    review_csv: str


class TemplateInfo(BaseModel):
    name: str
    label: str
    instruction: str


class ReviewRequest(BaseModel):
    record_name: str
    generated_report: str
    edited_report: str
    measured_hr: Optional[float] = None
    flagged_for_review: bool = False
    mode: Mode = "demo"
    likert_accuracy: int = Field(..., ge=1, le=5)
    likert_completeness: int = Field(..., ge=1, le=5)
    likert_safety: int = Field(..., ge=1, le=5)
    reviewer_notes: str = ""


class ReviewResponse(BaseModel):
    saved: bool
    path: str
    rows: int
