from typing import List, Optional, Any, Union, Dict
from pydantic import BaseModel, Field, AliasChoices, ConfigDict, field_validator


class Patient(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Use Union to prevent crashes when LLM sends a FHIR name object
    name: Optional[Union[str, Any]] = None
    cf: Optional[str] = Field(None, description="Italian Tax ID (Codice Fiscale)")

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v: Any) -> str:
        if isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], dict):
                family = v[0].get("family", "")
                given = " ".join(v[0].get("given", []))
                return f"{family} {given}".strip()
        return str(v) if v else "Unknown"


class Observation(BaseModel):
    model_config = ConfigDict(extra="allow")
    test_name: str = Field(
        ..., validation_alias=AliasChoices("test_name", "esame", "parametro")
    )
    value: str = Field(
        ..., validation_alias=AliasChoices("value", "risultato", "valore")
    )
    unit: Optional[str] = None
    date: Optional[str] = None


class MedicationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    medication: str = Field(..., validation_alias=AliasChoices("medication", "farmaco"))
    dosage: Optional[str] = None
    timing: Optional[str] = None
    status: str = "active"


class FHIRBundle(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Usiamo default_factory per garantire che siano sempre liste, mai None
    patient: Optional[Patient] = Field(default_factory=Patient)
    medications: List[MedicationRequest] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    conditions: List[Any] = Field(default_factory=list)

    # Validatore per trasformare eventuali None in liste vuote
    @field_validator("medications", "observations", "conditions", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> List:
        return v if isinstance(v, list) else []


class SummarySchema(BaseModel):
    brief_clinical_snapshot: str = Field(
        ..., description="Sintesi dello stato di salute."
    )
    active_medications: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    recent_vitals: Dict[str, str] = Field(
        default_factory=dict,
        description="Parametri vitali recenti (es. Creatinina: 1.5 mg/dL)",
    )

    @field_validator("recent_vitals", mode="before")
    @classmethod
    def stringify_vitals(cls, v: Any) -> Dict[str, str]:
        """Intercepts LLM objects and flattens them into strings to prevent crashes."""
        if not isinstance(v, dict):
            return {}
        flattened = {}
        for key, val in v.items():
            if isinstance(val, dict):
                # Handles cases like: "creatinina": {"value": 1.5, "unit": "mg/dL", "date": "2025-11-20"}
                value = val.get("value", val.get("risultato", ""))
                unit = val.get("unit", val.get("u_m", ""))
                date = val.get("date", "")
                entry = f"{value} {unit}".strip()
                if date:
                    entry += f" ({date})"
                flattened[key] = entry
            else:
                flattened[key] = str(val)
        return flattened
