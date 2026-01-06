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
    patient: Optional[Patient] = Field(default_factory=Patient)
    medications: List[MedicationRequest] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    conditions: List[Any] = Field(default_factory=list)
