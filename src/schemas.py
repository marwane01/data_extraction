from typing import List, Optional
from pydantic import BaseModel, Field, AliasChoices


class Patient(BaseModel):
    name: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("name", "paziente", "nome_cognome"),
        description="Full Name of the patient. NEVER use John Doe.",
    )
    cf: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("cf", "codice_fiscale", "tax_id"),
        description="Italian Tax ID (Codice Fiscale).",
    )


class Observation(BaseModel):
    test_name: str = Field(
        ...,
        validation_alias=AliasChoices("test_name", "esame", "parametro", "analisi"),
        description="The lab test or vital sign name (e.g., Emoglobina).",
    )
    value: str = Field(
        ...,
        validation_alias=AliasChoices("value", "risultato", "valore"),
        description="The numerical or textual result.",
    )
    unit: Optional[str] = Field(
        None, validation_alias=AliasChoices("unit", "unità", "u_m")
    )
    date: Optional[str] = Field(
        None, validation_alias=AliasChoices("date", "data", "data_prelievo")
    )


class MedicationRequest(BaseModel):
    medication: str = Field(
        ...,
        validation_alias=AliasChoices(
            "medication", "farmaco", "principio_attivo", "terapia"
        ),
        description="The drug name (e.g., Bisoprololo).",
    )
    dosage: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("dosage", "dosaggio", "posologia", "frequenza"),
        description="Dose, frequency, and timing instructions.",
    )


class Condition(BaseModel):
    code: str = Field(
        ...,
        validation_alias=AliasChoices("code", "diagnosi", "patologia", "problema"),
        description="Diagnosis or clinical event name.",
    )
    onset_date: Optional[str] = Field(
        None, validation_alias=AliasChoices("onset_date", "insorgenza", "data_diagnosi")
    )


class FHIRBundle(BaseModel):
    """A collection of FHIR-aligned clinical resources."""

    patient: Optional[Patient] = None
    medications: List[MedicationRequest] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    conditions: List[Condition] = Field(default_factory=list)
