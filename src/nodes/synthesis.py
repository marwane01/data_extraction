from pydantic import BaseModel, Field
from src.utils.llm import get_llm
import json


class SummarySchema(BaseModel):
    brief_clinical_snapshot: str = Field(
        ..., description="2-3 sentence overview of current health state."
    )
    active_medications: list[str] = Field(
        ..., description="List of currently prescribed drugs."
    )
    key_findings: list[str] = Field(
        ..., description="Critical lab trends or diagnoses."
    )
    recent_vitals: dict[str, str] = Field(
        ..., description="Latest lab values/vitals recorded."
    )


async def synthesize_node(state: dict):
    llm = get_llm()
    master = state.get("master_bundle", {})

    prompt = f"""
    ### COMPITO: SINTESI CLINICA PER AGENTE AI
    Analizza il record FHIR e crea un riassunto strutturato in ITALIANO.
    - Riassumi la storia clinica (Brief History).
    - Elenca i farmaci ATTUALMENTE in uso.
    - Evidenzia gli ultimi valori di laboratorio rilevanti (Creatinina, Glicemia, Colesterolo).
    
    DATI:
    {json.dumps(master)}
    """

    structured_llm = llm.with_structured_output(SummarySchema)
    summary = await structured_llm.ainvoke(prompt)
    return {"summary_json": summary.model_dump_json(indent=2)}
