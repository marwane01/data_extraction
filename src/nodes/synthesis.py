import json
import logging
from src.utils.llm import get_llm
from src.schemas import SummarySchema  # <--- IMPORTA DA SCHEMAS.PY

logger = logging.getLogger(__name__)


async def synthesize_node(state: dict):
    llm = get_llm()
    master = state.get("master_bundle", {})

    # Check di sicurezza per evitare di chiamare l'LLM con dati vuoti
    if not master.get("medications") and not master.get("observations"):
        return {"summary_json": json.dumps({"status": "Dati insufficienti"})}

    prompt = f"""
### TASK: SINTESI CLINICA AD ALTA PRECISIONE
Analizza il record FHIR e crea un sommario per l'AI Agent.
1. BRIEF_SNAPSHOT: Riassunto della storia (Paziente: Maurizio Forlanelli).
2. RECENT_VITALS: Estrai gli ultimi VALORI NUMERICI reali (es. 'Creatinina: 1.58 mg/dL'). 
   - Se un valore ha una data, includila.
   - Restituisci i valori come STRINGHE semplici, NON oggetti.
3. MEDICATIONS: Lista farmaci attivi con orari.

DATI: {json.dumps(master)}
"""

    # Usa lo schema importato
    structured_llm = llm.with_structured_output(SummarySchema)

    try:
        summary = await structured_llm.ainvoke(prompt)
        if summary is None:
            return {"summary_json": "{}"}
        return {"summary_json": summary.model_dump_json(indent=2)}
    except Exception as e:
        logger.error(f"Synthesis Error: {e}")
        # Restituiamo un JSON di errore valido per non rompere il grafo
        return {
            "summary_json": json.dumps({"error": "Impossibile generare la sintesi"})
        }
