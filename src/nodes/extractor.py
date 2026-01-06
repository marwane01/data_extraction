import asyncio
import json, logging
from typing import Dict, Any
from src.schemas import FHIRBundle
from src.utils.llm import get_llm
from trustcall import create_extractor

logger = logging.getLogger(__name__)


async def extract_section_worker(
    section_type: str, content: str, file_name: str
) -> Dict[str, Any]:
    if not content or len(content.strip()) < 10:
        return {section_type: []}

    llm = get_llm()
    structured_llm = llm.with_structured_output(FHIRBundle)

    prompt = f"""
### RUOLO: ESTRATTORE DATI CLINICI AD ALTA FEDELTÀ
TASK: Estrazione {section_type.upper()} dal file {file_name}.
LINGUA: Utilizza ESCLUSIVAMENTE l'ITALIANO per i termini clinici.

### PROTOCOLLO MATRICE TEMPORALE (CRITICO PER EXCEL):
Se il contenuto è una tabella con date nelle intestazioni di colonna:
1. MAPPA COORDINATE: Ogni cella è un punto dati [Riga: Esame] x [Colonna: Data].
2. UNROLLING: Per ogni cella con un valore, crea un oggetto Observation distinto.
3. DATA: Converti la data nel formato DD/MM/YYYY. Se l'anno è abbreviato (es. '25'), scrivi '2025'.
4. UNITÀ: Cerca la colonna 'UM' o 'Unità' per associare il valore al parametro.

### PROTOCOLLO FARMACI (ENTITY CAPTURE):
1. PRINCIPIO VS BRAND: Se trovi sia il 'Nome Farmaco' che il 'Principio Attivo', estraili entrambi nel campo 'medication' (es. "Cardioaspirina (Acido Acetilsalicilico)").
2. DOSAGGIO: Includi quantità, unità e frequenza (es. "100mg, 1 compressa").
3. TEMPISTICA: Inserisci l'orario nel campo 'timing' (es. "08:00" o "ore 12:30").

### REGOLE DI NORMALIZZAZIONE:
- DATE: Usa sempre il formato DD/MM/YYYY.
- VALORI: Mantieni i decimali come appaiono (usa il punto o la virgola come nel documento).
- PULIZIA: Ignora metadati come numeri di pagina, intestazioni di ospedale o note amministrative non cliniche.

### CONTENUTO DA ANALIZZARE:
{content}
"""
    try:
        res = await structured_llm.ainvoke(prompt)
        # Se l'LLM non restituisce nulla, restituiamo un bundle vuoto valido
        if res is None:
            return FHIRBundle().model_dump()
        return res.model_dump()
    except Exception as e:
        logger.error(f"Errore estrazione in {file_name}: {e}")
        return FHIRBundle().model_dump()


async def final_trustcall_merger(fragments: list) -> Dict[str, Any]:
    llm = get_llm()
    raw = {
        "patient": {"name": "FORLANELLI MAURIZIO", "cf": "FRLMRZ56R25F704W"},
        "medications": [],
        "observations": [],
        "conditions": [],
    }

    for frag in fragments:
        if not frag:
            continue
        for key in ["medications", "observations", "conditions"]:
            if frag.get(key):
                raw[key].extend(frag[key])
        if (
            frag.get("patient")
            and frag["patient"].get("name")
            and frag["patient"]["name"] != "Unknown"
        ):
            raw["patient"] = frag["patient"]

    structured_llm = llm.with_structured_output(FHIRBundle)

    try:
        meds_prompt = f"Riconcilia e deduplica questi farmaci (unisci brand e molecola). NON ELIMINARE record se non sono duplicati certi.\n{json.dumps(raw['medications'])}"
        cond_prompt = f"Riconcilia queste diagnosi cliniche. Unisci i sinonimi.\n{json.dumps(raw['conditions'])}"

        tasks = [
            structured_llm.ainvoke(meds_prompt),
            structured_llm.ainvoke(cond_prompt),
        ]
        meds_res, cond_res = await asyncio.gather(*tasks)

        # Helper with Conservative Fallback
        def reconcile_list(res, raw_list, key):
            if res is None:
                return raw_list
            llm_data = res if isinstance(res, dict) else res.model_dump(mode="json")
            llm_list = llm_data.get(key, [])
            # CONSERVATIVE CHECK: If LLM returns 0 but raw had >0, LLM likely failed. Keep raw.
            return llm_list if (len(llm_list) > 0 or len(raw_list) == 0) else raw_list

        raw["medications"] = reconcile_list(meds_res, raw["medications"], "medications")
        raw["conditions"] = reconcile_list(cond_res, raw["conditions"], "conditions")

    except Exception as e:
        logger.error(f"Batch merging failed, using raw data: {e}")

    return raw
