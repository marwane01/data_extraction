# src/nodes/refiner.py
import json
from src.utils.llm import get_llm
from src.schemas import FHIRBundle


async def refine_data_node(state: dict):
    llm = get_llm()
    master = state.get("master_bundle", {})
    if not master or not any(master.values()):
        return {"master_bundle": master}

    print("🧪 [REFINER] Performing Semantic Entity Resolution...")

    prompt = f"""
    You are a Clinical Data Architect. Your goal is to transform this fragmented FHIR Bundle into a high-fidelity record.
    
    TASKS:
    1. MEDICATIONS DEDUPLICATION (CRITICAL): 
       - Merge generic names with brand names (e.g., Selegilina = Jumex; Evolocumab = Repatha; Pramipexolo = Mirapexin; Duloxetina = Cymbalta/Alikres).
       - Ensure each medication appears ONLY ONCE. If multiple dosages exist, keep the most current or detailed one.
    2. CONDITIONS CLEANUP:
       - Merge 'MALATTIA DI PARKINSON' and 'Parkinson' into 'Parkinson's Disease'.
       - REMOVE any entries in 'Conditions' that are actually medications (e.g., 'JARDIANCE 10mg...').
    3. OBSERVATIONS FILTERING:
       - Keep ONLY actual clinical findings (labs, vitals, findings). 
       - Remove therapy metadata artifacts like 'JARDIANCE', 'PANTOPRAZOLO' from the Observations list.
    4. DATA NORMALIZATION:
       - Convert units like 'µbcg' or 'mcg' to standard 'µg'.
       - Translate Italian finding values to English if possible for FHIR alignment.

    DIRTY DATA:
    {json.dumps(master)}
    """

    # We use Cerebras with reasoning_effort="high" to handle the complex merging logic
    structured_llm = llm.with_structured_output(FHIRBundle)

    try:
        refined = await structured_llm.ainvoke(prompt)
        if refined:
            return {"master_bundle": refined.model_dump()}
        return {"master_bundle": master}
    except Exception as e:
        print(f"⚠️ [REFINER ERROR]: {e}")
        return {"master_bundle": master}
