import json
from src.utils.llm import get_llm
from src.schemas import FHIRBundle


async def refine_data_node(state: dict):
    llm = get_llm()
    master = state.get("master_bundle", {})

    if not master or not any(master.values()):
        return {"master_bundle": master}

    print(
        "🧹 [REFINER] Resolving clinical entities and cleaning administrative noise..."
    )

    prompt = f"""
    You are a Senior Clinical Data Engineer. I will provide a messy FHIR Bundle extracted from multiple Italian documents.
    Your goal is to produce a single, high-fidelity FHIRBundle.

    RULES:
    1. MEDICATIONS: Merge Brand names with Generics (e.g., 'Bivis' is 'olmesartan/amlodipina', 'Crestor' is 'Rosuvastatina'). Keep the most complete dosage string.
    2. CONDITIONS vs LABS: You have lab tests (like 'Globuli Bianchi') in the 'Conditions' list. MOVE them to 'Observations' if they are measurements, or DELETE if redundant. 
    3. OBSERVATIONS: Remove ALL administrative keys (Cognome, Nome, CF, Nato il, Nosologico, Operatore, Firmato, SI CONSIGLIA, Referto). These are NOT clinical observations.
    4. PATIENT: Ensure 'patient' object has the name 'MAURIZIO FORLANELLI' and CF 'FRLMRZ56R25F704W'.
    5. UNITS: Fix encoding issues (e.g., replace '\u0003bcg' with 'mcg' or 'µg').

    MESSY DATA:
    {json.dumps(master)}
    """

    # We use a higher temperature or reasoning effort if possible to ensure it handles the logic
    structured_llm = llm.with_structured_output(FHIRBundle)

    try:
        refined = await structured_llm.ainvoke(prompt)
        if refined:
            return {"master_bundle": refined.model_dump()}
        return {"master_bundle": master}
    except Exception as e:
        print(f"⚠️ [REFINER ERROR]: {e}")
        return {"master_bundle": master}
