import json
from src.utils.llm import get_llm
from src.schemas import FHIRBundle


async def refine_data_node(state: dict):
    """
    Refiner Node: Performs semantic deduplication and record normalization.
    Includes a protection layer to prevent loss of longitudinal data.
    """
    llm = get_llm()
    master = state.get("master_bundle", {})
    if not master.get("observations") and not master.get("medications"):
        print("⚠️ [REFINER] Input bundle is empty. Skipping refinement.")
        return {"master_bundle": master}

    prompt = """
### CLINICAL AUDIT PROTOCOL
1. DEDUPLICATION: Merge medications with same name and dose. 
2. CHRONOLOGY: Do NOT delete duplicate Lab Test names if they have different dates.
3. IDENTITY: Ensure Patient Name is 'FORLANELLI MAURIZIO' if clear.

"""
    structured_llm = llm.with_structured_output(FHIRBundle)
    refined = await structured_llm.ainvoke(f"{prompt}\n\nDATA:\n{json.dumps(master)}")
    return {"master_bundle": refined.model_dump()}
