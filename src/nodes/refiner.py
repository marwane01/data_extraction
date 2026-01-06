import json
import logging
from src.schemas import FHIRBundle
from src.utils.llm import get_llm

logger = logging.getLogger(__name__)


async def refine_data_node(state: dict):
    llm = get_llm()
    master = state.get("master_bundle", {})

    # Garantiamo che master sia un dizionario serializzabile
    if hasattr(master, "model_dump"):
        master_data = master.model_dump(mode="json")
    else:
        master_data = master

    prompt = f"""
    ### CLINICAL REFINER PROTOCOL
    Revisiona il Master Record clinico per eliminare ogni ridondanza residua.
    1. FARMACI: Unisci molecole identiche con nomi commerciali diversi.
    2. CONDIZIONI: Unisci diagnosi sovrapponibili.
    3. OSSERVAZIONI: NON eliminare osservazioni con lo stesso nome se hanno DATE diverse. 
       Se nome, valore e data sono identici, allora è un duplicato: eliminalo.
    
    DATI: {json.dumps(master_data)}
    """

    structured_llm = llm.with_structured_output(FHIRBundle)
    try:
        refined = await structured_llm.ainvoke(prompt)
        if refined is None:
            return {"master_bundle": master_data}
        return {
            "master_bundle": (
                refined
                if isinstance(refined, dict)
                else refined.model_dump(mode="json")
            )
        }
    except Exception as e:
        logger.error(f"Refinement error: {e}")
        return {"master_bundle": master_data}
