import json
from typing import List, Optional, Any, Dict
from pydantic import create_model, Field

from src.utils.llm import get_llm
from src.schemas import FHIRBundle, MedicationRequest, Observation, Condition
from trustcall import create_extractor


async def extract_section_worker(
    section_type: str, content: str, file_name: str
) -> Dict[str, Any]:
    """
    Worker node with robust handling for empty LLM responses.
    """
    llm = get_llm()

    model_map = {
        "medications": MedicationRequest,
        "observations": Observation,
        "conditions": Condition,
    }

    TargetModel = model_map.get(section_type)
    if not TargetModel:
        return {section_type: [], "patient": None}

    # Create model with clear defaults to prevent NoneType errors
    WorkerSchema = create_model(
        "WorkerSchema",
        items=(
            List[TargetModel],
            Field(default_factory=list, description=f"List of {section_type}"),
        ),
        patient_name=(Optional[str], Field(None, description="Patient name")),
    )

    structured_llm = llm.with_structured_output(WorkerSchema)

    prompt = f"""
    Extract {section_type} from this Italian medical document.
    
    CLINICAL LOGIC:
    - If section is 'conditions': Extract ONLY diagnoses, chronic diseases, or surgical history (e.g., 'Parkinson', 'Bypass'). 
      NEVER extract lab results (like 'Leucociti') as conditions.
    - If section is 'observations': Extract ONLY lab values, vital signs, or clinical findings with results.
    - If section is 'medications': Extract drug names and dosages.
    
    FILE: {file_name}
    CONTENT:
    {content}
    """

    try:
        res = await structured_llm.ainvoke(prompt)

        # SAFETY CHECK: If LLM returns None or structured_output fails
        if res is None:
            return {section_type: [], "patient": None}

        return {
            section_type: (
                [item.model_dump() for item in res.items] if res.items else []
            ),
            "patient": (
                {"name": res.patient_name}
                if res.patient_name and "John" not in res.patient_name
                else None
            ),
        }
    except Exception as e:
        # This catches the 'NoneType' error you saw
        print(f"[EXTRACTOR ERROR] {section_type} in {file_name}: {e}")
        return {section_type: [], "patient": None}


async def final_trustcall_merger(fragments: list):
    llm = get_llm()
    extractor = create_extractor(
        llm, tools=[FHIRBundle], tool_choice="FHIRBundle", enable_inserts=True
    )

    master_dict = {}
    valid_frags = [
        f for f in fragments if any(v for k, v in f.items() if k != "patient" and v)
    ]

    for frag in valid_frags:
        try:
            res = await extractor.ainvoke(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Update the 'FHIRBundle' tool. Do NOT create keys like 'bundle_1' or 'fhir_bundle_1'. Use the existing 'FHIRBundle' structure.",
                        },
                        {
                            "role": "user",
                            "content": f"Integrate this data: {json.dumps(frag)}",
                        },
                    ],
                    # This key MUST match the class name 'FHIRBundle'
                    "existing": {"FHIRBundle": master_dict} if master_dict else None,
                }
            )
            if res.get("responses"):
                master_dict = res["responses"][0].model_dump()
        except Exception as e:
            print(f"[MERGER ERROR]: {e}")
            continue

    return FHIRBundle(**master_dict) if master_dict else FHIRBundle()
