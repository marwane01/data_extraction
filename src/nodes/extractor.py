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
### RUOLO: ARCHITETTO DATI CLINICI
ESTRAI: {section_type.upper()}
SORGENTE: {file_name}
LINGUA: Mantenere i termini medici in ITALIANO.

### ISTRUZIONI PER MATRICI EXCEL (ESAMI SANGUE):
Il documento contiene una griglia temporale. Procedi come segue:
1. IDENTIFICA LE COLONNE: Ogni colonna dopo la seconda rappresenta una DATA (es. 04/11/25, 24/04/25).
2. IDENTIFICA LE RIGHE: Ogni riga rappresenta un PARAMETRO (es. Creatinina, Emoglobina).
3. MAPPA LE CELLE: Per ogni incrocio [Riga x Colonna] che contiene un numero o un valore:
   - Crea un oggetto Observation.
   - test_name = Nome del parametro in riga.
   - value = Valore nella cella.
   - date = Data nell'intestazione di colonna.
   - unit = Unità di misura trovata nella colonna 'UM' o vicino al test.
4. RISULTATO: Devi produrre una lista piatta di tutte le osservazioni trovate. NON saltare le date storiche.

### ISTRUZIONI PER FARMACI:
- Se trovi tabelle con 'Dose' e 'Data Inizio', uniscile nel campo dosage.
- Mantieni lo status 'active' a meno che non sia specificata una data di fine.

### CONTENUTO DA ANALIZZARE:
{content}
"""
    try:
        # We increase the reasoning effort via the LLM config if possible
        res = await structured_llm.ainvoke(prompt)
        return (
            res.model_dump()
            if res
            else {
                "patient": {},
                "medications": [],
                "observations": [],
                "conditions": [],
            }
        )
    except Exception as e:
        logger.error(f"Validation failure in {file_name}: {e}")
        # Return empty structure instead of crashing
        return {"patient": {}, "medications": [], "observations": [], "conditions": []}


async def final_trustcall_merger(fragments: list) -> Dict[str, Any]:
    llm = get_llm()
    extractor = create_extractor(
        llm, tools=[FHIRBundle], tool_choice="FHIRBundle", enable_inserts=True
    )

    master_record = FHIRBundle().model_dump()

    for frag in fragments:
        if not frag:
            continue

        try:
            # Attempt AI-driven merge
            res = await extractor.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Unisci questi dati al record master: {json.dumps(frag)}",
                        }
                    ],
                    "existing": {"FHIRBundle": master_record},
                }
            )
            if res.get("responses"):
                master_record = res["responses"][0].model_dump()
            else:
                raise ValueError("No response from Trustcall")
        except Exception as e:
            logger.warning(f"Trustcall merge failed, using manual fallback: {e}")
            # MANUAL FALLBACK: Append lists to prevent data loss
            for key in ["medications", "observations", "conditions"]:
                if key in frag and isinstance(frag[key], list):
                    master_record[key].extend(frag[key])
            if frag.get("patient") and not master_record["patient"].get("name"):
                master_record["patient"].update(frag["patient"])

    return master_record
