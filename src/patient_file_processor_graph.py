import os, json, logging, asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from src.state import OverallState
from src.nodes.parser import parse_medical_file
from src.nodes.extractor import extract_section_worker, final_trustcall_merger
from src.nodes.refiner import refine_data_node
from src.nodes.synthesis import synthesize_node

logger = logging.getLogger(__name__)


# --- 1. DETECT FILES ---
def file_mapper_edge(state: OverallState):
    file_uris = state.get("file_uris")
    if not file_uris:
        data_dir = "data"
        file_uris = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith((".pdf", ".xlsx", ".xls"))
        ]
    # Send each file to its own independent processing branch
    return [Send("process_file_node", {"file_path": uri}) for uri in file_uris]


# --- 2. UNIFIED FILE PROCESSOR (Parse + Extract 3x) ---
async def process_file_node(state: dict):
    """
    Independent branch for a single file.
    Handles parsing and fanning out extractions internally.
    """
    file_path = state["file_path"]

    # A. Parse the file
    try:
        parsed = await parse_medical_file(file_path)
    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        return {"all_fragments": []}

    # B. Extract 3 sections in parallel (Internal Async Fan-out)
    # This is much faster and safer than fanning out in the graph twice
    tasks = [
        extract_section_worker("medications", parsed.content, parsed.file_name),
        extract_section_worker("observations", parsed.content, parsed.file_name),
        extract_section_worker("conditions", parsed.content, parsed.file_name),
    ]

    results = await asyncio.gather(*tasks)

    # Return fragments to be added to the global 'all_fragments' list
    # operator.add handles the merge automatically
    return {"all_fragments": results}


# --- 3. CONSOLIDATION & SYNTHESIS ---
async def consolidate_node(state: OverallState):
    fragments = state.get("all_fragments", [])
    master_dict = await final_trustcall_merger(fragments)
    return {"master_bundle": master_dict}


def format_node(state: OverallState):
    bundle = state.get("master_bundle") or {}
    return {"final_json": json.dumps(bundle, indent=2, ensure_ascii=False)}


# --- 4. GRAPH ASSEMBLY ---
builder = StateGraph(OverallState)

builder.add_node("process_file_node", process_file_node)
builder.add_node("consolidate", consolidate_node)
builder.add_node("refine", refine_data_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("format", format_node)

# Flow: Start -> (Parallel Files) -> Consolidate -> Refine -> Synthesize -> End
builder.add_conditional_edges(START, file_mapper_edge, ["process_file_node"])
builder.add_edge("process_file_node", "consolidate")
builder.add_edge("consolidate", "refine")
builder.add_edge("refine", "synthesize")
builder.add_edge("synthesize", "format")
builder.add_edge("format", END)

graph = builder.compile()
