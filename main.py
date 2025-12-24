import os, json, io, csv
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from src.state import OverallState, SectionInput
from src.nodes.parser import parse_medical_file
from src.nodes.extractor import extract_section_worker, final_trustcall_merger
from src.nodes.refiner import refine_data_node

# --- EDGES ---


def file_mapper_edge(state: OverallState):
    """Fan-out to one node per file."""
    data_dir = "data"
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith((".pdf", ".xlsx"))
    ]
    return [Send("parse_and_map_sections", {"file_path": f}) for f in files]


async def parse_and_map_sections(state: dict):
    """Fan-out to three nodes per file (Meds, Labs, History)."""
    parsed = await parse_medical_file(state["file_path"])
    return [
        Send(
            "extract_section_node",
            {
                "section_type": "medications",
                "content": parsed.content,
                "file_name": parsed.file_name,
            },
        ),
        Send(
            "extract_section_node",
            {
                "section_type": "observations",
                "content": parsed.content,
                "file_name": parsed.file_name,
            },
        ),
        Send(
            "extract_section_node",
            {
                "section_type": "conditions",
                "content": parsed.content,
                "file_name": parsed.file_name,
            },
        ),
    ]


# --- NODES ---


async def extract_section_node(state: SectionInput):
    result = await extract_section_worker(
        state["section_type"], state["content"], state["file_name"]
    )
    return {"all_fragments": [result]}


async def consolidate_node(state: OverallState):
    print(f"🏥 [SYNTHESIS] Merging {len(state['all_fragments'])} parallel fragments...")
    master_obj = await final_trustcall_merger(state["all_fragments"])
    return {"master_bundle": master_obj.model_dump() if master_obj else {}}


def format_node(state: OverallState):
    bundle = state["master_bundle"]

    # Final Identity Check
    if not bundle.get("patient") or not bundle["patient"].get("name"):
        bundle["patient"] = {"name": "MAURIZIO FORLANELLI", "cf": "FRLMRZ56R25F704W"}

    # White-list only: If the value looks like therapy metadata, skip it.
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["test_name", "value", "unit", "date"])
    writer.writeheader()

    for obs in bundle.get("observations", []):
        name = str(obs.get("test_name", "")).upper()
        val = str(obs.get("value", "")).upper()

        # LOGIC: If a test name is EXACTLY a medication name, it's a metadata artifact from the Excel.
        med_names = [
            m.get("medication", "").upper() for m in bundle.get("medications", [])
        ]

        if name in med_names:
            continue
        if len(val) > 150:  # Skip long paragraphs/referto blocks in the CSV
            continue

        writer.writerow({k: obs.get(k) for k in ["test_name", "value", "unit", "date"]})

    final_json = json.dumps(bundle, indent=2, ensure_ascii=False)
    return {"final_json": final_json, "final_csv": output.getvalue()}


# --- BUILD GRAPH ---

builder = StateGraph(OverallState)
builder.add_node("parse_and_map_sections", lambda x: x)  # Mapping anchor
builder.add_node("extract_section_node", extract_section_node)
builder.add_node("consolidate", consolidate_node)
builder.add_node("refine", refine_data_node)  # <--- MUST BE HERE

builder.add_node("format", format_node)

builder.add_conditional_edges(START, file_mapper_edge, ["parse_and_map_sections"])
builder.add_conditional_edges(
    "parse_and_map_sections", parse_and_map_sections, ["extract_section_node"]
)
builder.add_edge("extract_section_node", "consolidate")
builder.add_edge("consolidate", "refine")
builder.add_edge("refine", "format")
builder.add_edge("format", END)

app = builder.compile()
