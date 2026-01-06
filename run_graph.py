import asyncio
import json
from src.patient_file_processor_graph import graph


async def run():
    print("🏥 Starting Enterprise Parallel Pipeline (Dual-JSON Mode)...")
    initial_state = {"all_fragments": [], "master_bundle": {}, "file_uris": []}

    final_state = await graph.ainvoke(initial_state)

    # 1. The 'Huge' JSON (Full Longitudinal Record)
    with open("master_record.json", "w", encoding="utf-8") as f:
        f.write(final_state.get("final_json", "{}"))

    # 2. The 'Snapshot' JSON (Agent Context Ready)
    with open("summary_snapshot.json", "w", encoding="utf-8") as f:
        f.write(final_state.get("summary_json", "{}"))

    patient_name = (
        final_state.get("master_bundle", {}).get("patient", {}).get("name", "Unknown")
    )
    patient_obj = final_state.get("master_bundle", {}).get("patient")
    name = "Unknown"
    if isinstance(patient_obj, dict):
        name = patient_obj.get("name", "Unknown")
    elif hasattr(patient_obj, "name"):
        name = patient_obj.name

    print(
        f"\n🏁 DONE!\nPatient: {name}\nFiles generated: master_record.json, summary_snapshot.json"
    )
    print(f"\n🏁 PIPELINE COMPLETE")
    print(f"Patient Identifed: {patient_name}")
    print(f"Generated: master_record.json (FHIR) & summary_snapshot.json (Summary)")


if __name__ == "__main__":
    asyncio.run(run())
