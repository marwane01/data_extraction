import asyncio
from main import app


async def run():
    print("🏥 Starting Enterprise Double-Parallel Pipeline (Hybrid Mode)...")
    initial_state = {"all_fragments": [], "master_bundle": {}}
    final_state = await app.ainvoke(initial_state)

    with open("master_record.json", "w", encoding="utf-8") as f:
        f.write(final_state["final_json"])
    with open("master_labs.csv", "w", encoding="utf-8") as f:
        f.write(final_state["final_csv"])

    name = final_state["master_bundle"].get("patient", {}).get("name", "Unknown")
    print(
        f"\n🏁 DONE!\nPatient: {name}\nFiles generated: master_record.json, master_labs.csv"
    )


if __name__ == "__main__":
    asyncio.run(run())
