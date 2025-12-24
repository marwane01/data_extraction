from typing import Annotated, List, TypedDict, Dict, Any
import operator


class OverallState(TypedDict):
    # This collects fragments from all 21 workers (7 files x 3 sections)
    all_fragments: Annotated[List[Dict[str, Any]], operator.add]
    master_bundle: Dict[str, Any]
    final_json: str
    final_csv: str


class SectionInput(TypedDict):
    section_type: str
    content: str
    file_name: str
