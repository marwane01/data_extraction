from typing import Annotated, List, TypedDict, Dict, Any
import operator
from pydantic import BaseModel


class ParsedFileContent(BaseModel):
    file_name: str
    content: str


class OverallState(TypedDict):
    file_uris: List[str]
    # This is the secret sauce: it gathers all results from parallel nodes
    all_fragments: Annotated[List[Dict[str, Any]], operator.add]
    master_bundle: Dict[str, Any]
    final_json: str
    summary_json: str
