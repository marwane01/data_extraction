import os
from llama_parse import LlamaParse
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()


class ParsedFileContent(BaseModel):
    file_name: str
    content: str


async def parse_medical_file(file_path: str) -> ParsedFileContent:
    """
    Converts a medical PDF/Excel into Markdown using LlamaParse.
    Optimized for medical tables and complex layouts.
    """
    parser = LlamaParse(
        result_type="markdown",  # Markdown is best for Trustcall/LLM reasoning
        num_workers=4,  # Faster processing for multi-page docs
        verbose=True,
        language="en",
    )

    print(f"[PARSER] Processing: {file_path}")

    # Simple logic to handle both PDF and Excel via LlamaParse
    documents = await parser.aload_data(file_path)

    # Combine all pages into one clinical text block
    full_content = "\n\n".join([doc.text for doc in documents])

    return ParsedFileContent(
        file_name=os.path.basename(file_path), content=full_content
    )
