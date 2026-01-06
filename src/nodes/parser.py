import os
import logging
from llama_parse import LlamaParse
from src.state import ParsedFileContent

logger = logging.getLogger(__name__)


async def parse_medical_file(file_path: str) -> ParsedFileContent:
    parser = LlamaParse(
        result_type="markdown",
        language="it",
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="openai-gpt4o",
    )

    try:
        documents = await parser.aload_data(file_path)
        full_content = "\n\n".join([doc.text for doc in documents])
        if not full_content.strip():
            logger.warning(f"File {file_path} parsed but content is empty.")
    except Exception as e:
        logger.error(f"Critical Parsing Error for {file_path}: {str(e)}")
        # Fallback content to allow extraction workers to run/fail gracefully
        full_content = ""

    return ParsedFileContent(
        file_name=os.path.basename(file_path), content=full_content
    )
