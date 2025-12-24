from langchain_cerebras import ChatCerebras
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    return ChatCerebras(
        model="gpt-oss-120b",
        cerebras_api_key=os.getenv("CEREBRAS_API_KEY"),
        temperature=0,
        reasoning_effort="high",  # CRITICAL for 1-shot merging
    )
