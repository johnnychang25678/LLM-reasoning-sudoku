import json
import logging
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

def parse_json(text: str):
    cleaned_text = text.replace("'", '"')
    cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
    cleaned_text = cleaned_text.strip()
    logger.info(f"Cleaned text: {cleaned_text}")
    try:
        parsed_data = json.loads(cleaned_text)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")

def parse_ai_message(text: AIMessage):
    return parse_json(text.content)