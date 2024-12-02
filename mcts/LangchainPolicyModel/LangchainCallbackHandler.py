import logging
from langchain_core.callbacks.base import BaseCallbackHandler
from pprint import pformat

class LoggingCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, chain, inputs, **kwargs):
        logging.info(f"Chain started: {chain.__class__.__name__} with inputs: {pformat(inputs)}")

    def on_chain_end(self, outputs, **kwargs):
        logging.info(f"Chain ended with outputs: {pformat(outputs)}")

    def on_llm_start(self, llm, prompts, **kwargs):
        logging.info(f"LLM started: {llm.__class__.__name__} with prompts: {pformat(prompts)}")

    def on_llm_end(self, result, **kwargs):
        # Extract and prettify only key information
        try:
            generations = [
                {
                    "text": gen.text,
                    "info": {
                        "model": gen.generation_info.get("model", ""),
                    "created_at": gen.generation_info.get("created_at", ""),
                        "done_reason": gen.generation_info.get("done_reason", ""),
                    },
                }
                for gen in result.generations[0]
            ]
            logging.info(f"LLM result:\n{pformat(generations)}")
        except Exception as e:
            logging.error(f"Error formatting LLM result: {e}")
        

    def on_llm_error(self, error, **kwargs):
        logging.error(f"LLM error: {error}")