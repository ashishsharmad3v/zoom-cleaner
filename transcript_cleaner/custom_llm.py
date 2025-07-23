from typing import Any, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import openai
import json
import logging
from config.settings import OPENAI_API_KEY, DEFAULT_MODEL

logger = logging.getLogger(__name__)

class CustomOpenAILLM(LLM):
    """Custom LLM implementation using OpenAI API"""
    
    model_name: str = DEFAULT_MODEL
    temperature: float = 0.1
    max_tokens: int = 2000
    openai_api_key: str = OPENAI_API_KEY
    
    @property
    def _llm_type(self) -> str:
        return "custom_openai"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the LLM call"""
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.openai_api_key
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise e
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def process_json_request(self, prompt: str, max_tokens: int = None) -> dict:
        """Process a request that expects JSON response"""
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        original_max_tokens = self.max_tokens
        self.max_tokens = max_tokens
        
        try:
            response_text = self._call(prompt)
            # Try to parse as JSON
            try:
                return {
                    "success": True,
                    "parsed": json.loads(response_text),
                    "raw": response_text
                }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Failed to parse JSON response",
                    "raw": response_text
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw": ""
            }
        finally:
            # Restore original max_tokens
            self.max_tokens = original_max_tokens
    
    def batch_process(self, prompts: List[str]) -> List[str]:
        """Process multiple prompts in batch"""
        results = []
        for prompt in prompts:
            try:
                result = self._call(prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing failed for prompt: {e}")
                results.append("")
        return results