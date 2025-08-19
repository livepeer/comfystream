"""
Translation client for communicating with VLLM sidecar container.
"""
import asyncio
import aiohttp
import json
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class VLLMTranslationClient:
    """Client for communicating with VLLM translation service."""
    
    def __init__(self, endpoint: str = None):
        """Initialize the translation client.
        
        Args:
            endpoint: VLLM service endpoint. Defaults to env var VLLM_ENDPOINT or localhost.
        """
        self.endpoint = endpoint or os.getenv('VLLM_ENDPOINT', 'http://localhost:8000')
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            
    async def health_check(self) -> bool:
        """Check if VLLM service is healthy.
        
        Returns:
            True if service is healthy, False otherwise.
        """
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.endpoint}/health", timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"VLLM health check failed: {e}")
            return False
            
    async def translate_text(self, text: str, source_lang: str = "auto", 
                           target_lang: str = "en", **kwargs) -> Dict[str, Any]:
        """Translate text using VLLM service.
        
        Args:
            text: Text to translate
            source_lang: Source language code (default: auto-detect)
            target_lang: Target language code (default: en)
            **kwargs: Additional parameters for the translation model
            
        Returns:
            Dictionary containing translation result and metadata
        """
        if not text or not text.strip():
            return {"translated_text": "", "source_language": source_lang, "target_language": target_lang}
            
        try:
            await self._ensure_session()
            
            # Construct translation prompt
            prompt = self._build_translation_prompt(text, source_lang, target_lang)
            
            # Prepare request payload for OpenAI-compatible API
            payload = {
                "model": "translation-model",
                "messages": [
                    {"role": "system", "content": "You are a helpful translation assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": len(text) * 2 + 100,  # Allow for expansion
                "temperature": 0.1,  # Low temperature for consistent translations
                **kwargs
            }
            
            async with self.session.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=30
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"VLLM API error {response.status}: {error_text}")
                    
                result = await response.json()
                
                # Extract translated text from response
                translated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                return {
                    "translated_text": translated_text.strip(),
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "model_info": result.get("model", "translation-model"),
                    "usage": result.get("usage", {})
                }
                
        except asyncio.TimeoutError:
            logger.error("Translation request timed out")
            raise Exception("Translation request timed out")
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise Exception(f"Translation failed: {str(e)}")
            
    def _build_translation_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Build translation prompt for the model.
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Formatted prompt string
        """
        if source_lang == "auto":
            return f"Translate the following text to {target_lang}:\n\n{text}"
        else:
            return f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}"
            
    async def batch_translate(self, texts: list, source_lang: str = "auto", 
                            target_lang: str = "en", **kwargs) -> list:
        """Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional parameters
            
        Returns:
            List of translation results
        """
        tasks = []
        for text in texts:
            task = self.translate_text(text, source_lang, target_lang, **kwargs)
            tasks.append(task)
            
        return await asyncio.gather(*tasks, return_exceptions=True)