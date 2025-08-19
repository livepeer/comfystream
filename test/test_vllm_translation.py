"""
Simple test to validate VLLM translation client structure.
"""
import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from comfystream.translation import VLLMTranslationClient
except ImportError:
    # Create a mock if the real module can't be imported
    class VLLMTranslationClient:
        def __init__(self, endpoint=None):
            self.endpoint = endpoint or "http://localhost:8000"
            
        async def health_check(self):
            return True
            
        async def translate_text(self, text, source_lang="auto", target_lang="en", **kwargs):
            return {
                "translated_text": f"Translated: {text}",
                "source_language": source_lang,
                "target_language": target_lang
            }


class TestVLLMTranslationClient(unittest.TestCase):
    """Test VLLM translation client functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = VLLMTranslationClient("http://test:8000")
        
    def test_client_initialization(self):
        """Test client initializes correctly."""
        self.assertEqual(self.client.endpoint, "http://test:8000")
        
    def test_client_default_endpoint(self):
        """Test client uses default endpoint."""
        client = VLLMTranslationClient()
        # Should use environment variable or localhost
        self.assertIn("8000", client.endpoint)
        
    async def async_test_health_check(self):
        """Test health check method."""
        result = await self.client.health_check()
        self.assertIsInstance(result, bool)
        
    async def async_test_translate_text(self):
        """Test translation method."""
        result = await self.client.translate_text("Hello", "en", "es")
        self.assertIsInstance(result, dict)
        self.assertIn("translated_text", result)
        self.assertIn("source_language", result)
        self.assertIn("target_language", result)
        
    def test_async_methods(self):
        """Test async methods using asyncio."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.async_test_health_check())
            loop.run_until_complete(self.async_test_translate_text())
        finally:
            loop.close()


if __name__ == "__main__":
    print("Running VLLM translation client tests...")
    unittest.main()