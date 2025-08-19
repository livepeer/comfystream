#!/usr/bin/env python3
"""
Example usage of ComfyStream with VLLM translation support.

This script demonstrates how to use the translation API endpoints
when running ComfyStream with the VLLM sidecar container.
"""

import asyncio
import aiohttp
import json
import time


async def test_translation_endpoints():
    """Test translation endpoints."""
    base_url = "http://localhost:8889"
    
    async with aiohttp.ClientSession() as session:
        print("Testing ComfyStream VLLM Translation Integration")
        print("=" * 50)
        
        # Test health endpoint
        print("\n1. Checking translation service health...")
        try:
            async with session.get(f"{base_url}/translate/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✓ Translation service status: {health_data}")
                else:
                    print(f"✗ Health check failed: {response.status}")
                    return
        except Exception as e:
            print(f"✗ Cannot connect to ComfyStream: {e}")
            print("Make sure ComfyStream is running with: docker-compose -f docker/docker-compose.yml up")
            return
        
        # Test single translation
        print("\n2. Testing single text translation...")
        translation_request = {
            "text": "Hello, how are you today?",
            "source_lang": "en",
            "target_lang": "es"
        }
        
        try:
            async with session.post(
                f"{base_url}/translate",
                json=translation_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✓ Original: {translation_request['text']}")
                    print(f"✓ Translated: {result.get('translated_text', 'N/A')}")
                    print(f"✓ Languages: {result.get('source_language')} → {result.get('target_language')}")
                else:
                    error_data = await response.json()
                    print(f"✗ Translation failed: {error_data}")
        except Exception as e:
            print(f"✗ Translation request failed: {e}")
        
        # Test batch translation
        print("\n3. Testing batch translation...")
        batch_request = {
            "texts": [
                "Good morning!",
                "How are you?",
                "See you later!"
            ],
            "source_lang": "en",
            "target_lang": "fr"
        }
        
        try:
            async with session.post(
                f"{base_url}/translate/batch",
                json=batch_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✓ Batch translation results:")
                    for i, (original, translated) in enumerate(zip(batch_request['texts'], result.get('results', []))):
                        if isinstance(translated, dict):
                            print(f"  {i+1}. '{original}' → '{translated.get('translated_text', 'N/A')}'")
                        else:
                            print(f"  {i+1}. '{original}' → Error: {translated}")
                else:
                    error_data = await response.json()
                    print(f"✗ Batch translation failed: {error_data}")
        except Exception as e:
            print(f"✗ Batch translation request failed: {e}")
        
        print("\n" + "=" * 50)
        print("Translation testing completed!")


async def main():
    """Main function."""
    print(__doc__)
    await test_translation_endpoints()


if __name__ == "__main__":
    asyncio.run(main())