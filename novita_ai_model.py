"""
Novita AI Model Wrapper for Jahr Chatbot
Integrates Novita's meta-llama/llama-3.2-1b-instruct model
"""

import requests
import json
from typing import Optional, Dict, Any
import time


class NovitaAIWrapper:
    """Wrapper for Novita AI API integration"""
    
    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-3.2-1b-instruct"):
        """
        Initialize Novita AI wrapper
        
        Args:
            api_key: Novita AI API key
            model_name: Model name (default: meta-llama/llama-3.2-1b-instruct)
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Try different possible Novita API endpoints
        self.possible_urls = [
            "https://api.novita.ai/v3/openai/chat/completions",
            "https://api.novita.ai/v1/chat/completions",
            "https://api.novitaai.com/v1/chat/completions"
        ]
        
        # Different auth header formats to try
        self.auth_formats = [
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            {"X-API-Key": api_key, "Content-Type": "application/json"},
            {"api-key": api_key, "Content-Type": "application/json"}
        ]
    
    def generate_response(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.1) -> str:
        """
        Generate response using Novita AI API
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Generated response text
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        # Try different URL and auth combinations
        last_error = None
        
        for url in self.possible_urls:
            for headers in self.auth_formats:
                try:
                    print(f"🔄 Trying {url} with auth format: {list(headers.keys())}")
                    
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            print(f"✅ Success with {url}")
                            # Cache successful combination for future use
                            self.base_url = url
                            self.headers = headers
                            return result['choices'][0]['message']['content'].strip()
                        else:
                            last_error = "No choices in API response"
                    else:
                        last_error = f"API Error {response.status_code}: {response.text}"
                        print(f"❌ {url}: {last_error}")
                        
                except requests.exceptions.Timeout:
                    last_error = "API request timed out"
                    print(f"⏰ Timeout: {url}")
                except requests.exceptions.RequestException as e:
                    last_error = f"Network error: {str(e)}"
                    print(f"🌐 Network error: {url}")
                except json.JSONDecodeError as e:
                    last_error = f"Failed to parse API response: {str(e)}"
                    print(f"📄 JSON error: {url}")
                except Exception as e:
                    last_error = f"Unexpected error: {str(e)}"
                    print(f"❓ Unexpected: {url}")
        
        # If all attempts failed
        raise Exception(f"All API endpoints failed. Last error: {last_error}")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the API connection
        
        Returns:
            Dictionary with test results
        """
        try:
            test_response = self.generate_response(
                "Responde solo 'Conexión exitosa' si puedes leer este mensaje.",
                max_tokens=50
            )
            
            return {
                "status": "success",
                "message": "Connection successful",
                "response": test_response,
                "model": self.model_name
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection failed: {str(e)}",
                "model": self.model_name
            }


def update_ai_model_wrapper():
    """
    Update the existing AIModelWrapper to support Novita AI
    This function modifies the ai_contextual_chatbot.py file
    """
    novita_integration = '''
        elif self.model_type == "novita" and api_key:
            from novita_ai_model import NovitaAIWrapper
            self.client = NovitaAIWrapper(api_key, model_name or "novita/meta-llama/llama-3.2-1b-instruct")
            self.model_name = model_name or "novita/meta-llama/llama-3.2-1b-instruct"'''
    
    novita_response = '''
            elif self.model_type == "novita":
                return self.client.generate_response(prompt, max_tokens, temperature)'''
    
    print("To integrate Novita AI with your existing chatbot:")
    print("1. Add this to the AIModelWrapper.__init__ method after line 74:")
    print(novita_integration)
    print("\n2. Add this to the generate_response method after line 111:")
    print(novita_response)
    print("\n3. Update the convenience function to create Novita chatbot")


# Convenience function for creating Novita AI chatbot
def create_novita_contextual_chatbot(api_key: str, model: str = "novita/meta-llama/llama-3.2-1b-instruct"):
    """
    Create chatbot with Novita AI model
    
    Args:
        api_key: Novita AI API key
        model: Model name
        
    Returns:
        AIContextualChatbot instance configured for Novita AI
    """
    # This will be used after integrating with AIModelWrapper
    try:
        from ai_contextual_chatbot import AIContextualChatbot
        return AIContextualChatbot("novita", api_key, model)
    except ImportError:
        print("Error: ai_contextual_chatbot module not found")
        return None


# Test function
def test_novita_ai(api_key: str):
    """
    Test Novita AI integration
    
    Args:
        api_key: Novita AI API key
    """
    print("🔄 Testing Novita AI connection...")
    
    try:
        novita = NovitaAIWrapper(api_key)
        result = novita.test_connection()
        
        if result["status"] == "success":
            print("✅ Novita AI connection successful!")
            print(f"Model: {result['model']}")
            print(f"Test response: {result['response']}")
            return True
        else:
            print(f"❌ Connection failed: {result['message']}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test the Novita AI integration
    API_KEY = "sk_mRJQJG23UKQk4Z6v-p7BOignQM-3e6huQEaqX7ZRfRo"
    
    if test_novita_ai(API_KEY):
        print("\n🎉 Ready to integrate with your chatbot!")
        print("\nNext steps:")
        print("1. Update ai_contextual_chatbot.py with Novita support")
        print("2. Test with your PDF documents")
        print("3. Enjoy enhanced AI capabilities!")
    else:
        print("\n❌ Please check your API key and try again")
