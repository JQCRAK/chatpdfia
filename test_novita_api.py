"""
Simple test script to verify Novita AI API format
"""

import requests
import json

def test_novita_api():
    """Test different Novita API configurations"""
    
    api_key = "sk-ZJbqpjB/RVuqnmyS3AHDuaxudIDpBE/kQoL0qCYuEaHRS9zHCOb0RhRUv9Bssw4ZPIDAZvLcz4b00zOaW3NfY83UHR+uEdHkS9GdPp7AJg0="
    
    # Different model name formats to try
    model_names = [
        "novita/meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.2-1b-instruct",
        "llama-3.2-1b-instruct",
        "llama-3.2-1b"
    ]
    
    # Different URL patterns
    urls = [
        "https://api.novita.ai/v3/openai/chat/completions",
        "https://api.novita.ai/v3/chat/completions",
        "https://api.novita.ai/openai/v1/chat/completions"
    ]
    
    # Different auth headers
    auth_headers = [
        {"Authorization": f"Bearer {api_key}"},
        {"X-API-Key": api_key},
        {"api-key": api_key},
        {"novita-api-key": api_key}
    ]
    
    for url in urls:
        print(f"\n🔍 Testing URL: {url}")
        
        for auth in auth_headers:
            for model in model_names:
                headers = {**auth, "Content-Type": "application/json"}
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 50
                }
                
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=10)
                    
                    print(f"  📊 Model: {model}")
                    print(f"  🔑 Auth: {list(auth.keys())[0]}")
                    print(f"  📈 Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        print(f"  ✅ SUCCESS!")
                        result = response.json()
                        print(f"  📝 Response: {result}")
                        return True
                    elif response.status_code in [401, 403]:
                        try:
                            error = response.json()
                            print(f"  🔐 Auth error: {error}")
                        except:
                            print(f"  🔐 Auth error: {response.text[:100]}")
                    else:
                        print(f"  ❌ Error: {response.text[:100]}")
                        
                except Exception as e:
                    print(f"  💥 Exception: {str(e)[:100]}")
                
                print("  " + "-" * 50)
    
    return False

if __name__ == "__main__":
    print("🚀 Testing Novita AI API configurations...")
    
    if test_novita_api():
        print("\n🎉 Found working configuration!")
    else:
        print("\n❌ No working configuration found.")
        print("\n💡 Suggestions:")
        print("1. Verify the API key is correct")
        print("2. Check if you need to activate the API key")
        print("3. Ensure you have credits/access to the model")
        print("4. Check Novita AI documentation for exact API format")
