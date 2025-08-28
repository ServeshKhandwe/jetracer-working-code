#!/usr/bin/env python3

import requests
import json
import sys

def test_ollama_connection(server_ip="192.168.137.54", server_port=11434, model="llama3.2"):
    """Test connection to Ollama server and model availability"""
    
    ollama_url = f"http://{server_ip}:{server_port}"
    
    print(f"Testing Ollama connection to {ollama_url}")
    print("="*50)
    
    try:
        # Test server connection
        print("1. Testing server connection...")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            print("✅ Server is running")
            
            # List available models
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            print(f"📋 Available models: {model_names}")
            
            # Check if requested model is available
            model_available = any(model in name for name in model_names)
            if model_available:
                print(f"✅ Model '{model}' is available")
            else:
                print(f"⚠️  Model '{model}' not found. Available: {model_names}")
                print("   Ollama will try to pull the model automatically when first used")
            
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to connect to Ollama server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is installed and running")
        print("2. Check if the server IP and port are correct")
        print("3. Try: ollama serve (to start the server)")
        return False
    
    try:
        # Test model inference
        print(f"\n2. Testing model '{model}' inference...")
        
        test_request = {
            "model": model,
            "prompt": "Respond with exactly: {'test': 'success'}",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 50
            }
        }
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data.get('response', '').strip()
            print(f"✅ Model responded: {response_text}")
            
            # Check if model can produce JSON-like output
            if '{' in response_text and '}' in response_text:
                print("✅ Model can produce structured output")
            else:
                print("⚠️  Model response doesn't contain JSON structure")
                
        else:
            print(f"❌ Model inference failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Model inference test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Ollama is ready for robot navigation.")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    server_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.137.54"
    server_port = int(sys.argv[2]) if len(sys.argv) > 2 else 11434
    model = sys.argv[3] if len(sys.argv) > 3 else "llama3.2"
    
    print(f"Testing Ollama setup:")
    print(f"Server: {server_ip}:{server_port}")
    print(f"Model: {model}")
    print()
    
    success = test_ollama_connection(server_ip, server_port, model)
    
    if success:
        print(f"\n✅ Ready to launch: roslaunch openai_mocap_controller ollama_mocap_controller.launch")
        sys.exit(0)
    else:
        print(f"\n❌ Please fix the issues above before launching the controller")
        sys.exit(1)