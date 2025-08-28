#!/usr/bin/env python3

import os

def compare_versions():
    """Compare OpenAI and Ollama versions of the mocap controller"""
    
    print("OpenAI vs Ollama Mocap Controller Comparison")
    print("=" * 50)
    
    # File comparison
    openai_script = "src/openai_mocap_controller/scripts/openai_mocap_controller.py"
    ollama_script = "src/openai_mocap_controller/scripts/ollama_mocap_controller.py"
    openai_launch = "src/openai_mocap_controller/launch/openai_mocap_controller.launch"
    ollama_launch = "src/openai_mocap_controller/launch/ollama_mocap_controller.launch"
    
    print("\n📁 Files:")
    print(f"OpenAI Script:  {openai_script} ({'✅' if os.path.exists(openai_script) else '❌'})")
    print(f"Ollama Script:  {ollama_script} ({'✅' if os.path.exists(ollama_script) else '❌'})")
    print(f"OpenAI Launch:  {openai_launch} ({'✅' if os.path.exists(openai_launch) else '❌'})")
    print(f"Ollama Launch:  {ollama_launch} ({'✅' if os.path.exists(ollama_launch) else '❌'})")
    
    # Feature comparison
    print("\n🔧 Key Differences:")
    
    features = [
        ("LLM Provider", "OpenAI API (cloud)", "Ollama (local)"),
        ("Internet Required", "Yes", "No (after setup)"),
        ("Cost", "Pay per API call", "Free after setup"),
        ("Privacy", "Data sent to OpenAI", "All data stays local"),
        ("Models", "GPT-3.5, GPT-4, etc.", "Llama, Mistral, CodeLlama, etc."),
        ("Setup Complexity", "API key only", "Server + model download"),
        ("Response Time", "Fast (cloud)", "Varies (hardware dependent)"),
        ("Customization", "Limited to OpenAI models", "Any Ollama model"),
    ]
    
    print(f"{'Feature':<20} {'OpenAI Version':<25} {'Ollama Version':<25}")
    print("-" * 70)
    for feature, openai_val, ollama_val in features:
        print(f"{feature:<20} {openai_val:<25} {ollama_val:<25}")
    
    # Usage comparison
    print("\n🚀 Usage:")
    print("\nOpenAI Version:")
    print("  1. export OPENAI_API_KEY='your-key'")
    print("  2. roslaunch openai_mocap_controller openai_mocap_controller.launch")
    
    print("\nOllama Version:")
    print("  1. ollama serve")
    print("  2. ollama pull llama3.2")
    print("  3. roslaunch openai_mocap_controller ollama_mocap_controller.launch")
    
    # Configuration comparison
    print("\n⚙️  Configuration:")
    print("\nOpenAI Parameters:")
    print("  - openai_api_key: Your OpenAI API key")
    print("  - model_to_use: gpt-3.5-turbo, gpt-4, etc.")
    
    print("\nOllama Parameters:")
    print("  - ollama_server_ip: Server IP address")
    print("  - ollama_server_port: Server port (default 11434)")
    print("  - ollama_model: llama3.2, mistral, codellama, etc.")
    print("  - request_timeout: Request timeout in seconds")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("\nUse OpenAI Version when:")
    print("  ✅ You want the fastest setup")
    print("  ✅ You need the most advanced models (GPT-4)")
    print("  ✅ You don't mind API costs")
    print("  ✅ Internet connectivity is reliable")
    
    print("\nUse Ollama Version when:")
    print("  ✅ You want privacy and local processing")
    print("  ✅ You want to avoid API costs")
    print("  ✅ You need offline capability")
    print("  ✅ You want to experiment with different models")
    print("  ✅ You have sufficient local hardware")
    
    print("\n🔧 Testing:")
    print("Test OpenAI:  python3 src/openai_mocap_controller/test_openai.py")
    print("Test Ollama:  python3 src/openai_mocap_controller/test_ollama.py")
    
    print("\n" + "=" * 50)
    print("Both versions use the same navigation logic and performance tracking!")

if __name__ == "__main__":
    compare_versions()