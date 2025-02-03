# To create new virtual environment
    py -m venv .venv  

# To activate the virtual environment
.venv\Scripts\activate
s
# To install dependencies from requirements.txt file
    pip install -r requirements.txt

# Standalone softwares required to run the app locally
    1.Ollama - Download Ollama from https://ollama.com/download. Once installed Ollama should be running in background
    2. Download the model of your choice - Open cmd prompt, type 'ollama pull llama3.2' to download llama3.2 LLM model. Visit https://ollama.com/search to explore other models like DeepSeek R1.
    3. Download embedded model - Open cmd prompt, type 'ollama pull nomic-embed-text' or Visit https://ollama.com/search, search nomic-embed-text and run the mention command.
