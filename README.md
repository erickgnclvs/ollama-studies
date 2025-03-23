# Ollama Studies

A simple Python application to interact with various AI models locally using Ollama on MacOS, with options for lightweight models suitable for Mac M1.

## Prerequisites

- Python 3.7 or higher
- Ollama installed on your system

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/erickgnclvs/ollama-studies.git
   cd ollama-studies
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Make sure Ollama is installed. For macOS, install Ollama using one of these methods:

   - Download from the official website:
     Visit https://ollama.com/download and download the macOS app

   - Or install via Homebrew (recommended):
     ```
     brew install ollama
     ```

   After installing:
   - Launch the Ollama app from your Applications folder, or
   - If installed via Homebrew, run: `ollama serve`

## Usage

Run the script with:
```
python main.py
```

The interactive menu will guide you through:
- Selecting an appropriate model for your hardware (from lightweight to more powerful options)
- Pulling the selected model (if not already downloaded)
- Setting conversation parameters
- Starting a conversation with the model
- Saving conversation history

## Features

- Interactive CLI menu
- Model selection based on hardware capabilities:
  - TinyLlama (1.1B) - Very lightweight model for minimal resource usage
  - Gemma 2B - Google's lightweight model, good performance/resource ratio
  - Phi-2 (2.7B) - Microsoft's small but capable model
  - Mistral (7B) - Good performance, moderate resource usage
  - Llama 2 (7B) - Full-size model (high resource usage)
- Simple prompt formatting
- Conversation history saving
