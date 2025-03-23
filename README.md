# Ollama Studies

A simple Python application to interact with the Llama 2 7B model locally using Ollama on MacOS.

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

3. Make sure Ollama is installed. If not, install it using:
   ```
   curl -fsSL https://ollama.com/install.sh | sh
   ```

## Usage

Run the script with:
```
python main.py
```

The interactive menu will guide you through:
- Pulling the Llama 2 7B model (if not already downloaded)
- Setting conversation parameters
- Starting a conversation with the model
- Saving conversation history

## Features

- Interactive CLI menu
- Conversation with Llama 2 7B model
- Simple prompt formatting
- Conversation history saving
