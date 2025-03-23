#!/usr/bin/env python3

import os
import json
import time
import subprocess
from datetime import datetime
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich import print as rprint
from pyfiglet import Figlet

# Initialize console
console = Console()

# Model configuration
MODEL_OPTIONS = {
    "tinyllama": {
        "name": "tinyllama",
        "description": "TinyLlama (1.1B) - Very lightweight model, minimal resource usage"
    },
    "gemma:2b": {
        "name": "gemma:2b",
        "description": "Gemma 2B - Google's lightweight model, good performance/resource ratio"
    },
    "phi": {
        "name": "phi",
        "description": "Phi-2 (2.7B) - Microsoft's small but capable model"
    },
    "mistral": {
        "name": "mistral",
        "description": "Mistral (7B) - Good performance, moderate resource usage"
    },
    "llama2": {
        "name": "llama2",
        "description": "Llama 2 (7B) - Full-size model, high resource usage"
    }
}

# Default settings
DEFAULT_MODEL = "tinyllama"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONTEXT_LENGTH = 2048

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_header():
    """Display application header."""
    clear_screen()
    f = Figlet(font='slant')
    header_text = f.renderText('Ollama Studies')
    console.print(f"[bold cyan]{header_text}[/bold cyan]")
    console.print("[bold yellow]Interact with Llama 2 7B locally using Ollama[/bold yellow]\n")

def check_ollama_installed():
    """Check if Ollama is installed on the system."""
    try:
        result = subprocess.run(['which', 'ollama'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              text=True)
        return result.returncode == 0
    except:
        return False

def install_ollama():
    """Show instructions for installing Ollama on macOS."""
    console.print("[bold yellow]Ollama needs to be installed manually on macOS.[/bold yellow]")
    console.print(Panel("""
[bold]Installation Options for macOS:[/bold]

1. Download from the official website:
   Visit https://ollama.com/download and download the macOS app

2. Install via Homebrew (recommended):
   brew install ollama

[bold]After installing:[/bold]
- Launch the Ollama app from your Applications folder
- Or if installed via Homebrew, run: ollama serve

Once Ollama is running, restart this script.
    """, title="Ollama Installation Instructions"))
    return False

def check_model_exists(model_name):
    """Check if the specified model is already pulled."""
    try:
        models = ollama.list()
        for model in models['models']:
            if model['name'] == model_name:
                return True
        return False
    except Exception as e:
        console.print(f"[bold red]Error checking models: {e}[/bold red]")
        return False

def pull_model(model_name):
    """Pull the specified model."""
    console.print(f"[bold yellow]Pulling {model_name} model... This may take a while.[/bold yellow]")
    try:
        # Stream progress to console
        for progress in ollama.pull(model_name, stream=True):
            if 'completed' in progress and 'total' in progress:
                percentage = (progress['completed'] / progress['total']) * 100
                console.print(f"Download progress: [bold green]{percentage:.2f}%[/bold green]", end="\r")
        console.print(f"\n[bold green]{model_name} model pulled successfully![/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]Error pulling model: {e}[/bold red]")
        return False

def get_conversation_settings():
    """Get conversation settings from user."""
    console.print("\n[bold]Conversation Settings[/bold]")
    console.rule()
    
    temperature = Prompt.ask(
        "Temperature (0.0-1.0, higher = more creative)", 
        default=str(DEFAULT_TEMPERATURE)
    )
    try:
        temperature = float(temperature)
        temperature = max(0.0, min(1.0, temperature))  # Clamp between 0.0 and 1.0
    except ValueError:
        temperature = DEFAULT_TEMPERATURE
        console.print(f"[yellow]Invalid input, using default temperature: {temperature}[/yellow]")
    
    context_length = Prompt.ask(
        "Context length (tokens to remember)", 
        default=str(DEFAULT_CONTEXT_LENGTH)
    )
    try:
        context_length = int(context_length)
        context_length = max(0, min(8192, context_length))  # Set reasonable limits
    except ValueError:
        context_length = DEFAULT_CONTEXT_LENGTH
        console.print(f"[yellow]Invalid input, using default context length: {context_length}[/yellow]")
    
    system_prompt = Prompt.ask(
        "System prompt (initial instruction to the model)", 
        default="You are a helpful AI assistant. Respond concisely and accurately."
    )
    
    return {
        "temperature": temperature,
        "num_ctx": context_length,
        "system": system_prompt
    }

def save_conversation(conversation_history):
    """Save the conversation history to a file."""
    if not conversation_history:
        console.print("[yellow]No conversation to save.[/yellow]")
        return
    
    # Create conversations directory if it doesn't exist
    os.makedirs("conversations", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversations/conversation_{timestamp}.json"
    
    with open(filename, 'w') as file:
        json.dump(conversation_history, file, indent=2)
    
    console.print(f"[bold green]Conversation saved to {filename}[/bold green]")

def chat_with_model(settings):
    """Start a conversation with the selected model."""
    conversation_history = []
    model_name = settings['model']
    
    console.print(f"\n[bold green]Starting conversation with {model_name}. Type 'exit' to end.[/bold green]")
    console.print("[bold green]Type 'save' to save the conversation history.[/bold green]")
    console.rule()
    
    console.print(f"[dim]System: {settings['system']}[/dim]\n")
    
    while True:
        # Get user input
        user_input = Prompt.ask("[bold cyan]USER>[/bold cyan]")
        
        # Check for exit command
        if user_input.lower() == 'exit':
            if Confirm.ask("Do you want to save this conversation before exiting?"):
                save_conversation(conversation_history)
            break
        
        # Check for save command
        if user_input.lower() == 'save':
            save_conversation(conversation_history)
            continue
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Get response from model
        try:
            console.print("[bold purple]LLM>[/bold purple] ", end="")
            response_text = ""
            
            # Stream the response
            for chunk in ollama.chat(
                model=model_name,
                messages=conversation_history,
                stream=True,
                options=settings
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    content_chunk = chunk['message']['content']
                    response_text += content_chunk
                    console.print(content_chunk, end="")
            
            console.print()  # New line after response
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
    
    return conversation_history

def select_model():
    """Display model selection menu and return the chosen model."""
    console.print("\n[bold]Model Selection[/bold]")
    console.rule()
    
    console.print("[yellow]Choose a model based on your hardware capabilities:[/yellow]\n")
    
    # Display model options
    for i, (key, model) in enumerate(MODEL_OPTIONS.items(), 1):
        console.print(f"[bold cyan]{i}.[/bold cyan] [bold]{model['name']}[/bold]")
        console.print(f"   {model['description']}")
    
    # Get user choice
    choice = 0
    while choice < 1 or choice > len(MODEL_OPTIONS):
        try:
            choice_input = Prompt.ask("\nSelect a model (1-5)", default="1")
            choice = int(choice_input)
            if choice < 1 or choice > len(MODEL_OPTIONS):
                console.print("[red]Invalid choice. Please select a number between 1 and 5.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
    
    # Get the selected model
    selected_model = list(MODEL_OPTIONS.values())[choice-1]['name']
    console.print(f"\n[bold green]Selected model: {selected_model}[/bold green]")
    
    return selected_model

def main_menu():
    """Display the main menu and handle user selection."""
    display_header()
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        console.print("[bold red]Ollama is not installed on your system.[/bold red]")
        if Confirm.ask("Do you want to install Ollama now?"):
            if not install_ollama():
                console.print("[bold red]Please install Ollama manually and try again.[/bold red]")
                return
        else:
            console.print("[bold yellow]Ollama is required to run this application. Exiting...[/bold yellow]")
            return
    
    # Select model
    selected_model = select_model()
    
    # Check if model exists and pull if necessary
    if not check_model_exists(selected_model):
        console.print(f"[yellow]The {selected_model} model is not downloaded yet.[/yellow]")
        if Confirm.ask(f"Do you want to pull the {selected_model} model now?"):
            if not pull_model(selected_model):
                console.print("[bold red]Failed to pull the model. Please try again later.[/bold red]")
                return
        else:
            console.print("[bold yellow]The model is required to run this application. Exiting...[/bold yellow]")
            return
    else:
        console.print(f"[bold green]{selected_model} model is already downloaded.[/bold green]")
    
    # Get conversation settings
    settings = get_conversation_settings()
    settings['model'] = selected_model
    
    # Start conversation
    conversation_history = chat_with_model(settings)
    
    # Ask to save if not already saved
    if conversation_history and Confirm.ask("Do you want to save this conversation?"):
        save_conversation(conversation_history)
    
    console.print("[bold yellow]Thank you for using Ollama Studies![/bold yellow]")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Program interrupted. Exiting...[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]An error occurred: {e}[/bold red]")
