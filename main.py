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
MODEL_NAME = "llama2"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONTEXT_LENGTH = 4096

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
    """Install Ollama if not already installed."""
    console.print("[bold yellow]Installing Ollama...[/bold yellow]")
    try:
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", 
                      shell=True, 
                      check=True)
        console.print("[bold green]Ollama installed successfully![/bold green]")
        return True
    except subprocess.CalledProcessError:
        console.print("[bold red]Failed to install Ollama.[/bold red]")
        return False

def check_model_exists():
    """Check if the Llama 2 model is already pulled."""
    try:
        models = ollama.list()
        for model in models['models']:
            if model['name'] == MODEL_NAME:
                return True
        return False
    except Exception as e:
        console.print(f"[bold red]Error checking models: {e}[/bold red]")
        return False

def pull_model():
    """Pull the Llama 2 model."""
    console.print(f"[bold yellow]Pulling {MODEL_NAME} model... This may take a while.[/bold yellow]")
    try:
        # Stream progress to console
        for progress in ollama.pull(MODEL_NAME, stream=True):
            if 'completed' in progress and 'total' in progress:
                percentage = (progress['completed'] / progress['total']) * 100
                console.print(f"Download progress: [bold green]{percentage:.2f}%[/bold green]", end="\r")
        console.print(f"\n[bold green]{MODEL_NAME} model pulled successfully![/bold green]")
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
    """Start a conversation with the Llama 2 model."""
    conversation_history = []
    
    console.print("\n[bold green]Starting conversation with Llama 2. Type 'exit' to end.[/bold green]")
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
                model=MODEL_NAME,
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
    
    # Check if model exists and pull if necessary
    if not check_model_exists():
        console.print(f"[yellow]The {MODEL_NAME} model is not downloaded yet.[/yellow]")
        if Confirm.ask(f"Do you want to pull the {MODEL_NAME} model now?"):
            if not pull_model():
                console.print("[bold red]Failed to pull the model. Please try again later.[/bold red]")
                return
        else:
            console.print("[bold yellow]The model is required to run this application. Exiting...[/bold yellow]")
            return
    else:
        console.print(f"[bold green]{MODEL_NAME} model is already downloaded.[/bold green]")
    
    # Get conversation settings
    settings = get_conversation_settings()
    
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
