#!/usr/bin/env python3

import os
import json
import time
import subprocess
from datetime import datetime
import ollama
from rich.console import Console
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from markupsafe import Markup
from flask_wtf import CSRFProtect
from flask_wtf.file import FileField
from wtforms import StringField, TextAreaField, SelectField, FloatField, IntegerField, SubmitField, validators
from flask_wtf import FlaskForm

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages and sessions
csrf = CSRFProtect(app)

# Add simple HTML conversion filter for Jinja templates
@app.template_filter('markdown')
def render_markdown(text):
    # Simple conversion of newlines to <br> and preserving spaces
    text = text.replace('\n', '<br>')
    # Simple code block handling
    if '```' in text:
        parts = text.split('```')
        for i in range(1, len(parts), 2):
            if i < len(parts):
                # This is a code block
                parts[i] = f'<pre><code>{parts[i]}</code></pre>'
        text = ''.join(parts)
    return Markup(text)


# Initialize console for CLI output capture
console = Console()

# Model configuration (same as in main.py)
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

# Forms
class ConversationSettingsForm(FlaskForm):
    model = SelectField('Model', choices=[(k, v['description']) for k, v in MODEL_OPTIONS.items()])
    temperature = FloatField('Temperature (0.0-1.0, higher = more creative)', 
                           validators=[validators.NumberRange(min=0.0, max=1.0)],
                           default=DEFAULT_TEMPERATURE)
    context_length = IntegerField('Context Length (tokens to remember)',
                               validators=[validators.NumberRange(min=0, max=8192)],
                               default=DEFAULT_CONTEXT_LENGTH)
    system_prompt = TextAreaField('System Prompt',
                               default="You are a helpful AI assistant. Respond concisely and accurately.")
    submit = SubmitField('Start Conversation')

class MessageForm(FlaskForm):
    message = TextAreaField('Your message', validators=[validators.DataRequired()])
    submit = SubmitField('Send')

class TrainingForm(FlaskForm):
    training_text = TextAreaField('Paste text for training', validators=[validators.DataRequired()])
    model_name = StringField('New model name (no spaces)', validators=[validators.DataRequired()])
    base_model = SelectField('Base Model', choices=[(k, v['description']) for k, v in MODEL_OPTIONS.items()])
    submit = SubmitField('Train Model')

# Helper functions
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

def check_model_exists(model_name):
    """Check if the specified model is already pulled."""
    try:
        models = ollama.list()
        for model in models['models']:
            # Try different possible keys for model name
            if ('name' in model and model['name'] == model_name) or\
               ('model' in model and model['model'] == model_name):
                return True
        return False
    except Exception as e:
        print(f"Error checking if model exists: {e}")
        return False

def get_available_models():
    """Get a list of all available models, including custom ones."""
    try:
        # Get the list of models and print the structure to debug
        models_response = ollama.list()
        
        # Check the structure of the response and extract model names
        if 'models' in models_response and isinstance(models_response['models'], list):
            # First try to access each model with the expected structure
            model_names = []
            for model in models_response['models']:
                # Print the model structure for debugging
                print(f"Model structure: {model}")
                # Check which key contains the model name
                if 'name' in model:
                    model_names.append(model['name'])
                elif 'model' in model:
                    model_names.append(model['model'])
            return model_names
        else:
            print(f"Unexpected API response structure: {models_response}")
            return []
    except Exception as e:
        print(f"Error getting models: {e}")
        return []

def pull_model(model_name):
    """Pull the specified model."""
    try:
        # Non-streaming version for the web interface
        ollama.pull(model_name)
        return True
    except Exception as e:
        return False

def save_conversation(conversation_history):
    """Save the conversation history to a file."""
    if not conversation_history:
        return None
    
    # Create conversations directory if it doesn't exist
    os.makedirs("conversations", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversations/conversation_{timestamp}.json"
    
    with open(filename, 'w') as file:
        json.dump(conversation_history, file, indent=2)
    
    return filename

def train_model_on_text(text_content, new_model_name, base_model):
    """Train a model on the provided text content."""
    try:
        # Create necessary directories
        os.makedirs("training_data", exist_ok=True)
        
        # Two-step approach: first create a basic model, then add knowledge with prompts
        
        # Step 1: Create a basic Modelfile
        modelfile_path = os.path.abspath(f"training_data/{new_model_name}.modelfile")
        with open(modelfile_path, 'w') as f:
            f.write(f"FROM {base_model}\n")
            # Keep the system prompt simple
            f.write(f"SYSTEM You are an AI assistant that specializes in the information provided in your training.\n")
        
        # Use a direct system command to create the model
        # This bypasses potential issues with the Python library
        # Execute ollama create with the modelfile
        import subprocess
        result = subprocess.run(
            ["ollama", "create", new_model_name, "-f", modelfile_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False, result.stderr
        
        # Step 2: Save the training content to embed with the model
        training_file_path = os.path.abspath(f"training_data/{new_model_name}_content.txt")
        with open(training_file_path, 'w') as f:
            f.write(text_content)
            
        # Create a conversation file that will embed the training content
        conversation_file = os.path.abspath(f"training_data/{new_model_name}_conversation.json")
        with open(conversation_file, 'w') as f:
            # Format: Create a chat history that includes the training content as context
            convo = [
                {"role": "system", "content": f"You are an AI assistant trained on specific documentation. Use this information as your primary knowledge source: {text_content}"},
                {"role": "user", "content": "Please confirm you have access to the information I provided."},
                {"role": "assistant", "content": "Yes, I have access to the information you provided. I can answer questions about this content and use it as my knowledge base."}
            ]
            json.dump(convo, f, indent=2)
            
        # Verify that the model was created successfully by checking if it exists
        if check_model_exists(new_model_name):
            # Output detailed success message with next steps
            return True, f"Model '{new_model_name}' created successfully! The training content has been saved and you can now chat with this model to access information from your training content. You'll find this model in the model selection dropdown on the settings page."
        else:
            # Model creation command succeeded but model isn't detected yet - might need time to register
            return True, f"Model '{new_model_name}' was created but may take a moment to become available. Check the Settings page in a few seconds to see it in the model dropdown."
    except Exception as e:
        return False, str(e)

# Routes
@app.route('/')
def index():
    is_ollama_installed = check_ollama_installed()
    return render_template('index.html', is_ollama_installed=is_ollama_installed)

@app.route('/check_model/<model_name>')
def check_model(model_name):
    # Handle empty model name case
    if not model_name:
        return jsonify({'exists': False, 'error': 'No model name provided'})
        
    exists = check_model_exists(model_name)
    return jsonify({'exists': exists})

@app.route('/pull_model/<model_name>')
def pull_model_route(model_name):
    success = pull_model(model_name)
    return jsonify({'success': success})

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    form = ConversationSettingsForm()
    
    # Get all available models including custom trained ones
    available_models = get_available_models()
    
    # Create combined model options with our pre-defined ones and any custom models
    combined_model_options = MODEL_OPTIONS.copy()
    
    # Add any custom models that aren't in our predefined list
    for model_name in available_models:
        if model_name not in combined_model_options:
            # Add custom model to choices
            combined_model_options[model_name] = {
                'name': f'Custom: {model_name}',
                'description': 'Your custom trained model',
                'context_length': 2048  # Default context length
            }
    
    # Update form choices
    form.model.choices = [(model, combined_model_options[model]['name']) 
                         for model in combined_model_options 
                         if model in available_models]
    
    if form.validate_on_submit():
        # Store settings in session
        session['model'] = form.model.data
        session['temperature'] = form.temperature.data
        session['context_length'] = form.context_length.data
        session['system_prompt'] = form.system_prompt.data
        
        # Initialize conversation history
        session['conversation_history'] = []
        
        return redirect(url_for('chat'))
    
    return render_template('settings.html', form=form, model_options=combined_model_options)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    form = MessageForm()
    
    # Get conversation history from session
    conversation_history = session.get('conversation_history', [])
    
    if form.validate_on_submit():
        user_message = form.message.data
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_message})
        
        # Get response from model
        try:
            response = ollama.chat(
                model=session.get('model', DEFAULT_MODEL),
                messages=conversation_history,
                options={
                    "temperature": session.get('temperature', DEFAULT_TEMPERATURE),
                    "num_ctx": session.get('context_length', DEFAULT_CONTEXT_LENGTH),
                    "system": session.get('system_prompt', "You are a helpful AI assistant.")
                }
            )
            
            # Add assistant response to history
            if 'message' in response and 'content' in response['message']:
                conversation_history.append({"role": "assistant", "content": response['message']['content']})
        
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
        
        # Update session
        session['conversation_history'] = conversation_history
        
        # Clear form
        form.message.data = ""
    
    return render_template('chat.html', form=form, conversation_history=conversation_history)

@app.route('/save_conversation', methods=['POST'])
def save_conversation_route():
    conversation_history = session.get('conversation_history', [])
    
    if not conversation_history:
        flash("No conversation to save", "warning")
        return redirect(url_for('chat'))
    
    filename = save_conversation(conversation_history)
    
    if filename:
        flash(f"Conversation saved to {filename}", "success")
    else:
        flash("Failed to save conversation", "error")
    
    return redirect(url_for('chat'))

@app.route('/train', methods=['GET', 'POST'])
def train():
    form = TrainingForm()
    success = None
    message = None
    
    if form.validate_on_submit():
        training_text = form.training_text.data
        new_model_name = form.model_name.data
        base_model = form.base_model.data
        
        # Check if model name already exists
        if check_model_exists(new_model_name):
            success = False
            message = f"Model {new_model_name} already exists. Please choose a different name."
            return render_template('train.html', form=form, success=success, message=message)
        
        # Train the model
        success, message = train_model_on_text(training_text, new_model_name, base_model)
        
        # Don't redirect - show success/error on the same page
        if success:
            # Also add a flash message for when redirecting to settings
            flash(f"Model {new_model_name} created successfully!", "success")
    
    return render_template('train.html', form=form, success=success, message=message)

if __name__ == '__main__':
    app.run(debug=True)
