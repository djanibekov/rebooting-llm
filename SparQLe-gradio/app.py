import gradio as gr
import torch
import os
import logging
import tempfile
import sys

# Add the SparQLe-fairseq directory to sys.path to find the generation module
# This assumes app.py is in SparQLe-gradio and SparQLe-fairseq is a sibling directory
fairseq_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SparQLe-fairseq'))
sys.path.insert(0, fairseq_dir)

# Ensure generation.py and Qformer.py are in the SparQLe-fairseq directory accessible in PYTHONPATH
# Qformer.py within SparQLe-fairseq must be the complete version.
from generation import get_model_and_processor, load_and_preprocess_audio

# --- Configuration (User may need to adjust these or use environment variables) ---
SPEECH_ENCODER_MODEL = os.getenv("SPEECH_ENCODER_MODEL", "hubertlarge") # or 'hubertbase'
NUM_QUERY_TOKEN = int(os.getenv("NUM_QUERY_TOKEN", "100"))
CROSS_ATTENTION_FREQ = int(os.getenv("CROSS_ATTENTION_FREQ", "2"))
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
MODEL_CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", None) # IMPORTANT: Set this path or env var
HF_TOKEN = os.getenv("HF_TOKEN") # Needed for gated models like Llama2
CACHE_DIR = os.getenv("CACHE_DIR", None) # Cache for Hugging Face models
HUBERT_CACHE_DIR = os.getenv("HUBERT_CACHE_DIR", os.path.join(CACHE_DIR, "hubert"))
LLAMA_CACHE_DIR = os.getenv("LLAMA_CACHE_DIR", os.path.join(CACHE_DIR, "llama"))


HUBERT_CACHE_DIR = CACHE_DIR
LLAMA_CACHE_DIR = CACHE_DIR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create cache directories if they don't exist
os.makedirs(HUBERT_CACHE_DIR, exist_ok=True)
os.makedirs(LLAMA_CACHE_DIR, exist_ok=True)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global variable for the model ---
model = None

def load_model_globally():
    global model
    if model is None:
        logging.info("Loading model for the first time...")
        try:
            model = get_model_and_processor(
                speech_encoder_model=SPEECH_ENCODER_MODEL,
                num_query_token=NUM_QUERY_TOKEN,
                cross_attention_freq=CROSS_ATTENTION_FREQ,
                llama_model_path=LLAMA_MODEL_PATH,
                model_checkpoint_path=MODEL_CHECKPOINT_PATH if os.path.exists(MODEL_CHECKPOINT_PATH) else None,
                hf_token=HF_TOKEN,
                hubert_cache_dir=HUBERT_CACHE_DIR,
                llama_cache_dir=LLAMA_CACHE_DIR
            )
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            # Propagate error to Gradio UI
            raise gr.Error(f"Failed to load model. Check logs. Ensure Qformer.py is complete and paths are correct. Error: {str(e)}")
    return model

# --- Gradio Interface Function ---
def speech_to_text_action(audio_file_path, task, target_lang, num_beams, max_new_tokens, temperature, top_p, do_sample):
    logging.info(f"Processing audio: {audio_file_path}, Task: {task}, Target Lang: {target_lang}")
    
    if audio_file_path is None:
        return "Error: No audio file provided. Please upload an audio file or use the microphone."

    try:
        # Load the model (ensures it's loaded only once globally)
        current_model = load_model_globally()
        if current_model is None: # Should be caught by load_model_globally, but as a safeguard
            return "Error: Model could not be loaded. Check server logs."

        # Preprocess audio
        waveform = load_and_preprocess_audio(audio_file_path)
        source_tensor = waveform.unsqueeze(0)  # Add batch dimension
        
        # Create a basic padding mask (assuming single, unpadded audio after preprocessing)
        # The `generate` function in generation.py handles conversion to Hubert's expected mask format.
        padding_mask = torch.zeros(source_tensor.shape, dtype=torch.bool) 

        samples = {
            "source": source_tensor.to(device),
            "padding_mask": padding_mask.to(device), 
            "tasks": [task]
        }

        # Perform generation
        with torch.no_grad():
            generated_texts = current_model.generate(
                samples,
                do_sample=do_sample,
                num_beams=int(num_beams),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                target_lang=target_lang if task == "translation" else None
            )
        
        result_text = "\n".join(generated_texts) if generated_texts else "No output generated."
        logging.info(f"Generated text: {result_text}")
        return result_text

    except gr.Error as e: # Catch Gradio specific errors from model loading
        logging.error(f"Gradio Error during processing: {e}", exc_info=True)
        return f"Application Error: {str(e)}"
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
        # Save audio for debugging if it's a temp file
        if hasattr(audio_file_path, 'name') and audio_file_path.name.startswith(tempfile.gettempdir()):
            debug_audio_path = os.path.join(tempfile.gettempdir(), f"error_audio_{os.path.basename(audio_file_path.name)}")
            try:
                os.rename(audio_file_path.name, debug_audio_path)
                logging.info(f"Problematic audio saved to {debug_audio_path} for debugging.")
            except Exception as ex_save:
                logging.error(f"Could not save problematic audio: {ex_save}")
        return f"Error: An unexpected issue occurred during processing. Details: {str(e)}"
    finally:
        # Clean up temporary audio file if it was created by Gradio microphone input
        if audio_file_path and isinstance(audio_file_path, tempfile._TemporaryFileWrapper):
            try:
                if os.path.exists(audio_file_path.name):
                    # os.remove(audio_file_path.name) # Gradio might handle this
                    pass 
            except Exception as e_clean:
                logging.warning(f"Could not clean up temp audio file {audio_file_path.name}: {e_clean}")


# --- Build Gradio Interface ---
css = """
body {font-family: 'Arial', sans-serif;}
.gr-interface {background-color: #f9f9f9;}
.gr-button {background-color: #007bff; color: white;}
.gr-button:hover {background-color: #0056b3;}
footer {display: none !important; visibility: hidden !important;}
"""

title = "SparQLe: Speech-to-Text and Translation Demo"
description = ("""
Upload an audio file or use your microphone to transcribe or translate speech using the SparQLe model.
**Note:** 
1. Ensure `MODEL_CHECKPOINT_PATH` environment variable (or the script variable) points to your SparQLe checkpoint file (e.g., `sparqle_checkpoint.pth`).
2. If using a gated Hugging Face model (like Llama-3), set the `HF_TOKEN` environment variable.
3. The first run might take longer due to model download and setup.
4. **CRITICAL**: `Qformer.py` next to `generation.py` and `app.py` MUST be the complete version from your original codebase.
""")

with gr.Blocks(css=css, title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input Audio")
            task_input = gr.Radio(choices=["transcription", "translation"], value="transcription", label="Task")
            target_lang_input = gr.Textbox(label="Target Language (for translation, e.g., French, German, Spanish)", value="French", visible=False) # Initially hidden
            
            with gr.Accordion("Generation Parameters", open=False):
                num_beams_slider = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Number of Beams")
                max_new_tokens_slider = gr.Slider(minimum=10, maximum=512, value=75, step=1, label="Max New Tokens")
                temperature_slider = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p (nucleus sampling)")
                do_sample_checkbox = gr.Checkbox(value=True, label="Use Sampling (do_sample)")

            run_button = gr.Button("Run Inference", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Output Text", lines=10, interactive=False)

    # Logic to show/hide target_lang_input based on task
    def update_lang_visibility(task_choice):
        if task_choice == "translation":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)
    task_input.change(fn=update_lang_visibility, inputs=task_input, outputs=target_lang_input)

    run_button.click(
        fn=speech_to_text_action,
        inputs=[
            audio_input, task_input, target_lang_input, 
            num_beams_slider, max_new_tokens_slider, 
            temperature_slider, top_p_slider, do_sample_checkbox
        ],
        outputs=output_text,
        api_name="predict" # For Hugging Face Spaces API endpoint
    )
    
    gr.Markdown("### Model Configuration Notes:")
    gr.Markdown(f"- **Speech Encoder**: `{SPEECH_ENCODER_MODEL}`")
    gr.Markdown(f"- **LLaMA Model**: `{LLAMA_MODEL_PATH}`")
    gr.Markdown(f"- **SparQLe Checkpoint**: `{MODEL_CHECKPOINT_PATH if os.path.exists(MODEL_CHECKPOINT_PATH) else 'Not Found (Using base models)'}`")
    gr.Markdown(f"- **Cache Directory**: `{CACHE_DIR}`")
    gr.Markdown("**Remember to provide your own checkpoint and adjust paths/tokens as needed.**")

# Load model once at startup for faster subsequent inferences (optional, but good for Spaces)
# However, Gradio reloads scripts on change. For Spaces, it's better if load happens on first request if state is not kept easily.
# The current `load_model_globally` function handles on-demand loading with a global variable.
# For Hugging Face Spaces, the first call to `speech_to_text_action` will trigger the model load.

demo.queue() # Enable queue for better handling of multiple users
if __name__ == "__main__":
    # Pre-load model when running locally for faster UI startup (optional)
    # print("Attempting to pre-load model locally...")
    # load_model_globally() 
    # print("Model pre-loading attempt finished. Launching Gradio demo...")
    demo.launch() 