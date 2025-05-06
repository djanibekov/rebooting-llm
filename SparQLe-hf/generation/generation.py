import os
import math
import string
import argparse
import glob
from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss

from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor, Qwen2ForCausalLM
from qformer import BertConfig, BertLMHeadModel # Assuming qformer.py is in the same directory or PYTHONPATH
from transformers import BertTokenizer
from transformers.modeling_outputs import ModelOutput

from transformers.feature_extraction_utils import BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.configuration_utils import PretrainedConfig

from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

from s3prl.nn import S3PRLUpstream # Assuming s3prl is installed

from torch.nn import functional as F
# from functions import ReverseLayerF # Not used in generation path
from torch.autograd import Variable

import logging

from dataclasses import dataclass, field
from typing import *

from loguru import logger

# Constants from training script
MAXDURATION=30
SAMPLINGRATE=16000
QFORMERBASE = 'google-bert/bert-base-multilingual-uncased'
PROMPT = '{target_lang}:{target_lang}:' 

@dataclass
class Config(PretrainedConfig):
    hidden_size: Optional[int] = None
    num_query_token: Optional[int] = None
    cross_attention_freq: Optional[int] = None
    bottleneck_dim: Optional[int] = None
    framework: Optional[str] = None
    encoder_name: Optional[str] = None

class SpeechPrenet(SequenceFeatureExtractor):
    model_input_names = ["input_features"]
    def __init__(self, sampling_rate, feature_extractor, max_duration=MAXDURATION):
        super().__init__(
            feature_size=80,
            sampling_rate=sampling_rate,
            padding_value=0,
            return_attention_mask=True,
        )
        self.n_samples = max_duration * sampling_rate
        self.feature_extractor = feature_extractor

        self.feature_size=128
        self.sampling_rate=16000
        self.hop_length=160
        self.chunk_length=30
        self.n_fft=400
        self.padding_value=0.0
        self.dither=0.0

        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def _np_extract_fbank_features(self, waveform_batch: np.array, device: str = 'cpu') -> np.ndarray:
        
        """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
        if device != "cpu":
            raise ValueError(
                f"Got device `{device}` for feature extraction, but feature extraction on CUDA accelerator "
                "devices requires torch, which is not installed. Either set `device='cpu'`, or "
                "install torch according to the official instructions: https://pytorch.org/get-started/locally/"
            )
        log_spec_batch = []
        for waveform in waveform_batch:
            log_spec = spectrogram(
                waveform,
                window_function(self.n_fft, "hann"),
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                # dither=self.dither,
                mel_filters=self.mel_filters,
                log_mel="log10",
            )
            log_spec = log_spec[:, :-1]
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            log_spec_batch.append(log_spec)
        log_spec_batch = np.array(log_spec_batch)
        return log_spec_batch

    def _torch_extract_fbank_features(self, waveform: np.array, device: str = "cpu") -> np.ndarray:
        """
        Compute the log-mel spectrogram of the audio using PyTorch's GPU-accelerated STFT implementation with batching,
        yielding results similar to cpu computing with 1e-5 tolerance.
        """
        waveform = torch.from_numpy(waveform).to(device, torch.float32)
        window = torch.hann_window(self.n_fft, device=device)

        # Note: it would be better to dither the chunked waveform,
        # so overlapping signal does not get the same dithering.
        # But, chunking is happening inside pytorch, so it is here.
        if self.dither != 0.0:
            waveform += self.dither * torch.randn(waveform.shape, dtype=waveform.dtype, device=waveform.device)

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        if device != "cpu":
            log_spec = log_spec.detach().cpu()
        return log_spec.numpy()
        
        
    def pre_process(self, raw_speech):
        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = BatchFeature({"input_features": raw_speech})

        padded_inputs = self.pad(
            batched_speech,
            padding="max_length",
            max_length=self.n_samples,
            truncation=True,
            pad_to_multiple_of=None,
            return_attention_mask=True,
        )

        input_features = padded_inputs.get("input_features").transpose(2, 0, 1)
        if self.feature_extractor == 'whisper':
            extract_fbank_features = self._np_extract_fbank_features
            input_features = extract_fbank_features(input_features[0])
            padded_inputs["attention_mask"] = padded_inputs["attention_mask"][:, :: self.hop_length]


        padded_inputs["input_features"] = input_features
        # padded_inputs["attention_mask"] = padded_inputs["attention_mask"]
        padded_inputs = padded_inputs.convert_to_tensors('pt')
        
        return padded_inputs["input_features"], padded_inputs["attention_mask"]

    def post_process(self, input_features, attention_mask):

        return input_features, attention_mask

    def __call__(self, *args, **kwds):
        return self.pre_process(*args, **kwds)


class SparQLe(nn.Module):
    @classmethod
    def init_Qformer(cls, num_query_token, speech_width, cross_attention_freq=2):
        # Use AutoConfig to ensure flexibility if QFORMERBASE changes
        encoder_config = BertConfig.from_pretrained(QFORMERBASE)
        encoder_config.encoder_width = speech_width
        # encoder_config.is_decoder = True # is_decoder for BertLMHeadModel is usually False? Check QFormer implementation details
        encoder_config.is_decoder = False # Let's assume standard BERT encoder for Qformer base
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        # Instantiate BertLMHeadModel FOR QFormer structure, might need custom class if specific BertForMaskedLM or similar needed
        Qformer = BertLMHeadModel.from_pretrained(
            QFORMERBASE, config=encoder_config, ignore_mismatched_sizes=True # Ignore size mismatch due to width change etc.
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_encoder(cls, config):
        encoder = None
        if config.framework == 'huggingface':
            encoder = AutoModel.from_pretrained(config.encoder_name)
        elif config.framework == 'whisper':
            from transformers import WhisperModel
            # Load only the encoder part
            encoder = WhisperModel.from_pretrained(config.encoder_name).encoder
        elif config.framework == 's3prl':
            encoder = S3PRLUpstream(config.encoder_name)
        else:
            raise NotImplementedError(f"Framework {config.framework} not supported for encoder loading.")

        # Freeze encoder - crucial for inference consistency if trained frozen
        if hasattr(encoder, 'parameters'):
            for param in encoder.parameters():
                param.requires_grad = False
            encoder = encoder.eval()
            logger.info(f"Loaded and froze encoder: {config.encoder_name}")
        return encoder

    def __init__(self, config, llm_backbone_path):
        super(SparQLe, self).__init__()
        self.config = config

        self.speech_encoder = self.init_encoder(config)
        qformer_hidden_size = BertConfig.from_pretrained(QFORMERBASE).hidden_size # Get base QFormer size
        self.Qformer, self.query_tokens = self.init_Qformer(
            config.num_query_token, config.bottleneck_dim, config.cross_attention_freq
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.speech_proj_adapter = nn.Linear(1280, qformer_hidden_size)

        # LLM components
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_backbone_path, use_fast=False, trust_remote_code=True)
        if self.llm_tokenizer.pad_token is None:
             logger.warning("LLM tokenizer does not have a pad token. Setting to eos_token.")
             self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm_model = Qwen2ForCausalLM.from_pretrained(llm_backbone_path, trust_remote_code=True)
        self.llm_proj_layer_norm = nn.LayerNorm(self.llm_model.config.hidden_size)

        # Freeze LLM - crucial for inference consistency if trained frozen
        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.llm_model = self.llm_model.eval()
        logger.info(f"Loaded and froze LLM: {llm_backbone_path}")

        self.llm_proj = nn.Linear(
            qformer_hidden_size, self.llm_model.config.hidden_size
        )

        self.prompt_format = PROMPT

    def get_speech_context_embeddings(self, speech_inputs, inputs_attention_mask):
        encoder_output = self.speech_encoder(speech_inputs, attention_mask=inputs_attention_mask)

        if isinstance(encoder_output, tuple): # Common for S3PRL or older HF?
            speech_embeds = encoder_output[0] # Assume first element is hidden states
        elif hasattr(encoder_output, 'last_hidden_state'): # Standard HF output
            speech_embeds = encoder_output.last_hidden_state
        elif torch.is_tensor(encoder_output): # Direct tensor output?
            speech_embeds = encoder_output
        else:
            raise TypeError(f"Unexpected speech encoder output type: {type(encoder_output)}")

        speech_embeds_proj = self.speech_proj_adapter(speech_embeds)

        batch_size, seq, _ = speech_embeds.shape
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        if hasattr(self.speech_encoder, '_get_feat_extract_output_lengths'):
            input_lengths = self.speech_encoder._get_feat_extract_output_lengths(inputs_attention_mask.sum(-1)).to(torch.long)
            positions = torch.arange(seq, device=speech_inputs.device).expand(batch_size, seq)
            speech_attn = positions < input_lengths.unsqueeze(1)
        # # elif self.config.framework == 's3prl':
        #      # S3PRL models often return lengths or masks, need to check specific model
        #      # Placeholder: Assume mask corresponds to sequence length directly if method above fails
        #      # input_lengths = speech_attn_mask_from_encoder.sum(-1) # If encoder returns its own mask
        #      # Or derive from non-padded features if possible
        #      # Fallback: Use the projected embedding sequence length
        #      # logger.warning("Cannot determine precise encoder output lengths automatically. Using full sequence length for Q-Former mask.")
        #      # input_lengths = torch.full((batch_size,), speech_embeds_proj.shape[1], device=device)
        else:
            speech_attn_mask = inputs_attention_mask
            if speech_attn_mask.shape[1] != speech_embeds_proj.shape[1]:
                logger.warning(f"Attention mask length {speech_attn_mask.shape[1]} doesn't match encoded sequence length {speech_embeds_proj.shape[1]}. QFormer mask might be inaccurate.")
                speech_attn_mask = torch.ones(speech_embeds_proj.shape[:2], device=speech_embeds_proj.device, dtype=torch.long)
            else:
                speech_attn_mask = speech_attn_mask.long() # Ensure long type

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds_proj,
            encoder_attention_mask=speech_attn,
            use_cache=True,
            return_dict=True,
        )
        qformer_output_embeds = query_output.last_hidden_state

        inputs_llm = self.llm_proj(qformer_output_embeds)
        inputs_llm = self.llm_proj_layer_norm(inputs_llm) # Apply LayerNorm

        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long, device=inputs_llm.device)

        return inputs_llm, atts_llm

    @torch.no_grad()
    def generate(
        self,
        speech_inputs,
        inputs_attention_mask,
        target_lang: str,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.2,
        num_beams: int = 1, # Default to greedy search
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device # Get model's device
        speech_inputs = speech_inputs.to(device)
        inputs_attention_mask = inputs_attention_mask.to(device)
        
        inputs_llm, atts_llm = self.get_speech_context_embeddings(
            speech_inputs, inputs_attention_mask
        )

        prompt = self.prompt_format.format(target_lang=target_lang)
        
        prompt_tokens = self.llm_tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
        prompt_input_ids = prompt_tokens.input_ids
        prompt_attention_mask = prompt_tokens.attention_mask

        
        prompt_embeds = self.llm_model.model.embed_tokens(prompt_input_ids)

        
        inputs_embeds = torch.cat([inputs_llm, prompt_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, prompt_attention_mask], dim=1)


        # gen_outputs = self.llm_model.generate(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     repetition_penalty=repetition_penalty,
        #     length_penalty=1.0,
        #     no_repeat_ngram_size=3,
        #     num_beams=num_beams,
        #     do_sample=do_sample,
        #     top_k=top_k,
        #     top_p=top_p,
        #     temperature=temperature,
        #     pad_token_id=self.llm_tokenizer.pad_token_id,
        #     eos_token_id=self.llm_tokenizer.eos_token_id,
        # )

        gen_outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_length=5,
            num_beams=num_beams,
            eos_token_id=self.llm_tokenizer.eos_token_id,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            repetition_penalty=1.0,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )

        # prompt_len = prompt_input_ids.shape[1]
        # generated_ids = gen_outputs[:, inputs_llm.shape[1]:]
        generated_text = self.llm_tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

        return generated_text



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Framework: {args.framework}, Encoder: {args.encoder_name}")
    encoder_output_dim = 768# Default for other HF/S3PRL? Adjust as needed!

    config = Config(
        hidden_size=768, # Base BERT size often used for QFormer base
        num_query_token=100, # Must match training
        cross_attention_freq=2, # Must match training
        bottleneck_dim=encoder_output_dim, # Critical: Output dim of the specific speech_encoder used
        framework=args.framework,
        encoder_name=args.encoder_name,
    )
    logger.info(f"Model Config: {config}")
    model = SparQLe(config, args.llm_backbone)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    model_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys during state dict loading: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys during state dict loading: {unexpected_keys}")

    model.to(device)
    model.eval()

    speech_preprocessor = SpeechPrenet(
        sampling_rate=SAMPLINGRATE,
        feature_extractor=args.framework
    )

    audio_files = []
    for ext in ('*.wav', '*.flac', '*.mp3', '*.opus'): 
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, '**', ext), recursive=True))

    if not audio_files:
        logger.error(f"No audio files found in directory: {args.audio_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files to process.")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for audio_path in tqdm(audio_files, desc="Generating"):
            try:
                audio, sr = sf.read(audio_path)
                if audio.ndim > 1:
                    logger.warning(f"Audio {os.path.basename(audio_path)} has multiple channels ({audio.shape[1]}), converting to mono.")
                    audio = librosa.to_mono(audio.T) # librosa needs (channels, samples)
                if sr != SAMPLINGRATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLINGRATE)

                audio = audio.astype(np.float32)
                inputs, attention_mask = speech_preprocessor(audio) # Expects single waveform or list

                generated_texts = model.generate(
                    speech_inputs=inputs.to(device), # Ensure device placement
                    inputs_attention_mask=attention_mask.to(device),
                    target_lang=args.target_lang,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature
                )

                result_text = generated_texts[0].strip()
                print(f"{audio_path}\t{result_text}\n")

                outfile.write(f"{audio_path}\t{result_text}\n")

            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                outfile.write(f"{audio_path}\tERROR: {e}\n")

    logger.info(f"Generation complete. Results saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from audio using SparQLeFineTune model.")

    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (.pt or .pth).")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files to process.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the tab-separated output (audio_path\\tgenerated_text).")
    parser.add_argument("--target_lang", type=str, required=True, help="Target language for generation (e.g., 'english', 'german', 'french'). Determines the prompt.")

    parser.add_argument("--llm_backbone", type=str, required=True, help="Identifier for the LLM backbone model (e.g., 'Qwen/Qwen2-7B-Instruct').")
    parser.add_argument("--encoder_name", type=str, required=True, help="Identifier for the speech encoder model (e.g., 'openai/whisper-large-v2').")
    parser.add_argument("--framework", type=str, required=True, choices=['whisper', 'huggingface', 's3prl'], help="Framework used for the speech encoder.")

    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty factor (1.0 means no penalty).")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search (1 means greedy search).")
    parser.add_argument("--do_sample", action='store_true', help="Whether to use sampling; otherwise greedy decoding.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")

    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True) # Integrate loguru with tqdm

    main(args)