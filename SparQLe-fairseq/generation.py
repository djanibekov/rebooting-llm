import os
import logging
import random
import contextlib
from argparse import Namespace

import torch
import torch.nn as nn
import torchaudio
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig

# Importing from Qformer.py, expected to be in the same directory (SparQLe-fairseq/)
from Qformer import BertLMHeadModel 


# --- Helper functions & classes ---

def disabled_train(self, mode=True):
    return self

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32)) # Cast to float32 for stability
        return ret.type(orig_type)

# --- SparQLe Model Base ---
class SparQLeBase(nn.Module): # Inherit from nn.Module instead of BaseFairseqModel
    def __init__(self, *args, **kwargs): # Add basic init
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @classmethod
    def init_tokenizer_bert(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    @classmethod
    def init_Qformer_bert(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu") and torch.cuda.is_available()
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def load_from_pretrained_checkpoint(self, checkpoint_path: str):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            raise RuntimeError(f"Checkpoint {checkpoint_path} not found.")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded checkpoint from {checkpoint_path}. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")
        return msg


# --- SparQLe Instruct Model ---
class SparQLeLLMInstruct(SparQLeBase):
    def __init__(self, args): 
        super().__init__()
        self.args = args
        self.speech_encoder_model = args.speech_encoder_model
        self.tokenizer = self.init_tokenizer_bert() 

        if args.speech_encoder_model == 'hubertbase':
            self.speech_encoder, speech_config = self.build_speech_model_hubertbase({
                'freeze_enc': True 
            })
            codebook_size = speech_config.hidden_size
        elif args.speech_encoder_model == 'hubertlarge':
            self.speech_encoder, speech_config = self.build_speech_model_hubertlarge({
                'freeze_enc': True,
                'cache_dir': getattr(args, 'hubert_cache_dir', os.environ.get('CACHE', None))
            })
            codebook_size = speech_config.hidden_size
        else:
            raise NotImplementedError(f"Speech encoder {args.speech_encoder_model} not implemented.")
        
        self.ln_speech = LayerNorm(codebook_size)
        if True: 
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder = self.speech_encoder.eval()
            self.speech_encoder.train = disabled_train
            logging.info("Froze speech encoder.")

        self.Qformer, self.query_tokens = self.init_Qformer_bert(
            num_query_token=args.num_query_token,
            vision_width=codebook_size, 
            cross_attention_freq=args.cross_attention_freq
        )
        self.Qformer.cls = None 
        if hasattr(self.Qformer, 'bert'):
             if hasattr(self.Qformer.bert.embeddings, 'word_embeddings'):
                self.Qformer.bert.embeddings.word_embeddings = None
             if hasattr(self.Qformer.bert.embeddings, 'position_embeddings'):
                self.Qformer.bert.embeddings.position_embeddings = None
             for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            args.llama_model, 
            use_fast=False, 
            token=getattr(args, 'hf_token', os.environ.get('HF_TOKEN', None))
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        self.llama_model = LlamaForCausalLM.from_pretrained(
            args.llama_model, 
            cache_dir=getattr(args, 'llama_cache_dir', os.environ.get('CACHE', None)),
            token=getattr(args, 'hf_token', os.environ.get('HF_TOKEN', None))
        )
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        self.llama_model.eval()

        self.eos_token_id = self.llm_tokenizer("\n", add_special_tokens=False).input_ids[0] 
        
        qformer_hidden_size = self.Qformer.config.hidden_size
        llama_hidden_size = self.llama_model.config.hidden_size
        self.llama_proj = nn.Linear(qformer_hidden_size, llama_hidden_size)

        self.prompts = {
            "transcription": [
                "<Speech><SpeechHere></Speech> Can you transcribe the speech into a written format?",
                "<Speech><SpeechHere></Speech> Listen to the speech and write down its content.",
                "<Speech><SpeechHere></Speech> What is the content of the speech you heard?",
                "<Speech><SpeechHere></Speech> Please write down the transcription of the speech.",
                "<Speech><SpeechHere></Speech> Please transcribe the speech into a written format.",
                "<Speech><SpeechHere></Speech> Write down the content of the speech you heard.",
                "<Speech><SpeechHere></Speech> Can you write down the transcription of the speech?",
                # "<Speech><SpeechHere></Speech> Put the speech into a written format.",
                # "<Speech><SpeechHere></Speech> Please help me to transcribe the speech into a written format.",
                # "<Speech><SpeechHere></Speech> Recognize the content of the speech you heard.",
                # "<Speech><SpeechHere></Speech> Can you recognize what you heard in the speech?",
                # "<Speech><SpeechHere></Speech> Recognize the speech and write it down in a written format.",
                # "<Speech><SpeechHere></Speech> Listen to the speech and recognize its content.",
                # "<Speech><SpeechHere></Speech> Give me the transcription of the speech you heard.",
                # "<Speech><SpeechHere></Speech> Recognize the speech and give me the transcription.",
            ],
            "translation": [
                "<Speech><SpeechHere></Speech> Can you translate the speech into TARGETLANG?",
                "<Speech><SpeechHere></Speech> Please translate the speech you heard into TARGETLANG.",
                "<Speech><SpeechHere></Speech> Listen to the speech and translate it into TARGETLANG.",
                "<Speech><SpeechHere></Speech> Give me the TARGETLANG translation of this speech.",
                # "<Speech><SpeechHere></Speech> Could you please provide a TARGETLANG translation for the speech?",
                # "<Speech><SpeechHere></Speech> Would you be willing to translate the speech into TARGETLANG for me?",
                # "<Speech><SpeechHere></Speech> Would you be able to render the speech in TARGETLANG?",
                # "<Speech><SpeechHere></Speech> Could you assist me in translating the speech into TARGETLANG?",
                # "<Speech><SpeechHere></Speech> Can you help me convert the speech into FTARGETLANGrench text?",
                # "<Speech><SpeechHere></Speech> Please convert the speech into TARGETLANG text.",
            ]
        }
        
        self.to(self.device)
        self.eval()


    @classmethod
    def build_speech_model_hubertbase(cls, cfg_dict):
        speech_encoder = AutoModel.from_pretrained("facebook/hubert-base-ls960")
        if cfg_dict.get('freeze_enc', True):
            for param in speech_encoder.parameters():
                param.requires_grad = False
            speech_encoder.eval()
        return speech_encoder, speech_encoder.config
    
    @classmethod
    def build_speech_model_hubertlarge(cls, cfg_dict):
        speech_encoder = AutoModel.from_pretrained(
            'facebook/hubert-large-ll60k', 
            cache_dir=cfg_dict.get('cache_dir')
        )
        if cfg_dict.get('freeze_enc', True):
            for param in speech_encoder.parameters():
                param.requires_grad = False
            speech_encoder.eval()
        return speech_encoder, speech_encoder.config

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if not prompt:
            return embeds, atts

        batch_size = embeds.shape[0] 
        p_before, p_after = prompt.split("<SpeechHere>")

        p_before_tokens = self.llm_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False
        ).to(embeds.device)
        p_after_tokens = self.llm_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False
        ).to(embeds.device)
        
        p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
        if batch_size > 1 and p_before_embeds.shape[0] == 1:
             p_before_embeds = p_before_embeds.expand(batch_size, -1, -1)
       
        p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
        if batch_size > 1 and p_after_embeds.shape[0] == 1:
            p_after_embeds = p_after_embeds.expand(batch_size, -1, -1)

        p_before_atts = p_before_tokens.attention_mask
        if batch_size > 1 and p_before_atts.shape[0] == 1:
            p_before_atts = p_before_atts.expand(batch_size, -1)
            
        p_after_atts = p_after_tokens.attention_mask
        if batch_size > 1 and p_after_atts.shape[0] == 1:
            p_after_atts = p_after_atts.expand(batch_size, -1)
            
        wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
        wrapped_atts = torch.cat([p_before_atts, atts, p_after_atts], dim=1)
        
        return wrapped_embeds, wrapped_atts

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=True,
        num_beams=30,
        max_new_length=75,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        target_lang='French',
        **kwargs
    ):
        import gc
        
        self.llama_model.eval()
        speech = samples["source"]
        
        if self.speech_encoder_model == 'speechtokenizer':
            speech_embeds, codes = self.speech_encoder.forward_feature(speech.unsqueeze(1), [0])
            speech_embeds = self.ln_speech(speech_embeds[0].transpose(1, 2))
            speech_atts = codes[0] != self.speechtokenizer_padding_idx
            del codes
            
        elif self.speech_encoder_model == 'hubertbase' or self.speech_encoder_model == 'hubertlarge':
            output = self.speech_encoder(speech, attention_mask=samples['padding_mask'])
            speech_embeds = output.last_hidden_state
            speech_embeds = self.ln_speech(speech_embeds)
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech.device)
            del output
        else:
            raise NotImplementedError

        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        
        inputs_llama_query = self.llama_proj(query_output.last_hidden_state)
        atts_llama_query = torch.ones(inputs_llama_query.size()[:-1], dtype=torch.long).to(speech.device)
        
        del query_tokens, query_output, speech_embeds, speech_atts
        torch.cuda.empty_cache()

        system_prompt = 'You are a speech-to-text conversion model. Your tasks include accurately transcribing spoken language and translating audio samples as per user instructions. Please ensure clarity and precision in both transcription and translation processes.'
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": None},
        ]
        input_ids = self.llm_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(speech.device)
        
        batch_size = inputs_llama_query.shape[0]
        bos = torch.ones([batch_size, 1], dtype=torch.int32, device=inputs_llama_query.device) * self.llm_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        
        # terminators = [
        #     self.llm_tokenizer.eos_token_id,
        #     self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]
        beg_input_ids1 = input_ids[:, :-6]
        aft_input_ids1 = input_ids[:, -5:]
        beg_embds = self.llama_model.model.embed_tokens(beg_input_ids1).to(speech.device)
        aft_embds = self.llama_model.model.embed_tokens(aft_input_ids1).to(speech.device)
        
        del input_ids, beg_input_ids1, aft_input_ids1, bos
        torch.cuda.empty_cache()

        unique_tasks = set(samples['tasks'])
        all_results = []
        
        for task in unique_tasks:
            for prompt in self.prompts[task]:
                current_inputs_llama_query = inputs_llama_query.clone()
                current_atts_llama_query = atts_llama_query.clone()
                
                prompt = prompt.replace('TARGETLANG', target_lang)
                current_inputs_llama_query, _ = self.prompt_wrap(
                    current_inputs_llama_query, current_atts_llama_query, prompt
                )
                
                inputs_embeds = torch.cat([
                    bos_embeds, 
                    beg_embds, 
                    current_inputs_llama_query, 
                    aft_embds
                ], dim=1)
                attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(speech.device)

                outputs = self.llama_model.generate(
                    inputs_embeds=inputs_embeds, 
                    attention_mask=attention_mask,
                    num_beams=num_beams,
                    max_new_tokens=max_new_length,
                    min_length=min_length,
                    pad_token_id=self.eos_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2, 
                    early_stopping=True
                )
                
                output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                output_text = [text.strip() for text in output_text]
                all_results.extend(output_text)
                
                # print(output_text[0])
                
                del outputs, output_text, inputs_embeds, attention_mask
                del current_inputs_llama_query, current_atts_llama_query
                torch.cuda.empty_cache()
                gc.collect()

        del inputs_llama_query, atts_llama_query, bos_embeds, beg_embds, aft_embds
        torch.cuda.empty_cache()
        gc.collect()
        # print('#######################################################')
        return all_results


# --- Audio Processing Helper ---
def load_and_preprocess_audio(audio_path, target_sample_rate=16000):
    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception as e:
        logging.error(f"Error loading audio file {audio_path}: {e}")
        raise
    
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)
        sr = target_sample_rate
        
    wav = wav / torch.max(torch.abs(wav)) 
    
    return wav.squeeze(0) 


# --- Main function to load model and run generation (for testing) ---
def get_model_and_processor(
    speech_encoder_model="hubertlarge", 
    num_query_token=32, 
    cross_attention_freq=2, 
    llama_model_path="meta-llama/Llama-2-7b-chat-hf", 
    model_checkpoint_path=None, 
    hf_token=None, 
    hubert_cache_dir=None, 
    llama_cache_dir=None   
    ):

    args = Namespace(
        speech_encoder_model=speech_encoder_model,
        num_query_token=num_query_token,
        cross_attention_freq=cross_attention_freq,
        llama_model=llama_model_path,
        hf_token=hf_token,
        hubert_cache_dir=hubert_cache_dir,
        llama_cache_dir=llama_cache_dir
    )
    
    logging.basicConfig(level=logging.INFO)
    
    model = SparQLeLLMInstruct(args)
    
    if model_checkpoint_path:
        logging.info(f"Loading model weights from: {model_checkpoint_path}")
        model.load_from_pretrained_checkpoint(model_checkpoint_path)
    else:
        logging.warning("No model checkpoint path provided. Using randomly initialized or pretrained Hugging Face weights.")
        
    model.eval()
    return model

if __name__ == '__main__':
    _SPEECH_ENCODER_MODEL = 'hubertlarge' 
    _NUM_QUERY_TOKEN = 32 
    _CROSS_ATTENTION_FREQ = 2
    _LLAMA_MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf" 
    _MODEL_CHECKPOINT_PATH = "/path/to/your/sparqle_finetuned_checkpoint.pth" 
    _AUDIO_FILE_PATH = "/path/to/your/sample_audio.wav" 
    _TASK = "transcription" 
    _TARGET_LANG = "French" 

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        sparqle_model = get_model_and_processor(
            speech_encoder_model=_SPEECH_ENCODER_MODEL,
            num_query_token=_NUM_QUERY_TOKEN,
            cross_attention_freq=_CROSS_ATTENTION_FREQ,
            llama_model_path=_LLAMA_MODEL_PATH,
            model_checkpoint_path=_MODEL_CHECKPOINT_PATH if os.path.exists(_MODEL_CHECKPOINT_PATH) else None
        )
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        logging.error("Please ensure Qformer.py is complete, model paths are correct, and necessary environment variables (like HF_TOKEN for gated models) are set.")
        exit(1)

    if not os.path.exists(_AUDIO_FILE_PATH):
        logging.error(f"Audio file not found: {_AUDIO_FILE_PATH}. Please set _AUDIO_FILE_PATH correctly.")
        exit(1)
        
    try:
        audio_waveform = load_and_preprocess_audio(_AUDIO_FILE_PATH)
        logging.info(f"Loaded audio: {_AUDIO_FILE_PATH}, shape: {audio_waveform.shape}")
    except Exception as e:
        logging.error(f"Failed to load or preprocess audio: {e}")
        exit(1)

    source_tensor = audio_waveform.unsqueeze(0) 
    padding_mask = torch.zeros(source_tensor.shape, dtype=torch.bool) 
    
    samples = {
        "source": source_tensor,
        "padding_mask": padding_mask, 
        "tasks": [_TASK]
    }

    logging.info(f"Starting generation for task: '{_TASK}'...")
    try:
        with torch.no_grad():
            generated_texts = sparqle_model.generate(
                samples,
                num_beams=5,
                max_new_tokens=100, 
                target_lang=_TARGET_LANG if _TASK == "translation" else None
            )
        
        logging.info(f"Generated Output ({_TASK}):")
        for text in generated_texts:
            print(text)
            
    except Exception as e:
        logging.error(f"Error during generation: {e}", exc_info=True)

    logging.info("Generation example finished.") 