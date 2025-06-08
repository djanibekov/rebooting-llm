import os
import math
import string

import torch
import torchaudio
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss

from dataset_fairseq import CustomDataset
from librispeechdataset import LIBRISPEECH
from torch.utils.data import ConcatDataset

from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor, Qwen2ForCausalLM
from qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput

from transformers.feature_extraction_utils import BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.configuration_utils import PretrainedConfig

from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from datasets import load_dataset

from s3prl.nn import S3PRLUpstream

from torch.nn import functional as F
from functions import ReverseLayerF
from torch.autograd import Variable

import logging

from dataclasses import dataclass
from typing import *

from loguru import logger

MAXDURARION=30
SAMPLINGRATE=16000
QFORMERBASE = 'google-bert/bert-base-multilingual-uncased'
PROMPT = '{target_lang}:'

@dataclass
class SparQleOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None


@dataclass
class Config(PretrainedConfig):
    hidden_size: Optional[int] = None
    num_query_token: Optional[int] = None
    cross_attention_freq: Optional[int] = None
    bottleneck_dim: Optional[int] = None

    framework: Optional[str] = None
    encoder_name: Optional[str] = None

@dataclass
class DataCollatorWithPadding:
    text_pad_int: int
    steps_per_epoch: int
    
    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_data = [item['input'] for item in features]
        target_lang = [item['target_lang'] for item in features]
        source_lang = ['english' for item in features]
        
        source_labels_data = [item['source_labels']['input_ids'].squeeze() for item in features]
        target_labels_data = [item['target_labels']['input_ids'].squeeze() for item in features]

        filepath = [item['filepath']['path'] for item in features]
        idx = [item['idx'] for item in features]

        try:
            padded_source_labels = pad_sequence(source_labels_data, batch_first=True, padding_value=self.text_pad_int)
            padded_target_labels = pad_sequence(target_labels_data, batch_first=True, padding_value=self.text_pad_int)
        except:
            print(filepath)
            print(idx)
            print(list(map(lambda x: x.shape, target_labels_data)))
       
        inputs_attention_mask = [item['attention_mask'] for item in features]
        
        source_attention_mask = (padded_source_labels != self.text_pad_int).long()
        target_attention_mask = (padded_target_labels != self.text_pad_int).long()
        
        return {
            'speech_inputs': torch.stack(input_data).squeeze(),
            'inputs_attention_mask': torch.stack(inputs_attention_mask).squeeze(),
            'source_labels': padded_source_labels,
            'source_labels_attention_mask': source_attention_mask,
            'target_labels': padded_target_labels,
            'target_labels_attention_mask': target_attention_mask,
            'steps_per_epoch': self.steps_per_epoch,
            'target_lang': target_lang,
            'source_lang': source_lang
        }

class CurriculumDataset(Dataset):
    def __init__(self, data, sampling_rate, feature_extractor=None, target_lang=None):
        self.data = data        
        self.texttokenizer = TextPrenetFineTune()
        self.feature_extractor = feature_extractor
        
        self.speechtokenizer = SpeechPrenet(sampling_rate, feature_extractor)
        self.target_lang = target_lang

        self.translator = str.maketrans('', '', string.punctuation)
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_lang = self.data[idx]['target_lang']
        # input_data, input_attention_mask = self.speechtokenizer.pre_process(self.data[idx]['audio']['array'].squeeze())
        try:
            audio, sr = sf.read(self.data[idx]['file'])
        except:
            print(self.data[idx]['file'])
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        input_data, input_attention_mask = self.speechtokenizer.pre_process(audio.squeeze())
        source_text = self.data[idx]['sentence'].lower().translate(self.translator)
        target_text = self.data[idx]['translation'].lower().translate(self.translator)
        
        source_labels = self.texttokenizer(source_text, 'english')
        target_labels = self.texttokenizer(target_text, target_lang)
        
        return {
            'input': input_data,
            'attention_mask': input_attention_mask,
            'source_labels': source_labels,
            'target_labels': target_labels,
            'target_lang': target_lang,
            'source_text': source_text,
            'target_text': target_text,
            'filepath': self.data[idx]['audio'],
            'idx': idx
        }
    
    # def __getitem__(self, idx):
    #     input_data, input_attention_mask = self.speechtokenizer.pre_process(self.data[idx][0].squeeze())
    #     # input_data, input_attention_mask = self.speechtokenizer.post_process(input_data, input_attention_mask)
    #     source_text = self.data[idx][2].lower()
    #     # target_text = self.data[idx][3].lower()
    #     source_labels = self.texttokenizer(source_text)
    #     # target_labels = self.texttokenizer(target_text)
    #     return {
    #         'input': input_data,
    #         'attention_mask': input_attention_mask,
    #         'source_labels': source_labels,
    #         'target_labels': None, # target_labels,
    #         'target_lang': self.target_lang,
    #         'source_text': source_text,
    #         'target_text': None, # target_text,
    #         'filepath': self.data[idx][4]
    #     }


class TextPrenetFineTune(nn.Module):
    def __init__(self, truncation_side='right'):
        super(TextPrenetFineTune, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(os.environ['LLM_BACKBONE'], truncation_side=truncation_side, cache_dir=os.environ['CACHE'])
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        self.prompt = PROMPT
        
    def forward(self, x, lang):
        # return self.tokenizer(f"[DEC]{PROMPT.format(target_lang=lang)}" + x + self.tokenizer.eos_token, return_tensors='pt', max_length=200, padding=True, truncation=True, add_special_tokens=True)
        # return self.tokenizer(f"[DEC]" + x + self.tokenizer.eos_token, return_tensors='pt', max_length=200, padding=True, truncation=True, add_special_tokens=True)
        # return self.tokenizer(x + self.tokenizer.eos_token, return_tensors='pt', max_length=200, padding=True, truncation=True, add_special_tokens=True)
        return self.tokenizer(f"{PROMPT.format(target_lang=lang)}" + x + self.tokenizer.eos_token, return_tensors='pt', max_length=200, padding=True, truncation=True, add_special_tokens=True)


class TextPrenetPreTrain(nn.Module):
    def __init__(self, truncation_side='right'):
        super(TextPrenetPreTrain, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(QFORMERBASE, truncation_side=truncation_side)
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        
    def forward(self, x):
        return self.tokenizer(x, return_tensors='pt', max_length=200, padding=True, truncation=True, add_special_tokens=True)
    

class SpeechPrenet(SequenceFeatureExtractor):
    model_input_names = ["input_features"]
    def __init__(self, sampling_rate, feature_extractor, max_duration=MAXDURARION):
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


class SparQLeFineTune(nn.Module):
    @classmethod
    def init_Qformer(cls, num_query_token, speech_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(QFORMERBASE)
        encoder_config.encoder_width = speech_width
        encoder_config.is_decoder = True
        
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            QFORMERBASE, config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_encoder(cls, config):
        if config.framework == 'huggingface':
            encoder = AutoModel.from_pretrained(config.encoder_name)
            if hasattr(encoder, 'named_parameters'):
                for name, param in encoder.named_parameters():
                    param.requires_grad = False
                encoder = encoder.eval()
                logger.info("freeze encoder")
        elif config.framework == 'whisper':
            from transformers import WhisperModel
            encoder = WhisperModel.from_pretrained(config.encoder_name).encoder
            if hasattr(encoder, 'named_parameters'):
                for name, param in encoder.named_parameters():
                    param.requires_grad = False
                encoder = encoder.eval()
                logger.info("freeze encoder")

        elif config.framework == 's3prl':
            encoder = S3PRLUpstream(config.encoder_name)
            encoder = encoder.eval()
            logger.info("freeze encoder")

    
        else:
            raise NotImplemented
        return encoder

    def __init__(self, config, sampling_rate, checkpoint_path):
        super(SparQLeFineTune, self).__init__()
        self.text_embedding = TextPrenetPreTrain()
        self.tokenizer = self.text_embedding.tokenizer
        self.speech_encoder = self.init_encoder(config)
        
        self.Qformer, self.query_tokens = self.init_Qformer(
            config.num_query_token, config.hidden_size, config.cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.speech_proj_adapter = nn.Linear(1280, self.Qformer.config.hidden_size)

        self.alpha = 0.4
        self.current_epoch = None
        self.current_iter = None
        self.num_iters_per_epoch = None

        self.llm_tokenizer = AutoTokenizer.from_pretrained(os.environ['LLM_BACKBONE'], use_fast=False, cache_dir=os.environ['CACHE'], token=os.environ['HF_TOKEN'])
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = Qwen2ForCausalLM.from_pretrained(os.environ['LLM_BACKBONE'], cache_dir=os.environ['CACHE'], token=os.environ['HF_TOKEN'])
        self.llm_proj_layer_norm = nn.LayerNorm(self.llm_model.config.hidden_size)
        
        logging.info("freeze llm")
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        
        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )
        self.prompt = PROMPT
        # self.prompt = None

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            logger.info(f"Missing keys: {missing_keys}")
            logger.info(f"Unexpected keys: {unexpected_keys}")

        self.llm_model.gradient_checkpointing_enable()
        self.speech_encoder.gradient_checkpointing_enable()
        
        
    def forward(
            self, 
            speech_inputs, 
            inputs_attention_mask, 
            source_labels, 
            source_labels_attention_mask, 
            target_labels, 
            target_labels_attention_mask,
            steps_per_epoch,
            target_lang,
            source_lang
        ):
        
        # outputs = self.speech_encoder(input_values=speech_inputs, attention_mask=inputs_attention_mask)
        outputs = self.speech_encoder(speech_inputs, attention_mask=inputs_attention_mask)
        # speech_embeds = self.lang_speech_proj(outputs.last_hidden_state)
        speech_embeds = outputs.last_hidden_state
        speech_embeds = self.speech_proj_adapter(speech_embeds)
        # if normalize:
        #     speech_embeds = F.normalize(speech_embeds)
        batch, seq, _ = speech_embeds.shape
        input_lengths = self.speech_encoder._get_feat_extract_output_lengths(inputs_attention_mask.sum(-1)).to(torch.long)
        positions = torch.arange(seq, device=speech_inputs.device).expand(batch, seq)
        
        speech_attn = positions < input_lengths.unsqueeze(1)

        query_tokens = self.query_tokens.expand(batch, -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_attn,
            use_cache=True,
            return_dict=True,
        )

        ##================= Speech Generation ========================##
        inputs = self.llm_proj(query_output.last_hidden_state)
        inputs = self.llm_proj_layer_norm(inputs)
        
        atts_llm = torch.ones(inputs.size()[:-1], dtype=torch.long).to(speech_embeds.device)
        LOSS = 0
        for targets, targets_attention, lang in zip(
                [source_labels, target_labels], 
                [source_labels_attention_mask, target_labels_attention_mask],
                [source_lang, target_lang]
            ):
            lang = lang[0]
            # targets = target_labels
            # targets_attention = target_labels_attention_mask
            
            labels = targets.clone()
            empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(speech_embeds.device).fill_(self.llm_tokenizer.pad_token_id)
            labels = torch.cat([empty_targets, targets], dim=1)
            
            if self.prompt:
                batch, _ = targets.shape
                prompt = self.prompt.format(target_lang=lang)
                prompt_tokens = self.llm_tokenizer(prompt, return_tensors='pt').input_ids.expand(batch, -1).to(targets.device)
                prompt_length = prompt_tokens.shape[1]

                targets = torch.concat([prompt_tokens, targets], axis=1)
                targets_attention = torch.cat([targets_attention[:, :prompt_length], targets_attention], axis=1)
                targets_attention[:, :prompt_length] = 1  # enable attending prompt
                
                labels = torch.concat([prompt_tokens, labels], axis=1)
                labels[:, :prompt_length] = self.llm_tokenizer.pad_token_id  # do not apply loss to the prompt
                
            
            inputs_tokens = self.llm_model.model.embed_tokens(targets)
            inputs_embeds = torch.cat([inputs, inputs_tokens], dim=1)
            attention_mask = torch.cat([atts_llm, targets_attention], dim=1)
            
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                # labels=labels,
            )
            # padding_idx = targets == self.llm_tokenizer.pad_token_id
            # targets[padding_idx] = -100

            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=self.llm_tokenizer.pad_token_id)
            shift_logits = shift_logits.view(-1, self.llm_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            LOSS += loss

            print(f'{lang} loss {loss.item()}')

        return SparQleOutput(
            loss=LOSS,
        )
    

def librispeechdataset():
    train_data = LIBRISPEECH(
        root=f'{os.environ["CACHE"]}',
        url='train-clean-100',

    )
    print(f"Loaded train with {len(train_data)} examples")
    
    val_data = LIBRISPEECH(
        root=f'{os.environ["CACHE"]}',
        url='dev-clean',

    )
    return train_data, val_data

def covostdataset():
    LANGDICT = {
        'en_de': 'german',
        'en_fr': 'french',
        'en_tr': 'turkish'
    }
    def add_lang(example, lang):
        example['target_lang'] = lang
        return example
    
    train_datasets = []
    val_datasets = []
    langs = os.environ['LANGS'].split()
    for lang in langs:
        
        dataset = load_dataset(
            'facebook/covost2', 
            lang,
            data_dir=os.environ['DATADIR'],
            cache_dir=os.environ['CACHE'], 
            trust_remote_code=True
        )
        print(f"Loaded {lang} train with {len(dataset['train'])} examples")
        
        dataset['train'] = dataset['train'].map(add_lang, fn_kwargs={'lang': LANGDICT[lang]})
        dataset['validation'] = dataset['validation'].map(add_lang, fn_kwargs={'lang': LANGDICT[lang]})

        corrupted_ids = [
            218745, 218746, 218747, 218748, 218749, 218750, 218751, 218752, 218753, 218754, 218755, 218756, 218757, 218758, 218759, 218760, 218762,
            33561,
        ]

        train_size = len(dataset['train'])
        valid_indices = [i for i in range(train_size) if i not in corrupted_ids]

        # Select only the valid indices
        dataset['train'] = dataset['train'].select(valid_indices)
        
        train_datasets.append(dataset['train'])
        val_datasets.append(dataset['validation'])

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)

def main():
    BATCHSIZE = int(os.environ['BATCHSIZE'])
    GRADIENT_ACCUMULATION = 1

    tokenizer = AutoTokenizer.from_pretrained(os.environ['LLM_BACKBONE'], cache_dir=os.environ['CACHE'])
    framework = os.environ['FRAMEWORK']

    train_data, val_data = covostdataset()
    
    train_dataset = CurriculumDataset(train_data, SAMPLINGRATE, feature_extractor=framework)
    val_dataset = CurriculumDataset(val_data, SAMPLINGRATE, feature_extractor=framework)
    
    config_model = Config(
        hidden_size=768,
        num_query_token=100,
        cross_attention_freq=2,
        bottleneck_dim=1280 if framework=='whisper' else 1024,    
        framework=framework,
        encoder_name=os.environ['ENCODERNAME'],  
    )
    
    total_batch_size = BATCHSIZE * GRADIENT_ACCUMULATION
    steps_per_epoch = len(train_dataset) / total_batch_size

    model = SparQLeFineTune(config_model, SAMPLINGRATE, os.environ.get('CHECKPOINT', None))
    args = TrainingArguments(
        output_dir=os.environ['SAVEDIR'],
        per_device_train_batch_size=BATCHSIZE,
        per_device_eval_batch_size=BATCHSIZE,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=10,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=5,
        weight_decay=0.1,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        save_steps=1000,
        # fp16=True,
        push_to_hub=False,
        report_to=['tensorboard'],
        remove_unused_columns=False,
        max_grad_norm=1,
        save_safetensors=False,

        dataloader_num_workers=0,
        dataloader_pin_memory=True,  # Enables faster data transfer to GPU
        load_best_model_at_end=True,
    )
    data_collator = DataCollatorWithPadding(text_pad_int=tokenizer.pad_token_id, steps_per_epoch=steps_per_epoch)

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    trainer.save_model(os.environ['SAVEDIR'])

if __name__ == "__main__":
    main()
