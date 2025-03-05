import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from dataset import CustomDataset
from torch.utils.data import ConcatDataset

from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor
from qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput

from transformers.feature_extraction_utils import BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.configuration_utils import PretrainedConfig

from transformers.audio_utils import mel_filter_bank, spectrogram, window_function


from s3prl.nn import S3PRLUpstream

from torch.nn import functional as F
from functions import ReverseLayerF
from torch.autograd import Variable

from dataclasses import dataclass
from typing import *

from loguru import logger

MAXDURARION=30
SAMPLINGRATE=16000
QFORMERBASE = 'google-bert/bert-base-multilingual-uncased'


def _rampup_factor(epoch, iters, num_iters_per_epoch):
    return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))


class MomentumDistilationMixin:
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )


@dataclass
class BlipOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_stc: Optional[torch.FloatTensor] = None
    loss_stm: Optional[torch.FloatTensor] = None
    loss_lm: Optional[torch.FloatTensor] = None


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
        source_labels_data = [item['source_labels']['input_ids'].squeeze() for item in features]
        target_labels_data = [item['target_labels']['input_ids'].squeeze() for item in features]
        padded_source_labels = pad_sequence(source_labels_data, batch_first=True, padding_value=self.text_pad_int)
        padded_target_labels = pad_sequence(target_labels_data, batch_first=True, padding_value=self.text_pad_int)
    
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
            'steps_per_epoch': self.steps_per_epoch
        }

class CurriculumDataset(Dataset):
    def __init__(self, data, sampling_rate, feature_extractor):
        self.data = data        
        self.texttokenizer = TextPrenet()
        self.speechtokenizer = SpeechPrenet(sampling_rate, feature_extractor)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_data, input_attention_mask = self.speechtokenizer.pre_process(self.data[idx][0].squeeze())
        input_data, input_attention_mask = self.speechtokenizer.post_process(input_data, input_attention_mask)
        source_labels = self.texttokenizer(self.data[idx][2].lower())
        target_labels = self.texttokenizer(self.data[idx][2].lower())
        
        return {
            'input': input_data,
            'attention_mask': input_attention_mask,
            'source_labels': source_labels,
            'target_labels': target_labels,
        }


class TextPrenet(nn.Module):
    def __init__(self, truncation_side='right'):
        super(TextPrenet, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(QFORMERBASE, truncation_side=truncation_side)
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        
    def forward(self, x):
        return self.tokenizer(x, return_tensors='pt', max_length=200, padding=True, truncation=True)
    

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



class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj


        
class SparQLePreTrain(nn.Module, MomentumDistilationMixin):
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
    def init_SLAM_Qformer(cls, num_query_token, speech_width, qformer_layers=12):
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = speech_width
        configuration.num_hidden_layers = qformer_layers

        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, configuration.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=1.0)
        Qformer = Blip2QFormerModel(configuration)

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

        # elif config.framework == 'custom':
        #     if config.encoder_name == 'dMel':
        #         encoder = dMel(config.hidden_size)
        #     else:
        #         raise NotImplemented

        else:
            raise NotImplemented
        return encoder

    def __init__(self, config, sampling_rate, momentum=0.995, queue_size=2 ** 15):
        super(SparQLePreTrain, self).__init__()
        self.text_embedding = TextPrenet()
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

        self.speech_proj = nn.Linear(self.Qformer.config.hidden_size, config.bottleneck_dim)
        self.speech_proj_m = nn.Linear(self.Qformer.config.hidden_size, config.bottleneck_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, config.bottleneck_dim)
        self.text_proj_m = nn.Linear(self.Qformer.config.hidden_size, config.bottleneck_dim)

        self.stm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.register_buffer("speech_queue", torch.randn(config.bottleneck_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(config.bottleneck_dim, queue_size))

        self.speech_queue = nn.functional.normalize(self.speech_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.alpha = 0.4
        self.current_epoch = None
        self.current_iter = None
        self.num_iters_per_epoch = None


        self.loss_domain = torch.nn.NLLLoss()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(config.bottleneck_dim, 100))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))  # english vs all
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=-1))


        self.model_pairs = [
            [self.speech_proj, self.speech_proj_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()
        
        
    def forward(
            self, 
            speech_inputs, 
            inputs_attention_mask, 
            source_labels, 
            source_labels_attention_mask, 
            target_labels, 
            target_labels_attention_mask,
            steps_per_epoch,
            normalize=True
        ):
    
        alpha = self.alpha * _rampup_factor(
            epoch=self.current_epoch,
            iters=self.current_iter,
            num_iters_per_epoch=math.ceil(steps_per_epoch),
        )
        # print(speech_inputs.shape)
        # print(inputs_attention_mask.shape)
        # print(source_labels.shape)
        # print(source_labels_attention_mask.shape)
        # print(target_labels.shape)
        # print(target_labels_attention_mask.shape)

        outputs = self.speech_encoder(speech_inputs, attention_mask=inputs_attention_mask)
        speech_embeds = outputs.last_hidden_state
        speech_embeds = self.speech_proj(speech_embeds)

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

        text_output = self.Qformer.bert(
            source_labels, 
            attention_mask=source_labels_attention_mask, 
            return_dict=True,
        )


        ###============== Speech-text Contrastive ===================###
        with torch.no_grad():
            # self._momentum_update()
            speech_feat_m = F.normalize(
                self.speech_proj_m(query_output.last_hidden_state), dim=-1
            ).mean(1)
        
            speech_feat_all = torch.cat(
                [speech_feat_m.t(), self.speech_queue.clone().detach()], dim=1
            )
            
            text_feat_m = F.normalize(self.text_proj_m(text_output.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_s2t_m = speech_feat_m @ text_feat_all / self.temp
            sim_t2s_m = text_feat_m @ speech_feat_all / self.temp

            sim_targets = torch.zeros(sim_s2t_m.size()).to(speech_embeds.device)
            sim_targets.fill_diagonal_(1)

            sim_s2t_targets = (
                alpha * F.softmax(sim_s2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            sim_t2s_targets = (
                alpha * F.softmax(sim_t2s_m, dim=1) + (1 - alpha) * sim_targets
            )

        sim_s2t = speech_feat_m @ text_feat_all / self.temp
        sim_t2s = text_feat_m @ speech_feat_all / self.temp

        loss_s2t = -torch.sum(
            F.log_softmax(sim_s2t, dim=1) * sim_s2t_targets, dim=1
        ).mean()
        loss_t2s = -torch.sum(
            F.log_softmax(sim_t2s, dim=1) * sim_t2s_targets, dim=1
        ).mean()

        loss_stc = (loss_s2t + loss_t2s) / 2


        ###============== Multilingual-text Matching ===================###
        # source_domain_label = torch.zeros(batch)
        # source_domain_label = Variable(source_domain_label.long().to(speech_inputs.device))
        # source_domain_output = self.domain_classifier(speech_feat_m)
        # source_domain_loss = self.loss_domain(source_domain_output, source_domain_label)

        # target_domain_label = torch.ones(batch)
        # target_domain_label = Variable(target_domain_label.long().to(speech_inputs.device)) 

        # target_domain_text_output = self.Qformer.bert(target_labels, attention_mask=target_labels_attention_mask, return_dict=True,)
        # target_domain_text_feat = F.normalize(
        #     self.text_proj(target_domain_text_output.last_hidden_state[:, 0, :]), dim=-1
        # )
        # reverse_feature = ReverseLayerF.apply(target_domain_text_feat, alpha)

        # target_domain_output = self.domain_classifier(reverse_feature)
        # target_domain_loss = self.loss_domain(target_domain_output, target_domain_label)
        
        # loss_stm = source_domain_loss + target_domain_loss
        ##================= Speech Matching ========================##

        encoder_input_ids = target_labels
        query_atts_stm = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            speech_embeds.device
        )
        attention_mask_all = torch.cat([query_atts_stm, source_labels_attention_mask], dim=1)

        output_pos = self.Qformer.bert(
            encoder_input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask_all,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_attn,
            return_dict=True,
        )

        with torch.no_grad():
            weights_t2s = F.softmax(sim_t2s[:, :batch], dim=1) + 1e-4
            weights_t2s.fill_diagonal_(0)
            weights_s2t = F.softmax(sim_s2t[:, :batch], dim=1) + 1e-4
            weights_s2t.fill_diagonal_(0)

        # select a negative image for each text
        speech_embeds_neg = []
        for b in range(batch):
            neg_idx = torch.multinomial(weights_t2s[b], 1).item()
            speech_embeds_neg.append(speech_embeds[neg_idx])
        speech_embeds_neg = torch.stack(speech_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(batch):
            neg_idx = torch.multinomial(weights_s2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(source_labels_attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([source_labels_attention_mask, text_atts_neg], dim=0)
        
        query_tokens_matching = self.query_tokens.expand(batch * 2, -1, -1)
        query_atts_stm_matching = torch.ones(query_tokens_matching.size()[:-1], dtype=torch.long).to(
            speech_embeds.device
        )
        text_atts_all = torch.cat([query_atts_stm_matching, text_atts_all], dim=1)

        speech_embeds_all = torch.cat([speech_embeds_neg, speech_embeds], dim=0)
        speech_atts_all = torch.cat([speech_attn, speech_attn], dim=0)

        output_neg = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_matching,
            attention_mask=text_atts_all,
            encoder_hidden_states=speech_embeds_all,
            encoder_attention_mask=speech_atts_all,
            return_dict=True,
        )

        sp_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, self.query_tokens.size(1), :],
                output_neg.last_hidden_state[:, self.query_tokens.size(1), :],
            ],
            dim=0,
        )
        stm_logits = self.stm_head(sp_embeddings)
        stm_labels = torch.cat(
            [torch.ones(batch, dtype=torch.long), torch.zeros(2 * batch, dtype=torch.long)],
            dim=0,
        ).to(speech_embeds.device)
        loss_stm = F.cross_entropy(stm_logits, stm_labels)


        ##================= Speech Generation ========================##
        decoder_input_ids = source_labels.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        source_labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            speech_inputs.device
        )
        attention_mask = torch.cat([query_atts, source_labels_attention_mask], dim=1)
        lm_output = self.Qformer(decoder_input_ids,attention_mask=attention_mask,past_key_values=query_output.past_key_values,return_dict=True,labels=source_labels,)

        loss_lm = lm_output.loss

        logger.info(f"loss: {loss_stc + loss_stm + loss_lm}, loss_stc: {loss_stc}, loss_stm: {loss_stm}, loss_lm: {loss_lm}", loss_stc=loss_stc, loss_stm=loss_stm, loss_lm=loss_lm)

        return BlipOutput(
            loss=loss_stc + loss_lm,
            loss_stc=loss_stc,
            loss_stm=loss_stm,
            loss_lm=loss_lm,
        )
    
from transformers import TrainerCallback
import math

class TrainingProgressCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is not None:
            current_step = state.global_step
            current_epoch = state.epoch

            model.current_epoch = current_epoch
            model.current_iter = current_step



def main():
    BATCHSIZE = 8
    GRADIENT_ACCUMULATION = 1
    framework='whisper'

    train_splits = [
        f'{os.environ["CACHE"]}/IWSLT.OfflineTask/data/en-de/tst2022/segmented/',
        f'{os.environ["CACHE"]}/IWSLT.OfflineTask/data/en-de/tst2022/segmented/',
        f'{os.environ["CACHE"]}/IWSLT.OfflineTask/data/en-ja/tst2022/segmented/',
        f'{os.environ["CACHE"]}/IWSLT.OfflineTask/data/en-zh/tst2022/segmented/',
        f'{os.environ["CACHE"]}/IWSLT.OfflineTask/data/en-de/tst2022/segmented/',
        f'{os.environ["CACHE"]}/IWSLT.OfflineTask/data/en-ja/tst2022/segmented/',
        f'{os.environ["CACHE"]}/IWSLT.OfflineTask/data/en-zh/tst2022/segmented/',
    ]
    train_datasets = []
    for split in train_splits:
        dataset = CustomDataset(
            root=split,
        )
        train_datasets.append(dataset)
        print(f"Loaded {split} with {len(dataset)} examples")
    train_data = ConcatDataset(train_datasets)
    val_data = CustomDataset(
        root=f'{os.environ["CACHE"]}/IWSLT.OfflineTask/data/en-de/tst2021/segmented/',
    )
    
    train_dataset = CurriculumDataset(train_data, SAMPLINGRATE, framework)
    val_dataset = CurriculumDataset(val_data, SAMPLINGRATE, framework)
    
    config_model = Config(
        hidden_size=768,
        num_query_token=100,
        cross_attention_freq=2,
        bottleneck_dim=512,     
        framework=framework,
        encoder_name='openai/whisper-large-v3',  
        # framework='huggingface',
        # encoder_name='facebook/hubert-large-ll60k',  
    )
    

    
    total_batch_size = BATCHSIZE * GRADIENT_ACCUMULATION
    steps_per_epoch = len(train_dataset) / total_batch_size

    model = SparQLePreTrain(config_model, SAMPLINGRATE)
    args = TrainingArguments(
        output_dir=os.environ['SAVEDIR'],
        per_device_train_batch_size=BATCHSIZE,
        per_device_eval_batch_size=BATCHSIZE,
        evaluation_strategy="steps",
        eval_steps=1_000,
        logging_steps=10,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=100,
        weight_decay=0.1,
        warmup_steps=5_000,
        lr_scheduler_type="cosine",
        learning_rate=1e-4,
        save_steps=10_000,
        fp16=False,
        push_to_hub=False,
        report_to=['tensorboard'],
        remove_unused_columns=False,
        max_grad_norm=10,
        save_safetensors=False,


        dataloader_num_workers=8,
        dataloader_pin_memory=True,  # Enables faster data transfer to GPU
        load_best_model_at_end=True,
    )
    data_collator = DataCollatorWithPadding(text_pad_int=1, steps_per_epoch=steps_per_epoch)

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[TrainingProgressCallback()]
    )
    trainer.train()
    trainer.save_model(os.environ['SAVEDIR'])

if __name__ == "__main__":
    main()