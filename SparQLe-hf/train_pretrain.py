import os
import torch
import string
import torch.nn as nn
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from dataset import CustomDataset
from dataset_fairseq import CustomDataset

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
from librispeechdataset import LIBRISPEECH

from torch.nn import functional as F
from functions import ReverseLayerF
from torch.autograd import Variable

from dataclasses import dataclass
from typing import *

from loguru import logger

MAXDURARION=30
SAMPLINGRATE=16000
QFORMERBASE = 'google-bert/bert-base-multilingual-uncased'
# QFORMERBASE = 'google-bert/bert-large-uncased'

PREVLOSS = None


def _rampup_factor(epoch, iters, num_iters_per_epoch):
    return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))


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

        filepaths = [item['filepath'] for item in features]
        idxs = [item['idx'] for item in features]
        
        return {
            'speech_inputs': torch.stack(input_data).squeeze(),
            'inputs_attention_mask': torch.stack(inputs_attention_mask).squeeze(),
            'source_labels': padded_source_labels,
            'source_labels_attention_mask': source_attention_mask,
            'target_labels': padded_target_labels,
            'target_labels_attention_mask': target_attention_mask,
            'steps_per_epoch': self.steps_per_epoch,
            'filepaths': filepaths,
            'idxs': idxs
        }

class CurriculumDataset(Dataset):
    def __init__(self, data, sampling_rate, feature_extractor):
        self.data = data        
        self.texttokenizer = TextPrenet()
        self.speechtokenizer = SpeechPrenet(sampling_rate=sampling_rate, feature_extractor=feature_extractor)

        self.translator = str.maketrans('', '', string.punctuation)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target_lang = self.data[idx]['target_lang']
        input_data, input_attention_mask = self.speechtokenizer.pre_process(self.data[idx]['audio']['array'].squeeze())
        # audio, sr = sf.read(self.data[idx]['audio_path'])
        # assert sr == 16000
        # input_data, input_attention_mask = self.speechtokenizer.pre_process(audio)
        source_text = self.data[idx]['sentence'].lower().translate(self.translator)
        target_text = self.data[idx]['translation'].lower().translate(self.translator)

        source_labels = self.texttokenizer(source_text)
        target_labels = self.texttokenizer(target_text)
        
        return {
            'input': input_data,
            'attention_mask': input_attention_mask,
            'source_labels': source_labels,
            'target_labels': target_labels,
            'target_lang': target_lang,
            'source_text': source_text,
            'target_text': target_text,
            'filepath': self.data[idx]['audio'],
            # 'filepath': self.data[idx]['audio_path'],
            'idx': idx
        }
    
    # def __getitem__(self, idx):
    #     input_data, input_attention_mask = self.speechtokenizer.pre_process(self.data[idx][0].squeeze())
    #     input_data, input_attention_mask = self.speechtokenizer.post_process(input_data, input_attention_mask)
    #     source_labels = self.texttokenizer(self.data[idx][2].lower())
    #     target_labels = self.texttokenizer(self.data[idx][2].lower())
        
    #     return {
    #         'input': input_data,
    #         'attention_mask': input_attention_mask,
    #         'source_labels': source_labels,
    #         'target_labels': target_labels,
    #     }


class TextPrenet(nn.Module):
    def __init__(self, truncation_side='right'):
        super(TextPrenet, self).__init__()
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
        self.n_samples = MAXDURARION * 16000
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
            sampling_rate=self.sampling_rate,
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

    # def __call__(self, *args, **kwds):
    #     return self.pre_process(*args, **kwds)


        
class SparQLePreTrain(nn.Module):
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

        # elif config.framework == 'custom':
        #     if config.encoder_name == 'dMel':
        #         encoder = dMel(config.hidden_size)
        #     else:
        #         raise NotImplemented

        else:
            raise NotImplemented
        return encoder

    def __init__(self, config, sampling_rate):
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
        self.speech_proj_adapter = nn.Linear(1280, self.Qformer.config.hidden_size)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, config.bottleneck_dim)

        self.stm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.alpha = 0.8
        self.current_epoch = None
        self.current_iter = None

        self.loss_domain = torch.nn.NLLLoss()
        # self.domain_classifier = nn.Sequential([
        #     nn.Linear(config.bottleneck_dim, 100),
        #     nn.BatchNorm1d(100),
        #     nn.ReLU(True),
        #     nn.Linear(100, 2),
        #     nn.LogSoftmax(dim=-1)
        # ])
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(config.bottleneck_dim, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))  # english vs all
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=-1))

        
    def forward(
            self, 
            speech_inputs, 
            inputs_attention_mask, 
            source_labels, 
            source_labels_attention_mask, 
            target_labels, 
            target_labels_attention_mask,
            steps_per_epoch,
            normalize=False,
            **kwargs
        ):
        
        outputs = self.speech_encoder(speech_inputs, attention_mask=inputs_attention_mask)
        speech_embeds = self.speech_proj_adapter(outputs.last_hidden_state)
        
        if normalize:
            speech_embeds = F.normalize(speech_embeds)
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

        speech_feats = F.normalize(self.speech_proj(query_output.last_hidden_state), dim=-1)

        text_output = self.Qformer.bert(source_labels, attention_mask=source_labels_attention_mask, return_dict=True,)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)


        ###============== Speech-text Contrastive ===================###
        speech_feats_all = speech_feats
        text_feat_all = text_feat
        bs = speech_inputs.size(0)

        sim_q2t = torch.matmul(
            speech_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze(dim=(1, 3))
       
        sim_s2t, _ = sim_q2t.max(-1)
        sim_s2t = sim_s2t / self.temp

        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), speech_feats_all.permute(0, 2, 1)
        ).squeeze()

        sim_t2s, _ = sim_t2q.max(-1)
        sim_t2s = sim_t2s / self.temp

        
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(speech_inputs.device)
        loss_stc = (
            F.cross_entropy(sim_s2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2s, targets, label_smoothing=0.1)
        ) / 2

        ###============== Adversarial-text Matching ===================###

        alpha = self.alpha * _rampup_factor(
            epoch=self.current_epoch,
            iters=self.current_iter,
            num_iters_per_epoch=math.ceil(steps_per_epoch),
        )

        source_domain_label = torch.zeros(bs)
        source_domain_label = Variable(source_domain_label.long().to(speech_inputs.device))
        source_domain_output = self.domain_classifier(text_feat)
        source_domain_loss = self.loss_domain(source_domain_output, source_domain_label)

        target_domain_label = torch.ones(bs)
        target_domain_label = Variable(target_domain_label.long().to(speech_inputs.device)) 

        target_domain_text_output = self.Qformer.bert(target_labels, attention_mask=target_labels_attention_mask, return_dict=True,)
        target_domain_text_feat = F.normalize(self.text_proj(target_domain_text_output.last_hidden_state[:, 0, :]), dim=-1)
        reverse_feature = ReverseLayerF.apply(target_domain_text_feat, alpha)

        target_domain_output = self.domain_classifier(reverse_feature)
        target_domain_loss = self.loss_domain(target_domain_output, target_domain_label)
        
        loss_adv = source_domain_loss + target_domain_loss

        ###============== Multilingual-text Matching ===================###

        text_input_ids_world = source_labels
        text_attention_mask_world = source_labels_attention_mask
        speech_embeds_world = speech_embeds
        with torch.no_grad():
            sim_t2s[:, 0: bs].fill_diagonal_(-10000)
            sim_s2t[:, 0: bs].fill_diagonal_(-10000)
                
            weights_t2s = F.softmax(sim_t2s, dim=1)
            weights_s2t = F.softmax(sim_s2t, dim=1)

        # select a negative speech for each text
        speech_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2s[b], 1).item()
            speech_embeds_neg.append(speech_embeds_world[neg_idx])
        speech_embeds_neg = torch.stack(speech_embeds_neg, dim=0)

        # select a negative text for each speech
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_s2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [source_labels, source_labels, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [source_labels_attention_mask, source_labels_attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_stm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_stm = torch.ones(query_tokens_stm.size()[:-1], dtype=torch.long).to(
            speech_embeds.device
        )
        attention_mask_all = torch.cat([query_atts_stm, text_atts_all], dim=1)

        speech_embeds_all = torch.cat(
            [speech_embeds, speech_embeds_neg, speech_embeds], dim=0
        )  # pos, neg, pos
        speech_atts_all = torch.ones(speech_embeds_all.size()[:-1], dtype=torch.long).to(
            speech_embeds.device
        )

        output_stm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_stm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=speech_embeds_all,
            encoder_attention_mask=speech_atts_all,
            return_dict=True,
        )
        
        sp_embeddings = output_stm.last_hidden_state[:, : query_tokens_stm.size(1), :]
        sp_output = self.stm_head(sp_embeddings)
        logits = sp_output.mean(dim=1)

        stm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(speech_embeds.device)
        loss_stm = F.cross_entropy(logits, stm_labels)
        ##================= Speech Captioning ========================##
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

        # logger.info(f"loss: {loss_stc + loss_stm + loss_lm}, loss_stc: {loss_stc}, loss_stm: {loss_stm}, loss_lm: {loss_lm}, loss_adv: {loss_adv}", loss_stc=loss_stc, loss_stm=loss_stm, loss_lm=loss_lm, loss_adv=loss_adv)
        logger.info(f"loss: {loss_stc + loss_stm + loss_lm}, loss_stc: {loss_stc}, loss_stm: {loss_stm}, loss_lm: {loss_lm}", loss_stc=loss_stc, loss_stm=loss_stm, loss_lm=loss_lm)

        return BlipOutput(
            loss=loss_stc + loss_stm + loss_lm, #+ loss_adv,
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

def loadlibrispeech(framework):
    train_splits = [
        "train-clean-100",
        # "train-clean-360",
        # "train-other-500"
    ]
    train_datasets = []
    for split in train_splits:
        dataset = LIBRISPEECH(
            root=os.environ['CACHE'],
            url=split,
            download=True,
            manifest_file=f'{os.environ["CACHE"]}/LibriSpeech/{split}.tsv',
            minkeep=16000,
            maxkeep=MAXDURARION * 16000
        )
        train_datasets.append(dataset)
        print(f"Loaded {split} with {len(dataset)} examples")
        
    train_data = ConcatDataset(train_datasets)
    val_data = LIBRISPEECH(
        root=os.environ['CACHE'],
        url='dev-clean',
        download=True
    )
    
    train_dataset = CurriculumDataset(train_data, SAMPLINGRATE, framework)
    val_dataset = CurriculumDataset(val_data, SAMPLINGRATE, framework)
    return train_dataset, val_dataset

def loadiwslt(framework):
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

    return train_dataset, val_dataset

def load_iwslt(framework):
    
    dataset = CustomDataset(
        manifest_file=os.environ['TRAINMANIFEST'],
        text_file=os.environ['TRAINTEXT'],
        maxkeep=30 * 16000,
        minkeep=5 * 16000
    )
    print(f"Loaded train with {len(dataset)} examples")
    
    val_data = CustomDataset(
        manifest_file=os.environ['DEVMANIFEST'],
        text_file=os.environ['DEVTEXT'],
        maxkeep=30 * 16000,
        minkeep=16000
    )
    
    train_dataset = CurriculumDataset(dataset, SAMPLINGRATE, feature_extractor=framework)
    val_dataset = CurriculumDataset(val_data, SAMPLINGRATE, feature_extractor=framework)

    return train_dataset, val_dataset
           
def covostdataset():
    from datasets import load_dataset
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

            46770, 77652, 128308, 227092
        ]

        train_size = len(dataset['train'])
        valid_indices = [i for i in range(train_size) if i not in corrupted_ids]

        # Select only the valid indices
        dataset['train'] = dataset['train'].select(valid_indices)
        
        train_datasets.append(dataset['train'])
        val_datasets.append(dataset['validation'])

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


def ldc_tunisian_english():
    from datasets import load_from_disk

    def add_lang(example, lang):
        example['target_lang'] = lang
        return example
    
    dataset = load_from_disk(
        os.environ['DATADIR'],
    )
    print(f"Loaded train with {len(dataset['train'])} examples")
    dataset['train'] = dataset['train'].map(add_lang, fn_kwargs={'lang': 'english'})
    dataset['dev'] = dataset['dev'].map(add_lang, fn_kwargs={'lang': 'english'})
        
    return dataset['train'], dataset['dev']


def main():
    BATCHSIZE = 16
    GRADIENT_ACCUMULATION = 1
    framework = os.environ['FRAMEWORK']
    # train_dataset, val_dataset = loadiwslt(framework)
    # train_dataset, val_dataset = loadlibrispeech(framework)
    train_dataset, val_dataset = covostdataset()

    train_dataset = CurriculumDataset(train_dataset, SAMPLINGRATE, feature_extractor=framework)
    val_dataset = CurriculumDataset(val_dataset, SAMPLINGRATE, feature_extractor=framework)
    
    config_model = Config(
        hidden_size=768,
        num_query_token=100,
        cross_attention_freq=2,
        bottleneck_dim=512, 
        framework=framework,
        encoder_name='openai/whisper-large-v3',  
    )
    
    model = SparQLePreTrain(config_model, SAMPLINGRATE)
    args = TrainingArguments(
        output_dir=os.environ['SAVEDIR'],
        per_device_train_batch_size=BATCHSIZE,
        per_device_eval_batch_size=BATCHSIZE,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=10,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=10,
        weight_decay=0.1,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        save_steps=1000,
        # fp16=True,
        push_to_hub=False,
        report_to=['tensorboard'],
        remove_unused_columns=False,
        save_safetensors=False,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,  # Enables faster data transfer to GPU
        load_best_model_at_end=True,
    )
    total_batch_size = BATCHSIZE * GRADIENT_ACCUMULATION
    steps_per_epoch = len(train_dataset) / total_batch_size
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