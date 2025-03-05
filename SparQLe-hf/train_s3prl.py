import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from librispeechdataset import LIBRISPEECH
from torch.utils.data import ConcatDataset

from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor
from qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput

from transformers.feature_extraction_utils import BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.configuration_utils import PretrainedConfig


from s3prl.nn import S3PRLUpstream

from torch.nn import functional as F

from dataclasses import dataclass
from typing import *

from loguru import logger

MAXDURARION=30
SAMPLINGRATE=16000


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
    
    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_data = [item['input'] for item in features]
        target_data = [item['target']['input_ids'].squeeze() for item in features]
        padded_targets = pad_sequence(target_data, batch_first=True, padding_value=self.text_pad_int)
    
        inputs_attention_mask = [item['attention_mask'] for item in features]
        targets_attention_mask = (padded_targets != self.text_pad_int).long()
        
        return {
            'inputs_ids': torch.stack(input_data).squeeze(),
            'inputs_attention_mask': torch.stack(inputs_attention_mask).squeeze(),
            'labels': padded_targets,
            'labels_attention_mask': targets_attention_mask
        }

class CurriculumDataset(Dataset):
    def __init__(self, data, sampling_rate):
        self.data = data        
        self.texttokenizer = TextPrenet()
        self.speechtokenizer = SpeechPrenet(sampling_rate)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_data, input_attention_mask = self.speechtokenizer.pre_process(self.data[idx][0].squeeze())
        input_data, input_attention_mask = self.speechtokenizer.post_process(input_data, input_attention_mask)
        target = self.texttokenizer(self.data[idx][2].lower())
        
        return {
            'input': input_data,
            'attention_mask': input_attention_mask,
            'target': target,
        }


class TextPrenet(nn.Module):
    def __init__(self, truncation_side='right'):
        super(TextPrenet, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        
    def forward(self, x):
        return self.tokenizer(x, return_tensors='pt', max_length=200, padding=True, truncation=True)
    

class SpeechPrenet(SequenceFeatureExtractor):
    model_input_names = ["input_features"]
    def __init__(self, sampling_rate, max_duration=MAXDURARION):
        super().__init__(
            feature_size=80,
            sampling_rate=sampling_rate,
            padding_value=0,
            return_attention_mask=True,
        )
        self.n_samples = max_duration * sampling_rate
        
        
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
        padded_inputs["input_features"] = input_features
        padded_inputs["attention_mask"] = padded_inputs["attention_mask"]
        padded_inputs = padded_inputs.convert_to_tensors('pt')
        
        return padded_inputs["input_features"], padded_inputs["attention_mask"]

    def post_process(self, input_features, attention_mask):

        return input_features, attention_mask


        
class SparQLePreTrain(nn.Module):
    @classmethod
    def init_Qformer(cls, num_query_token, speech_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = speech_width
        encoder_config.is_decoder = True
        
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
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
        elif config.framework == 's3prl':
            encoder = S3PRLUpstream(config.encoder_name)
            encoder = encoder.eval()
            logger.info("freeze encoder")

        elif config.framework == 'custom':
            if config.encoder_name == 'dMel':
                encoder = dMel(config.hidden_size)
            else:
                raise NotImplemented

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
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, config.bottleneck_dim)

        self.stm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
        
    def forward(self, inputs_ids, inputs_attention_mask, labels, labels_attention_mask):
        speech = inputs_ids
        text = labels

        speech_embeds = self.speech_encoder(input_values=inputs_ids, attention_mask=inputs_attention_mask).last_hidden_state
        batch, seq, _ = speech_embeds.shape
        input_lengths = self.speech_encoder._get_feat_extract_output_lengths(inputs_attention_mask.sum(-1)).to(torch.long)
        positions = torch.arange(seq, device=speech.device).expand(batch, seq)
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

        text_output = self.Qformer.bert(labels, attention_mask=labels_attention_mask, return_dict=True,)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)


        ###============== Speech-text Contrastive ===================###
        speech_feats_all = speech_feats
        text_feat_all = text_feat

        sim_q2t = torch.matmul(
            speech_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        sim_s2t, _ = sim_q2t.max(-1)
        sim_s2t = sim_s2t / self.temp

        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), speech_feats_all.permute(0, 2, 1)
        ).squeeze()

        sim_t2s, _ = sim_t2q.max(-1)
        sim_t2s = sim_t2s / self.temp  # [batch_size, batch_size*num_gpu]

        bs = speech.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(speech.device)        
        loss_stc = (
            F.cross_entropy(sim_s2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2s, targets, label_smoothing=0.1)
        ) / 2

        ###============== Speech-text Matching ===================###
        text_input_ids_world = labels
        text_attention_mask_world = labels_attention_mask
        speech_embeds_world = speech_embeds
        with torch.no_grad():
            sim_t2s[:, :bs].fill_diagonal_(-10000)
            sim_s2t[:, :bs].fill_diagonal_(-10000)            
                
            weights_t2i = F.softmax(sim_t2s, dim=1)
            weights_i2t = F.softmax(sim_s2t, dim=1)

        speech_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            speech_embeds_neg.append(speech_embeds_world[neg_idx])
        speech_embeds_neg = torch.stack(speech_embeds_neg, dim=0)

        # select a negative text for each speech
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [labels, labels, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [labels_attention_mask, labels_attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            speech.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        speech_embeds_all = torch.cat(
            [speech_embeds, speech_embeds_neg, speech_embeds], dim=0
        )  # pos, neg, pos
        speech_atts_all = torch.ones(speech_embeds_all.size()[:-1], dtype=torch.long).to(
            speech.device
        )

        output_stm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=speech_embeds_all,
            encoder_attention_mask=speech_atts_all,
            return_dict=True,
        )

        sp_embeddings = output_stm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        sp_output = self.stm_head(sp_embeddings)
        logits = sp_output.mean(dim=1)

        stm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(speech.device)
        loss_stm = F.cross_entropy(logits, stm_labels)

        ##================= Speech Captioning ========================##
        decoder_input_ids = labels.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            speech.device
        )
        attention_mask = torch.cat([query_atts, labels_attention_mask], dim=1)
        lm_output = self.Qformer(decoder_input_ids,attention_mask=attention_mask,past_key_values=query_output.past_key_values,return_dict=True,labels=labels,)

        loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_stc + loss_stm + loss_lm,
            loss_stc=loss_stc,
            loss_stm=loss_stm,
            loss_lm=loss_lm,
        )


def main():
    train_splits = [
        "train-clean-100",
        "train-clean-360",
        "train-other-500"
    ]
    train_datasets = []
    for split in train_splits:
        dataset = LIBRISPEECH(
            root=CACHE,
            url=split,
            download=True,
            manifest_file=f'{CACHE}/LibriSpeech/{split}.tsv',
            minkeep=16000,
            maxkeep=MAXDURARION * 16000
        )
        train_datasets.append(dataset)
        print(f"Loaded {split} with {len(dataset)} examples")
        
    train_data = ConcatDataset(train_datasets)
    val_data = LIBRISPEECH(
        root=CACHE,
        url='dev-clean',
        download=True
    )
    
    train_dataset = CurriculumDataset(train_data, SAMPLINGRATE)
    val_dataset = CurriculumDataset(val_data, SAMPLINGRATE)
    
    config_model = Config(
        hidden_size=1024,
        num_query_token=100,
        cross_attention_freq=2,
        bottleneck_dim=512,     
        framework='huggingface',
        encoder_name='facebook/wav2vec2-large-960h',  
    )
    
    model = SparQLePreTrain(config_model, SAMPLINGRATE)
    args = TrainingArguments(
        output_dir=os.environ['SAVEDIR'],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=1_000,
        logging_steps=10,
        gradient_accumulation_steps=1,
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
        max_grad_norm=1,
        save_safetensors=False,


        dataloader_num_workers=8,
        dataloader_pin_memory=True,  # Enables faster data transfer to GPU
        load_best_model_at_end=True,
    )
    data_collator = DataCollatorWithPadding(text_pad_int=1)

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(os.environ['SAVEDIR'])

if __name__ == "__main__":
    main()