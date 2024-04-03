"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import json

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from .blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from .blip_outputs import BlipOutput, BlipOutputFeatures

from fairseq.models import register_model, register_model_architecture
from models.speechtokenizer import SpeechTokenizer


from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from fairseq.modules import (
    FairseqDropout,
)

from .speech_tokenizer import create_speech_tokenizer


from fairseq import utils


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)

@register_model("speech_qformer_no_cl")
class Blip2Qformer(Blip2Base):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)
        speech_encoder, config = cls.build_speech_model({
            # '/l/users/amirbek.djanibekov/master-thesis/models/SpeechTokenizer/huggingface/SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt',
            # '/l/users/amirbek.djanibekov/master-thesis/models/SpeechTokenizer/huggingface/SpeechTokenizer/speechtokenizer_hubert_avg/config.json',
            'speech_model_file_path': args.speech_model_file_path,
            'speech_model_config_path': args.speech_model_config_path,
            'freeze_enc': True
        })      
        Qformer, query_tokens, tokenizer = cls.build_qformer({
            'codebook_size': config['codebook_size'],
            'cross_attention_freq': args.cross_attention_freq,
            'num_query_token': args.num_query_token
        })

        return cls(
            args,
            speech_encoder,
            Qformer, 
            query_tokens,
            tokenizer,
            config
        )
    @classmethod
    def build_speech_model(cls, args, dictionary=None, embed_tokens=None):
        speech_encoder = create_speech_tokenizer(
            args['speech_model_config_path'], args['speech_model_file_path'], None
        )
        if hasattr(speech_encoder, "decoder"):   # Pruning
            del speech_encoder.decoder

        with open(args['speech_model_config_path']) as config_file:
            config = json.load(config_file)

        if args['freeze_enc']:
            for name, param in speech_encoder.named_parameters():
                param.requires_grad = False
            speech_encoder.eval()


        return speech_encoder, config
    
    @classmethod
    def build_qformer(cls, args):
        tokenizer = cls.init_tokenizer()
        Qformer, query_tokens = cls.init_Qformer(
            args['num_query_token'], args['codebook_size'], args['cross_attention_freq']
        )
        Qformer.resize_token_embeddings(len(tokenizer))

        return Qformer, query_tokens, tokenizer

    @staticmethod
    def add_args(parser):
        parser.add_argument("--cross_attention_freq", type=int)
        parser.add_argument("--num_query_token", type=int)
        parser.add_argument("--speech_model_config_path", type=str)
        parser.add_argument("--speech_model_file_path", type=str)

    def __init__(
        self,
        args,
        speech_encoder,
        Qformer, 
        query_tokens,
        tokenizer,
        speech_config,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        embed_dim=256,
        max_txt_len=100,

    ):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.Qformer = Qformer
        self.query_tokens = query_tokens
        self.tokenizer = tokenizer
        self.speech_config = speech_config

        self.ln_speech = LayerNorm(speech_config['codebook_size'])

        
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.speech_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = args.num_query_token
    

    def forward(self, samples):
        speech = samples["source"]
        text = samples["target"][0]
        
        codes = self.speech_encoder(speech.unsqueeze(1), 1).transpose(1, 2)
        speech_embeds = self.ln_speech(codes)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(
            speech.device
        )

        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            use_cache=True,
            return_dict=True,
        )

        speech_feats = F.normalize(
            self.speech_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_txt_len, 
            return_tensors="pt",
        ).to(speech.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        ###============== Speech-text Contrastive ===================### 
        rank = 0
        bs = speech.size(0)
        speech_feats_all = speech_feats
        text_feat_all = text_feat  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            speech_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # speech-text similarity: aggregate across all query tokens
        sim_s2t, _ = sim_q2t.max(-1)
        sim_s2t = sim_s2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), speech_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-speech similarity: aggregate across all query tokens
        sim_t2s, _ = sim_t2q.max(-1)
        sim_t2s = sim_t2s / self.temp  # [batch_size, batch_size*num_gpu]

        loss_itc = torch.tensor(0.0)
        ###============== Speech-text Matching ===================###
        text_input_ids_world = text_tokens.input_ids
        text_attention_mask_world = text_tokens.attention_mask
        speech_embeds_world = speech_embeds
        with torch.no_grad():
            sim_t2s[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_s2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
                
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
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
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

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=speech_embeds_all,
            encoder_attention_mask=speech_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(speech.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Transcription Generation  ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            speech.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        
        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=True,
        num_beams=1,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - speech (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each speech.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        speech = samples["source"]
        codes = self.speech_encoder.encode(speech.unsqueeze(1), 1) # codes: (n_q, B, T)
        speech_embeds = self.speech_encoder.quantizer.decode(codes).transpose(1, 2)
        speech_embeds = self.ln_speech(speech_embeds)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(
            speech.device
        )
        model_kwargs = {
            "encoder_hidden_states": speech_embeds,  # used in cross attention
            "encoder_attention_mask": speech_atts,  # used in cross attention
        }

        # input_ids = (
        #     torch.LongTensor(speech.size(0), 1)
        #     .fill_(self.tokenizer.bos_token_id)
        #     .to(speech.device)
        # )
        prompt = ""
        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)

        ## No need for beam search or any sampling strategy since we just decoding query tokens
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            use_cache=True,
            return_dict=True,
        )
        lm_outputs = self.Qformer.cls(query_outputs[0])
        captions = self.tokenizer.batch_decode(lm_outputs, skip_special_tokens=True)

        breakpoint()
        return captions

    def forward_speech(self, speech):

        codes = self.speech_encoder.encode(speech.unsqueeze(1), 1) # codes: (n_q, B, T)
        speech_embeds = self.speech_encoder.quantizer.decode(codes).transpose(1, 2)
        speech_embeds = self.ln_speech(speech_embeds)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(
            speech.device
        )
        
        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, speech_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - speech (torch.Tensor): A tensor of shape (B, C, H, W) containing the speech.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "speech".
                If "multimodal", return speech features and multimodal features;
                if "text", return text features;
                if "speech", return speech features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        speech = samples.get("speech")
        caption = samples.get("text_input")

        # assert mode is one of "speech", "text", "multimodal"
        assert mode in [
            "speech",
            "text",
            "multimodal",
        ], "mode must be one of 'speech', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "speech":
            assert (
                speech is not None
            ), "Speech is not provided for mode 'speech' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(speech))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(speech))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        freeze_enc = cfg.get("freeze_enc", True)

        max_txt_len = cfg.get("max_txt_len", 64)

        model = cls(
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            freeze_enc=freeze_enc,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity s2t, t2s matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)



@register_model_architecture(model_name="speech_qformer_no_cl", arch_name="speech_qformer_no_cl")
def base_architecture(args):
    pass
