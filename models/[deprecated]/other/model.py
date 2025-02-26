# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import contextlib
from ast import literal_eval
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)

import sys
sys.path.append('/l/users/amirbek.djanibekov/master-thesis/models/adapters/speech_adapter/models')

from modules.encoder_speech import SpeechEncoderPrenet
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.transformer import Embedding
from torch import Tensor

from transformers import LlamaTokenizer
from .llama import LlamaForCausalLM

from blip2.Qformer import BertConfig, BertLMHeadModel
from models.speechtokenizer import SpeechTokenizer

from tqdm import tqdm
import json
import ast

logger = logging.getLogger(__name__)

DEFAULT_MAX_TEXT_POSITIONS = 450
DEFAULT_MAX_SPEECH_POSITIONS = 4000


from blip2.blip2 import Blip2Base, disabled_train

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)

@register_model("speech_adapter_llama")
class SpeechAdapterLlama(Blip2Base):
    def _validate_Qformer_loading(self, qformer_checkpoint_path):
        validator = {}
        checkpoint = torch.load(qformer_checkpoint_path)
        generator_values = checkpoint['model'].keys()
        for generator_ent in tqdm(generator_values):
            entities = generator_ent.split('.')[1:]

            if generator_ent.startswith('Qformer') and generator_ent not in [
                "Qformer.bert.embeddings.word_embeddings.weight",
                "Qformer.bert.embeddings.position_embeddings.weight",
                "Qformer.cls.predictions.bias",
                "Qformer.cls.predictions.transform.dense.weight",
                "Qformer.cls.predictions.transform.dense.bias",
                "Qformer.cls.predictions.transform.LayerNorm.weight",
                "Qformer.cls.predictions.transform.LayerNorm.bias",
                "Qformer.cls.predictions.decoder.weight",
                "Qformer.cls.predictions.decoder.bias",
            ]:
                network = self.Qformer
                for ent in entities:
                    network = getattr(network, ent)
                validator[generator_ent] = (torch.eq(checkpoint['model'][generator_ent], network) == True).sum()
            else:
                continue
        return validator

    def __init__(
            self, 
            args,
            speech_encoder_prenet,
            speech_encoder_config,
            qformer,
            qformer_query_tokens,
            llama_tokenizer,
            llama_model
        ):
        super().__init__()
        self.args = args

        self.speech_encoder = speech_encoder_prenet
        self.Qformer = qformer
        self.query_tokens = qformer_query_tokens
        self.llama_tokenizer = llama_tokenizer
        self.llama_model = llama_model

        self.llama_tokenizer.padding_side = "right"

        
        self.ln_speech = LayerNorm(speech_encoder_config['codebook_size'])  # Qformer
        self.qformer_proj = nn.Linear(qformer.config.hidden_size, llama_model.config.hidden_size)
        
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None


        if args.qformer_checkpoint_path is not None:
            self.load_from_pretrained(args.qformer_checkpoint_path)

        self.device = self.query_tokens.device


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--number-of-queries",
            type=int,
            help="Number of queries in Qformer model",
        )
        parser.add_argument(
            "--max-length",
            type=int,
            help="Number length for decoding",
        )
        parser.add_argument(
            "--qformer-checkpoint-path",
            type=str,
            default=None,
            help="Number length for decoding",
        )
        parser.add_argument(
            "--freeze-speech-encoder",
            type=bool,
            default=None,
            help="Number length for decoding",
        )
        parser.add_argument(
            "--freeze-qformer",
            type=bool,
            default=None,
            help="Number length for decoding",
        )
        parser.add_argument(
            "--freeze-llama",
            type=bool,
            default=None,
            help="Number length for decoding",
        )
        parser.add_argument(
            "--speechtokenizer-configpath",
            type=str,
            default=None,
            help="Number length for decoding",
        )
        parser.add_argument(
            "--speechtokenizer-ckptpath",
            type=str,
            default=None,
            help="Number length for decoding",
        )
        parser.add_argument(
            "--llama_model",
            type=str,
            default=None,
            help="Number length for decoding",
        )

        
    @classmethod
    def build_speech_model(cls, args, dictionary=None, embed_tokens=None):
        speech_encoder = SpeechTokenizer.load_from_checkpoint(
            args['speech_model_config_path'], args['speech_model_file_path']
        )
        if hasattr(speech_encoder, "decoder"):   # Pruning
            del speech_encoder.decoder

        with open(args['speech_model_config_path']) as config_file:
            config = json.load(config_file)

        return speech_encoder, config

    @classmethod
    def build_text_encoder_prenet(cls, args):
        return TextEncoderPrenet(args)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        speech_encoder_prenet, speech_encoder_config = cls.build_speech_model({
            'speech_model_file_path': args.speechtokenizer_ckptpath,
            'speech_model_config_path': args.speechtokenizer_configpath
        })
        if hasattr(speech_encoder_prenet, "decoder"): 
            del speech_encoder_prenet.decoder

        qformer, qformer_query_tokens = cls.build_qformer(
            args.number_of_queries, speech_encoder_config['codebook_size'], drop=True
        )
        
        llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

        llama_model = LlamaForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.float16,
        )

        if args.freeze_speech_encoder:
            logging.info("freezing SpeechTokenizer")
            speech_encoder_prenet.eval()
            for name, param in tqdm(speech_encoder_prenet.named_parameters()):
                param.requires_grad = False

        if args.freeze_qformer:
            logging.info("freezing Qformer")
            for name, param in tqdm(qformer.named_parameters()):
                param.requires_grad = False
            qformer = qformer.eval()
            qformer.train = disabled_train
            qformer_query_tokens.requires_grad = False
        
        if args.freeze_llama:
            logging.info("freezing Llama")
            for name, param in tqdm(llama_model.named_parameters()):
                param.requires_grad = False

            llama_model = llama_model.eval()

        return cls(
            args,
            speech_encoder_prenet,
            speech_encoder_config,
            qformer,
            qformer_query_tokens,
            llama_tokenizer,
            llama_model

        )

    @classmethod
    def build_qformer(cls, num_query_token, modality_width, cross_attention_freq=2, drop=True):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = modality_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        if not drop:
            encoder_config.hidden_dropout_prob = 0.0
            encoder_config.attention_probs_dropout_prob = 0.0
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def get_normalized_probs_for_ctc(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out_for_ctc"][0]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, sample, net_output, is_masked=True):
        if "logit_m_list" in net_output:
            logits_list = self.get_logits(net_output, is_masked)
            targets_list = [
                x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list
            ]
            return targets_list
        else:
            return sample["target"]

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        if "prob_perplexity" in net_output:
            extra_losses.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )
            names.append("prob_perplexity")

        return extra_losses, names
        

    def forward(
            self, 
            source, # collated speech
            target,
            padding_mask=None,
            task_name=None
        ):
        batch_size = source.shape[0]
        device = source[0].device
        with self.maybe_autocast():
            codes = self.speech_encoder.encode(source.unsqueeze(1), 1) # codes: (n_q, B, T)
            if torch.unique(codes).numel() < 0.20 * codes.shape[-1]:
                logging.info('Non-unique tensor is detected')
                raise NotImplemented

            speech_embeds = self.speech_encoder.quantizer.decode(codes).transpose(1, 2)
            speech_embeds = self.ln_speech(speech_embeds)  # from Qformer
            
        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(
            device
        )
        # copies qformer tokens for each element in batch

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True
        )    
        
        inputs_llama = self.qformer_proj(query_output.last_hidden_state)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            to_regress_tokens = self.llama_tokenizer(
                target[0],return_tensors="pt",padding="longest",truncation=True,max_length=self.args.max_length,add_special_tokens=False
            ).to(device)
            

        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)
        empty_targets = (torch.ones([atts_llama.shape[0], atts_llama.shape[1]+1],dtype=torch.long).to(device).fill_(-100))  # plus one for bos
        
        targets = torch.cat([empty_targets, targets], dim=1)
        
        bos = torch.ones([batch_size, 1],dtype=to_regress_tokens.input_ids.dtype, device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)

        atts_bos = atts_llama[:, :1]
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, inputs_llama, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_llama, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        # breakpoint()

        return {"loss": loss, 'output': outputs}

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


    def forward_encoder_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward_encoder(
                source=net_input["source"],
                padding_mask=net_input["padding_mask"]
            )
        else:
            return self.forward_encoder_non_torchscript(net_input)

    @torch.jit.unused
    def forward_encoder_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens" and k != "task_name"
        }
        return self.forward_encoder(**encoder_input)

    def forward_encoder(self, source, padding_mask=None):
        raise NotImplemented

    def forward_decoder(self, tokens, encoder_out, incremental_state):
       raise NotImplemented

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


@register_model_architecture(model_name="speech_adapter_llama", arch_name="speech_adapter_llama")
def base_architecture(args):
    pass   # no base architecture since we are using pre-trianed modules
