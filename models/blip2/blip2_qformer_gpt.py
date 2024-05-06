"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
import logging
from packaging import version

import torch
import torch.nn as nn
from fairseq.models import register_model, register_model_architecture

from .blip2 import Blip2Base, disabled_train
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModel
import transformers

from ..speechtokenizer.model import SpeechTokenizer

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)

@register_model("speech_qformer_base_gpt")
class Blip2GPT(Blip2Base):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        if args.speech_encoder_model == 'speechtokenizer':
            speech_encoder, config = cls.build_speech_model_speechtoknenizer({
                'speech_model_file_path': args.speechtokenizer_ckptpath,
                'speech_model_config_path': args.speechtokenizer_configpath,
                'freeze_enc': True
            }) 
            codebook_size = config['codebook_size']
        elif args.speech_encoder_model == 'hubertbase':
            speech_encoder, config = cls.build_speech_model_hubertbase({
                'speech_model_file_path': args.speechtokenizer_ckptpath,
                'speech_model_config_path': args.speechtokenizer_configpath,
                'freeze_enc': True
            })
            codebook_size = config.hidden_size
        
        elif args.speech_encoder_model == 'hubertlarge':
            speech_encoder, config = cls.build_speech_model_hubertlarge({
                'speech_model_file_path': args.speechtokenizer_ckptpath,
                'speech_model_config_path': args.speechtokenizer_configpath,
                'freeze_enc': True
            }) 
            codebook_size = config.hidden_size
        else:
            raise NotImplemented   
        Qformer, query_tokens, tokenizer = cls.build_qformer({
            'codebook_size': codebook_size,
            'cross_attention_freq': args.cross_attention_freq,
            'num_query_token': args.num_query_token
        })

        return cls(
            args,
            speech_encoder,
            Qformer, 
            query_tokens,
            tokenizer,
            config,
            codebook_size
        )

    @classmethod
    def build_speech_model_speechtoknenizer(cls, args, dictionary=None, embed_tokens=None):
        speech_encoder = SpeechTokenizer.load_from_checkpoint(
            args['speech_model_config_path'], args['speech_model_file_path']
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
    def build_speech_model_hubertbase(cls, args, dictionary=None, embed_tokens=None):
        speech_encoder = AutoModel.from_pretrained("facebook/hubert-base-ls960")

        if args['freeze_enc']:
            for name, param in speech_encoder.named_parameters():
                param.requires_grad = False
            speech_encoder.eval()

        return speech_encoder, speech_encoder.config
    
    @classmethod
    def build_speech_model_hubertlarge(cls, args, dictionary=None, embed_tokens=None):
        speech_encoder = AutoModel.from_pretrained("facebook/hubert-large-ll60k")

        if args['freeze_enc']:
            for name, param in speech_encoder.named_parameters():
                param.requires_grad = False
            speech_encoder.eval()

        return speech_encoder, speech_encoder.config
    
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
        parser.add_argument("--cross-attention-freq", type=int)
        parser.add_argument("--num-query-token", type=int)
        parser.add_argument("--speech-model-config-path", type=str)
        parser.add_argument("--speech-model-file-path", type=str)
        parser.add_argument("--qformer-dim", type=str)
        parser.add_argument("--gpt-model", type=str)
        parser.add_argument("--pretrained", type=str)
        parser.add_argument("--speech-encoder-model", type=str)
       

    def __init__(
        self,
        args,
        speech_encoder,
        Qformer, 
        query_tokens,
        tokenizer,
        speech_config,
        codebook_size,
        embed_dim=256,
        freeze_encoder=True,
        prompt=""
    ):
        super().__init__()
        self.speech_encoder_model = args.speech_encoder_model
        self.tokenizer = self.init_tokenizer()

        self.speech_encoder = speech_encoder
        self.ln_speech = LayerNorm(codebook_size)
        if freeze_encoder:
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder = self.speech_encoder.eval()
            self.speech_encoder.train = disabled_train
            logging.info("freeze speech encoder")

        self.Qformer, self.query_tokens = Qformer, query_tokens
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_model, use_fast=False)
        if self.gpt_tokenizer.pad_token is None:
            self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        self.gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_model)
        
        for name, param in self.gpt_model.named_parameters():
            param.requires_grad = False
    
        self.eos_token_id = self.gpt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.gpt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.gpt_model.config.hidden_size
        )

        self.prompt = prompt
        prompt_tokens = self.gpt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self.speechtokenizer_padding_idx = 931

        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % args.pretrained)
      

    def forward(self, samples):
        speech = samples["source"]
        if self.speech_encoder_model == 'speechtokenizer':
            speech_embeds, codes = self.speech_encoder.forward_feature(speech.unsqueeze(1), [0])
            speech_embeds = self.ln_speech(speech_embeds[0].transpose(1, 2))

            speech_atts = codes[0] != self.speechtokenizer_padding_idx
        elif self.speech_encoder_model == 'hubertbase' or self.speech_encoder_model == 'hubertlarge':
            output = self.speech_encoder(speech, attention_mask=samples['padding_mask'])
            speech_embeds = output.last_hidden_state
            speech_embeds = self.ln_speech(speech_embeds)
            
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(
                speech.device
            )
        else:
            raise NotImplemented

        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )

        inputs_gpt = self.gpt_proj(query_output.last_hidden_state)
        atts_gpt = torch.ones(inputs_gpt.size()[:-1], dtype=torch.long).to(speech.device)

        self.gpt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["target"][0]]

        gpt_tokens = self.gpt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
        ).to(speech.device)

        targets = gpt_tokens.input_ids
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_gpt.size(), dtype=torch.long).to(speech.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.gpt_model.transformer.wte(gpt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_gpt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_gpt, gpt_tokens.attention_mask], dim=1)

        outputs = self.gpt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def forward_encoder(self, samples):
        speech = samples["source"]
        speech_embeds, codes = self.speech_encoder.forward_feature(speech.unsqueeze(1), [0])
        speech_embeds = self.ln_speech(speech_embeds[0].transpose(1, 2))

        speech_atts = codes[0] != self.speechtokenizer_padding_idx
        return speech_embeds, speech_atts

    @torch.no_grad()
    def forward_decoder(self, tokens, speech_embeds, speech_atts):
        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )

        inputs_gpt = self.gpt_proj(query_output.last_hidden_state)
        atts_gpt = torch.ones(inputs_gpt.size()[:-1], dtype=torch.long).to(
            speech.device
        )

        prompt = self.prompt + token # user defined prompt
        prompt = [prompt] * speech.size(0)

        gpt_tokens = tokens
        attention_mask = torch.cat([atts_gpt, gpt_tokens.attention_mask], dim=1)
        
        inputs_embeds = self.gpt_model.get_input_embeddings()(gpt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_gpt, inputs_embeds],dim=1)
        
        outputs = self.gpt_model.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        output_text = self.gpt_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        output_text = [text.strip() for text in output_text]
        return output_text


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=30,
        max_length=100,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        speech = samples["source"]
        if self.speech_encoder_model == 'speechtokenizer':
            speech_embeds, codes = self.speech_encoder.forward_feature(speech.unsqueeze(1), [0])
            speech_embeds = self.ln_speech(speech_embeds[0].transpose(1, 2))

            speech_atts = codes[0] != self.speechtokenizer_padding_idx
        elif self.speech_encoder_model == 'hubertbase' or self.speech_encoder_model == 'hubertlarge':
            output = self.speech_encoder(speech, attention_mask=samples['padding_mask'])
            speech_embeds = output.last_hidden_state
            speech_embeds = self.ln_speech(speech_embeds)
            
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(
                speech.device
            )
        else:
            raise NotImplemented

        

        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(query_embeds=query_tokens,encoder_hidden_states=speech_embeds,encoder_attention_mask=speech_atts,return_dict=True,)

        inputs_gpt = self.gpt_proj(query_output.last_hidden_state)
        atts_gpt = torch.ones(inputs_gpt.size()[:-1], dtype=torch.long).to(speech.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        prompt = [prompt] * speech.size(0)
        
        gpt_tokens = self.gpt_tokenizer(prompt,return_tensors="pt",).to(speech.device)
        attention_mask = torch.cat([atts_gpt, gpt_tokens.attention_mask], dim=1)
        
        inputs_embeds = self.gpt_model.get_input_embeddings()(gpt_tokens.input_ids.to(torch.int)).to(speech.device)
        inputs_embeds = torch.cat([inputs_gpt, inputs_embeds],dim=1)
        
        outputs = self.gpt_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,do_sample=use_nucleus_sampling,top_p=top_p,temperature=temperature,num_beams=num_beams,max_length=max_length,min_length=min_length,eos_token_id=self.eos_token_id,repetition_penalty=repetition_penalty,length_penalty=length_penalty,num_return_sequences=num_captions,)
        output_text = self.gpt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        output_text = [text.strip() for text in output_text]
        breakpoint()
        return output_text
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_gpt = self.gpt_proj(query_output.last_hidden_state)
            atts_gpt = torch.ones(inputs_gpt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.gpt_tokenizer.padding_side = "left"
            gpt_tokens = self.gpt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            attention_mask = torch.cat([atts_gpt, gpt_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.gpt_model.get_input_embeddings()(gpt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_gpt,inputs_embeds],dim=1)
            
            outputs = self.gpt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.gpt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        gpt_model = cfg.get("gpt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            gpt_model=gpt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model

@register_model_architecture(model_name="speech_qformer_base_gpt", arch_name="speech_qformer_base_gpt")
def base_architecture(args):
    pass