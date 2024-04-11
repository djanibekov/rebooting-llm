from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.tasks import FairseqTask

import torch
from fairseq import metrics, utils
import math

import logging
logger = logging.getLogger(__name__)


@register_criterion("speech_qformer")
class SpeechtoQformerLoss(FairseqCriterion):
    def __init__(
        self,
        task: FairseqTask
    ):

        super().__init__(task)

    def forward(self, model, sample):
        output = model(sample["net_input"])
        ntokens = sample["ntokens"]

        sample_size = len(sample["target"])
        loss = output['loss']
        # loss_itc = output['loss_itc']
        # loss_itm = output['loss_itm']
        # loss_lm = output['loss_lm']

        logging_output = {
            "loss": output['loss'].item(),
            "loss_itc": output['loss_itc'].item() if output.get('loss_itc', None) is not None else 0,
            "loss_itm": output['loss_itm'].item() if output.get('loss_itm', None) is not None else 0,
            "loss_lm": output['loss_lm'].item() if output.get('loss_lm', None) is not None else 0,
            "ntokens": sample["ntokens"],
            "nsentences": len(sample["target"][0]),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        loss_itc_sum = utils.item(sum(log.get("loss_itc", 0) for log in logging_outputs))
        loss_itm_sum = utils.item(sum(log.get("loss_itm", 0) for log in logging_outputs))
        loss_lm_sum = utils.item(sum(log.get("loss_lm", 0) for log in logging_outputs))

        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        
        metrics.log_scalar("sample_size", sample_size)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        metrics.log_scalar("loss_itc", loss_itc_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("loss_itm", loss_itm_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("loss_lm", loss_lm_sum / sample_size, sample_size, round=3)