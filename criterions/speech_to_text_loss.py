from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.tasks import FairseqTask

import torch
from fairseq import metrics, utils
import math

import logging
logger = logging.getLogger(__name__)


@register_criterion("speech_adapter")
class SpeechtoTextLoss(FairseqCriterion):
    def __init__(
        self,
        task: FairseqTask
    ):

        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        output = model(**sample["net_input"])
        ntokens = sample["ntokens"]

        sample_size = len(sample["target"])
        loss = output['loss']

        logging_output = {
            "loss": output['loss'].item(),
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
        # nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        # ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        # ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
    