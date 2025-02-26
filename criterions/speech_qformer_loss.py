from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.tasks import FairseqTask

from fairseq import metrics, utils

import logging
logger = logging.getLogger(__name__)


@register_criterion("sparqle")
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
        logging_output = {
            "loss": output['loss'].item(),
            "loss_stc": output['loss_stc'].item() if output.get('loss_stc', None) is not None else 0,
            "loss_stm": output['loss_stm'].item() if output.get('loss_stm', None) is not None else 0,
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
        loss_stc_sum = utils.item(sum(log.get("loss_stc", 0) for log in logging_outputs))
        loss_stm_sum = utils.item(sum(log.get("loss_stm", 0) for log in logging_outputs))
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
        metrics.log_scalar("loss_stc", loss_stc_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("loss_stm", loss_stm_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("loss_lm", loss_lm_sum / sample_size, sample_size, round=3)