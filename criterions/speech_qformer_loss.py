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

    def forward(self, model, sample, reduce=True):
        output = model(sample["net_input"])
        ntokens = sample["ntokens"]

        sample_size = len(sample["target"])
        loss = output['loss']
        loss_itc = output['loss_itc']
        loss_itm = output['loss_itm']
        loss_lm = output['loss_lm']

        # breakpoint()

        logging_output = {
            "loss": output['loss'].item(),
            "loss_itc": output['loss_itc'].item(),
            "loss_itm": output['loss_itm'].item(),
            "loss_lm": output['loss_lm'].item(),
            "ntokens": sample["ntokens"],
            "nsentences": len(sample["target"][0]),
            "sample_size": sample_size,
        }
        # print(logging_output)

        # if torch.isinf(loss):
        #     raise NotImplemented

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

        # metrics.log_scalar(
        #     "ctc_loss", ctc_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
        # )
        # metrics.log_scalar(
        #     "ce_loss", ce_loss_sum / ntokens, ntokens, 2, round=3
        # )
        # metrics.log_scalar(
        #     "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, 2, round=3
        # )
        # metrics.log_derived(
        #     "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg, 2)
        # )
        
        # total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        # if total > 0:
        #     metrics.log_scalar("total", total)
        #     n_correct = utils.item(
        #         sum(log.get("n_correct", 0) for log in logging_outputs)
        #     )
        #     metrics.log_scalar("n_correct", n_correct)
        #     metrics.log_derived(
        #         "accuracy",
        #         lambda meters: round(
        #             meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
        #         )
        #         if meters["total"].sum > 0
        #         else float("nan"),
        #         2
        #     )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        # if sample_size != ntokens:
        #     metrics.log_scalar(
        #         "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
        #     )

        # c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        # metrics.log_scalar("_c_errors", c_errors)
        # c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        # metrics.log_scalar("_c_total", c_total)
        # w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        # metrics.log_scalar("_w_errors", w_errors)
        # wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        # metrics.log_scalar("_wv_errors", wv_errors)
        # w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        # metrics.log_scalar("_w_total", w_total)

        # if c_total > 0:
        #     metrics.log_derived(
        #         "uer",
        #         lambda meters: safe_round(
        #             meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
        #         )
        #         if meters["_c_total"].sum > 0
        #         else float("nan"),
        #     )
        # if w_total > 0:
        #     metrics.log_derived(
        #         "wer",
        #         lambda meters: safe_round(
        #             meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
        #         )
        #         if meters["_w_total"].sum > 0
        #         else float("nan"),
        #     )
        #     metrics.log_derived(
        #         "raw_wer",
        #         lambda meters: safe_round(
        #             meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
        #         )
        #         if meters["_w_total"].sum > 0
        #         else float("nan"),
        #     )