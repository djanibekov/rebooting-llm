from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.tasks import FairseqTask

import torch

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
        # ntokens = (
        #     sample["ntokens"] if "ntokens" in sample else sample["target_lengths"].sum().item()
        # )

        sample_size = len(sample["target"])
        loss = output['loss']
        breakpoint()

        logging_output = {
            "loss": output['loss'].item(),
            # "ntokens": sample["ntokens"],
            # "nsentences": sample["target"],
            "sample_size": sample_size,
        }

        if torch.isinf(loss):
            print("Loss is infinite, setting to zero.")
            loss = torch.tensor(0.0)

        return loss, sample_size, logging_output


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True