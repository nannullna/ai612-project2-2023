# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import inspect
from argparse import Namespace

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import utils
from loggings import meters, metrics

class MultiTaskCriterion(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    @classmethod
    def build_criterion(cls, args: Namespace):
        """Construct a criterion from command-line args."""
        # arguments in the __init__.
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if (
                p.kind == p.POSITIONAL_ONLY
                or p.kind == p.VAR_POSITIONAL
                or p.kind == p.VAR_KEYWORD
            ):
                raise NotImplementedError("{} not supported".format(p.kind))

            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}

            if p.name == "args":
                init_args["args"] = args
            elif hasattr(args, p.name):
                init_args[p.name] = getattr(args, p.name)
            elif p.default != p.empty:
                pass
            else:
                raise NotImplementedError(
                    "Unable to infer Criterion arguments, please implement "
                    "{}.build_criterion".format(cls.__name__)
                )
        return cls(**init_args)
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        
        Returns a tuple with three elements.
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample)
        
        logits = model.get_logits(net_output).float() # (batch, 52)
        target = model.get_targets(sample) # (batch, 28)

        assert logits.dim() == 2 and target.dim() == 2

        reduction = "none" if not reduce else "mean"
        sample_size = 1
        
        binary_weights = torch.tensor([[122.4],[11.72],[12.79],[3.28],[5.13],[1.01],[2.52],[2.47],[2.69],[0.44],
                                       [1.25],[2.14],[1.8],[247.3],[15.09],[6.1],[51.43],[2.46],[2.77],[2.25],[1.67],[7.45]], device='cuda:0')

        multi_weights = [[0.03, 0.37, 0.23, 0.11, 0.18, 0.08], [0.02, 0.01, 0.0, 0.03, 0.76, 0.19], [0.02, 0.06, 0.12, 0.37, 0.44],
                         [0.02, 0.1, 0.09, 0.33, 0.46], [0.01, 0.02, 0.04, 0.14, 0.79],[0.84, 0.06, 0.1]]
        
        
        idx = 22
        binary_target = target[:, :idx]
        binary_logits = logits[:, :idx]
        loss = 0
        for i in range(idx):
            loss += F.binary_cross_entropy_with_logits(
                        binary_logits[:,i:i+1],
                        binary_target[:,i:i+1].float(),
                        reduction=reduction,
                        pos_weight = binary_weights[i]
                    )
        loss = loss/idx
        # loss = F.binary_cross_entropy_with_logits(
        #     binary_logits,
        #     binary_target.float(),
        #     reduction=reduction
        # )

        num_classes = [6, 6, 5, 5, 5, 3]
        multi_loss=0
        weight_idx = 0
        for i, num_class in enumerate(num_classes, start=idx):
            multi_class_target = target[:, i]
            multi_class_logits = logits[:, idx: idx + num_class]
            
            # tmp_loss = F.cross_entropy(
            #     input=multi_class_logits,
            #     target=multi_class_target,
            #     reduction=reduction,
            #     ignore_index=-1
            # )
            # if torch.isfinite(tmp_loss): # 추가
            #     loss += tmp_loss 


            # loss_list = F.cross_entropy(
            #     input=multi_class_logits,
            #     target=multi_class_target,
            #     reduction='none',
            #     ignore_index=-1
            # )
            # loss_mask = torch.isfinite(loss_list)
            # loss += torch.sum(loss_list * loss_mask).item()

            multi_loss += F.cross_entropy(
                input=multi_class_logits,
                target=multi_class_target,
                reduction=reduction,
                weight=torch.tensor(multi_weights[weight_idx], device='cuda:0'),
                ignore_index=-1
            )
            idx += num_class
            weight_idx +=1
        loss += (multi_loss / len(num_classes))
        
        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "batch_size": len(target),
            "sample_size": sample_size
        }

        with torch.no_grad():
            idx = 22

            y_class = np.array(sum([[i for _ in range(len(target))] for i in range(idx)], []))
            y_true = target.T[:idx].flatten().cpu().numpy()
            y_score = torch.sigmoid(logits).T[:idx].flatten().cpu().numpy()

            logging_output["binary_y_true"] = y_true
            logging_output["binary_y_score"] = y_score
            logging_output["binary_y_class"] = y_class

            for i, num_class in enumerate(num_classes, start=idx):
                y_true = target[:, i].cpu().numpy()
                multiclass_logits = logits[:, idx: idx + num_class]
                multiclass_y_score = torch.softmax(multiclass_logits, dim=-1).cpu().numpy()

                activated_idx = np.where(y_true != -1)
                y_true = y_true[activated_idx]
                multiclass_y_score = multiclass_y_score[activated_idx]
                logging_output[f"multiclass_y_true_{i}"] = y_true
                logging_output[f"multiclass_y_score_{i}"] = multiclass_y_score

                idx += num_class

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        batch_size = utils.item(
            sum(log.get("batch_size", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("batch_size", batch_size)

        if "binary_y_true" in logging_outputs[0] and "binary_y_score" in logging_outputs[0]:
            y_true = np.concatenate(
                [log["binary_y_true"] for log in logging_outputs if "binary_y_true" in log]
            )
            y_score = np.concatenate(
                [log["binary_y_score"] for log in logging_outputs if "binary_y_score" in log]
            )
            y_class = np.concatenate(
                [log["binary_y_class"] for log in logging_outputs if "binary_y_class" in log]
            )
            
            metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true, y_class)

        builtin_keys = {
            "loss",
            "batch_size",
            "sample_size",
            "binary_y_true",
            "binary_y_score"
            "binary_y_class"
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                if k.startswith("multiclass"):
                    y_true = np.concatenate(
                        [
                            log["multiclass_y_true_" + k[-2:]] for log in logging_outputs
                            if "multiclass_y_true_" + k[-2:] in log
                        ]
                    )
                    y_score = np.concatenate(
                        [
                            log["multiclass_y_score_" + k[-2:]] for log in logging_outputs
                            if "multiclass_y_score_" + k[-2:] in log
                        ]
                    )
                    builtin_keys.add("multiclass_y_true_" + k[-2:])
                    builtin_keys.add("multiclass_y_score_" + k[-2:])
                    
                    metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true, cls=int(k[-2:]), multiclass=True)