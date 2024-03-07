import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
import numpy as np

from transformers import Trainer, AutoConfig
from transformers import EvalPrediction

from utils import print_rank_0


def compute_ece(y_true, y_prob, n_bins=5, strategy="uniform"):
    if len(y_true) == 0:
        return 0., 0., 0.
    
    if strategy == "quantile":
        quantiles = np.linspace(0., 1., n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0., 1., n_bins+1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy must be either 'quantile' or 'uniform'."
        )
    
    # the ith element in binids indicate which bin the ith element in y_prob belong to.
    binids = np.searchsorted(bins[1:-1], y_prob)

    # the ith element in bin_sums is the average probability of positive examples that model predict in the ith bin
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    # the ith element in bin_true is the real probablility of positive examples in the ith bin
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    # the ith element in bin_total is the total num of examples belong to the ith bin
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0

    try:
        expected_error = np.abs(bin_sums - bin_true).sum() / len(y_prob)
        average_error = (np.abs(bin_sums[nonzero] - bin_true[nonzero]) / bin_total[nonzero]).mean()
        max_error = (np.abs(bin_sums[nonzero] - bin_true[nonzero]) / bin_total[nonzero]).max()
    except:
        expected_error, average_error, max_error = 0., 0., 0.
    return expected_error, average_error, max_error

def rm_calibration_errors(args, labels: torch.Tensor, probs: torch.Tensor, masks: torch.Tensor, num_bins):
    label_list = labels.reshape(-1).tolist()
    prob_list = probs.reshape(-1).tolist()
    mask_list = masks.reshape(-1).tolist()

    y_true, y_prob = [], []
    for label, prob, mask in zip(label_list, prob_list, mask_list):
        if mask:
            y_true.append(label)
            y_prob.append(prob)
    
    if args.debug_mode:
        print_rank_0(f">>> Check calibration inputs mask filtered...")
        print_rank_0(f">>>>>>>>> y_true: {y_true[:10]}")
        print_rank_0(f">>>>>>>>> y_prob: {y_prob[:10]}")
    
    return compute_ece(np.array(y_true), np.array(y_prob), n_bins=num_bins)
    

def compute_metrics(args, predict: EvalPrediction):
    logits = torch.from_numpy(predict.predictions) # (batch_size, num_sample)
    scores = torch.from_numpy(predict.label_ids) # (batch_size, num_sample)

    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2) # shape: (batch_size, num_sample, num_sample)
    
    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1. * (scores >= 0).unsqueeze(2)

    # caculate accuracy
    pred_compare = ((logits_diff * score_mask).detach() > 0.) * 1.
    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask
    correct_compare = pred_compare * total_mask

    all_acc = correct_compare.sum() / total_mask.sum()
    first_two_acc = (correct_compare[:, 0, 1]).sum() / (total_mask[:, 0, 1]).sum()

    # caculate ece
    calibration_errors = {}
    if args.rm_calibration:
        for num_bins in args.calibration_bins:
            expected_error, average_error, max_error = rm_calibration_errors(
                args=args,
                labels=score_mask_larger,
                probs=F.sigmoid(logits_diff),
                masks=total_mask,
                num_bins=num_bins
            )

            calibration_errors[f"calibration_ECE_bin{num_bins}"] = expected_error
            #calibration_errors[f"calibration_ACE_bin{num_bins}"] = average_error
            #calibration_errors[f"calibration_MCE_bin{num_bins}"] = max_error
    
    if args.debug_mode:
        print_rank_0(f">>> Check eval_prediction outputs...")
        print_rank_0(f">>> correct_compare: {correct_compare}")
        print_rank_0(f">>> total_mask: {total_mask}")
        print_rank_0(f">>> all_acc: {all_acc}")
        print_rank_0(f">>> calibration error: {calibration_errors}")

    return {"Preference total Acc": all_acc.item(), "First-two Acc": first_two_acc.item(), **calibration_errors}



def language_modeling_loss(lm_logits, input_ids, scores, loss_mask, score_thresh=0.9, eps=1e-7): 
    batch_size, seq_length, vocab_size = lm_logits.shape
    
    lm_probs = torch.nn.functional.cross_entropy(
        input=lm_logits[:, :-1,:].reshape(-1, vocab_size), 
        target=input_ids[:, 1:].reshape(-1),
        reduction='none'
    ).view(batch_size, -1)

    loglikeli = (lm_probs * loss_mask[:, 1:].float()).sum(dim=-1) / loss_mask[:, 1:].float().sum(dim=-1)
    score_mask = (scores.reshape(-1) > score_thresh).float()
    return (loglikeli * score_mask).sum() / (score_mask.sum() + eps)


def ranking_loss(logits, scores): # with shape [bs, r]
    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2)

    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1. * (scores >= 0).unsqueeze(2)

    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask

    log_prob = torch.nn.functional.logsigmoid(logits_diff * score_mask * pad_mask)

    total_loss = - (log_prob * total_mask).sum()
    total_pairs = total_mask.sum()

    return  total_loss / total_pairs  if total_pairs > 0 else total_loss


def gather_all_with_local_grad(tensor, dim=0):
    local_rank = torch.distributed.get_rank()

    with torch.no_grad():
        all_tensors = [torch.zero_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(all_tensors, tensor)
    all_tensors[local_rank] = tensor

    return torch.stack(all_tensors, dim=dim)
    

class RewardModelTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[List[str]] = None):
        device = model.device
        labels = inputs['score'].to(device)

        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)

                
    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.device
        scores  = inputs['score'].to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        batch_size, sample_num, seq_length = input_ids.shape
        
        if self.args.debug_mode:
            print(f">>> input_ids shape {input_ids.shape}")
    
        outputs = model(
            input_ids=input_ids.view(-1, seq_length),
            attention_mask=attention_mask.view(-1, seq_length),
            padding_side=self.args.padding_side,
            pooling_type=self.args.pooling_type
        )

        hidden_states = outputs['hidden_states'] # shape [bs*r, seq_length, dim]
        
        batch_logits = outputs['rm_logits'].view(batch_size, sample_num)

        rm_loss = ranking_loss(batch_logits, scores)

        lm_loss = language_modeling_loss(
            lm_logits=outputs['lm_logits'], 
            input_ids=input_ids.view(-1, seq_length), 
            scores=scores, 
            loss_mask=attention_mask.view(-1,seq_length), 
            score_thresh=self.args.lm_score_thresh
        )

        total_loss = rm_loss + self.args.lm_loss_coeff * lm_loss

        if self.args.debug_mode:
            print_rank_0(f">>> debug")
            print_rank_0(f">>> Language modeling loss {lm_loss}")
            print_rank_0(f">>> Ranking loss {rm_loss}")
        
        return (total_loss, batch_logits) if return_outputs else total_loss            
