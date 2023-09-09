import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

from transformers import Trainer, AutoConfig
from transformers import EvalPrediction

from utils import print_rank_0


def compute_metrics(prediction: EvalPrediction):
    logits = torch.from_numpy(prediction.predictions)
    scores = torch.from_numpy(prediction.label_ids)
    
    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2)  # [batch_size, num_sample, num_sample]

    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1. * (scores >= 0).unsqueeze(2)

    # calculate accuracy...
    pred_compare = (logits_diff.detach() > 0.) * 1.
    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask
    correct_compare = (pred_compare == score_mask_larger) * total_mask
    
    all_acc = correct_compare.sum() / total_mask.sum()
    first_two_acc =  (correct_compare[:, 0, 1]).sum() / (total_mask[:, 0, 1]).sum() 
    
    return {"Preference total Acc": all_acc.item(), "First-two Acc": first_two_acc.item()}



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
