from typing import List, Optional, Tuple, Union

from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class CustomTrainingArguments(TrainingArguments):
    # experiment setups
    reward_domain: str = field(
        default="normal", 
        metadata={"help": "the domain for reward model training."}
    )
    # tokenizer params
    padding_side: str = field(
        default="right",
        metadata={"help": "the direction for tokenizer to add padding tokens."}
    )

    truncation_side: str = field(
        default="left",
        metadata={"help": "the direction for tokenizer to add padding tokens."}
    )

    # model params
    model_type: str = field(
        default="llama",
        metadata={"help": "the base model type for reward model, selected from [llama, bert]."}
    )

    pooling_type: str = field(
        default="average",
        metadata={"help": "the pooling method for reward model, selected from [average, max, last]."}
    )

    model_name_or_path: str = field(
        default="llama-7b-hf", 
        metadata={"help": "the path to load pretrained model."}
    )

    tokenizer_path: str = field(
        default="llama-7b-hf", 
        metadata={"help": "the path to load pretrained tokenizer."}
    )

    # data params

    max_response_num: int = field(
        default=1,
        metadata={"help": "the maximum response number of each data item"}
    )


    data_dir: str = field(
        default="path/to/cleaned_data",
        metadata={"help": "the directory to load data."}
    )   

    data_path: str = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "the path to load data."}
    )   

    train_data_path: List[str] = field(
        default_factory=lambda: ["/data/to/train/dataset"],
        metadata={"help": "train datasets paths."}
    )


    eval_data_path: List[str] = field(
        default_factory=lambda: ["/data/to/eval/dataset"],
        metadata={"help": "evaluation datasets paths."}
    )


    data_prefix: str = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "the prefix to load train and test data."}
    )   

    # training hyperparams
    eval_at_start: bool = field(
        default=False,
        metadata={"help": "whether make eval at start."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "whether use the debug mode."}
    )

    cache_dir: Optional[str] = field(default=None)

    optim: str = field(default="adamw_torch")

    lm_loss_coeff: float = field(default=0., metadata={"help": "the coefficient for language modeling loss."})

    contrast_loss_coeff: float = field(default=0., metadata={"help": "the coefficient for contrastive learning loss."})

    lm_score_thresh: float = field(default=0.85, metadata={"help": "the threshold to select response for language modeling"})

    max_length: int = field(
        default=256,
        metadata={"help": "the max sentence sequence length."}
    )   

    batch_size: int = field(
        default=256,
        metadata={"help": "the overall training batch size"}
    )   

    micro_batch_size: int = field(
        default=32,
        metadata={"help": "the batch size on each device, equavilent to `per_gpu_train_batch_size`"}
    )


    valid_data_size: int = field(
        default=0,
        metadata={"help": "the data size for validation data"}
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None, 
        metadata={"help":  "either training checkpoint or final adapter"}
    )

    # lora hyperparams
    lora_r: int = field(
        default=8,
        metadata={"help": "parameter r for lora."}
    )

    lora_alpha: int = field(
        default=16,
        metadata={"help": "parameter alpha for lora."}
    )

    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "dropout rate for lora."}
    )

    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj","v_proj"],
        metadata={"help": "target modules for lora optimization."}
    )

    rm_calibration: bool = field(default=True)

    calibration_bins: List[int] = field(
        default_factory=lambda: [10, 15]
    )
