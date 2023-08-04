import argparse

deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 0,
        "reduce_bucket_size": "auto",
    },
    "fp16": {
        "enabled": False,
        "auto_cast": False,
        "loss_scale": 0,
        "initial_scale_power": 32,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock": False,
}

trainer_args = {
    "per_device_train_batch_size": 1,
    "num_train_epochs": 2,
    "learning_rate": 2e-5,
    "optim": "adamw_torch",
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",

    "do_eval": True,
    "per_device_eval_batch_size": 1,
    "evaluation_strategy": "epoch",
    # "eval_steps": 60,

    "report_to": "none",
    "logging_dir": "./output/train.log",
    "log_level": "debug",
    "logging_steps": 2,

    "save_strategy": "epoch",
    "output_dir": "./output/model",

    "gradient_checkpointing": False,
    "deepspeed": deepspeed_config,
}

def str2bool(v):
    if type(v) is bool:
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_load_from_disk", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=True)
    args = parser.parse_args()
    return args