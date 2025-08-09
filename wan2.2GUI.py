import gradio as gr
import toml
import os
import subprocess
import threading
import queue

# 默认配置文件路径
BASE_DIR = "/workspace/diffusion-pipe"  # diffusion-pipe 的根目录
TRAIN_SCRIPT = "/root/diffusion-pipe/train.py"  # 训练脚本路径
DATASET_CONFIG = "/root/diffusion-pipe/examples/dataset.toml"  # 数据集配置文件
I2V_A14B_HIGH_TRAIN_CONFIG = "/root/diffusion-pipe/examples/wan2.2_i2v_a14b-high.toml"  # i2v (A14B) 高噪训练配置文件
I2V_A14B_LOW_TRAIN_CONFIG = "/root/diffusion-pipe/examples/wan2.2_i2v_a14b-low.toml"    # i2v (A14B) 低噪训练配置文件
T2V_A14B_HIGH_TRAIN_CONFIG = "/root/diffusion-pipe/examples/wan2.2_t2v_a14b-high.toml"  # t2v (A14B) 高噪训练配置文件
T2V_A14B_LOW_TRAIN_CONFIG = "/root/diffusion-pipe/examples/wan2.2_t2v_a14b-low.toml"    # t2v (A14B) 低噪训练配置文件
T2V_5B_TRAIN_CONFIG = "/root/diffusion-pipe/examples/wan2.2_t2v_5b.toml"  # t2v (5B) 训练配置文件
T2I_5B_TRAIN_CONFIG = "/root/diffusion-pipe/examples/wan2.2_t2i_5b.toml"  # t2i (5B) 训练配置文件

# i2v (A14B) 默认训练配置（高噪）
i2v_a14b_high_default_train_config = {
    "output_dir": "/data/diffusion_pipe_training_runs/wan2.2_lora_i2v_a14b_high",
    "dataset": "/root/diffusion-pipe/examples/dataset.toml",
    "epochs": 1000,
    "micro_batch_size_per_gpu": 1,
    "pipeline_stages": 1,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "warmup_steps": 100,
    "blocks_to_swap": 0,
    "eval_every_n_epochs": 1,
    "eval_before_first_step": True,
    "eval_micro_batch_size_per_gpu": 1,
    "eval_gradient_accumulation_steps": 1,
    "save_every_n_epochs": 2,
    "checkpoint_every_n_minutes": 120,
    "activation_checkpointing": True,
    "save_dtype": "bfloat16",
    "caching_batch_size": 1,
    "video_clip_mode": "single_beginning",
    "partition_method": "parameters",
    "partition_split": [],
    "disable_block_swap_for_eval": False,
    "eval_datasets": [],
    "model": {
        "type": "wan",
        "ckpt_path": "/root/diffusion-pipe/wan2.2/Wan2.2-I2V-A14B",
        "transformer_path": "/root/diffusion-pipe/wan2.2/Wan2.2-I2V-A14B/high_noise_model",
        "dtype": "bfloat16",
        "transformer_dtype": "float8",
        "timestep_sample_method": "logit_normal",
        "min_t": 0.9,
        "max_t": 1.0,
        "llm_path": ""
    },
    "adapter": {
        "type": "lora",
        "rank": 32,
        "dtype": "bfloat16",
        "init_from_existing": ""
    },
    "optimizer": {
        "type": "adamw_optimi",
        "lr": 2e-5,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01,
        "eps": 1e-8,
        "gradient_release": False
    }
}

# i2v (A14B) 默认训练配置（低噪）
i2v_a14b_low_default_train_config = {
    "output_dir": "/data/diffusion_pipe_training_runs/wan2.2_lora_i2v_a14b_low",
    "dataset": "/root/diffusion-pipe/examples/dataset.toml",
    "epochs": 1000,
    "micro_batch_size_per_gpu": 1,
    "pipeline_stages": 1,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "warmup_steps": 100,
    "blocks_to_swap": 0,
    "eval_every_n_epochs": 1,
    "eval_before_first_step": True,
    "eval_micro_batch_size_per_gpu": 1,
    "eval_gradient_accumulation_steps": 1,
    "save_every_n_epochs": 2,
    "checkpoint_every_n_minutes": 120,
    "activation_checkpointing": True,
    "save_dtype": "bfloat16",
    "caching_batch_size": 1,
    "video_clip_mode": "single_beginning",
    "partition_method": "parameters",
    "partition_split": [],
    "disable_block_swap_for_eval": False,
    "eval_datasets": [],
    "model": {
        "type": "wan",
        "ckpt_path": "/root/diffusion-pipe/wan2.2/Wan2.2-I2V-A14B",
        "transformer_path": "/root/diffusion-pipe/wan2.2/Wan2.2-I2V-A14B/low_noise_model",
        "dtype": "bfloat16",
        "transformer_dtype": "float8",
        "timestep_sample_method": "logit_normal",
        "min_t": 0,
        "max_t": 0.9,
        "llm_path": ""
    },
    "adapter": {
        "type": "lora",
        "rank": 32,
        "dtype": "bfloat16",
        "init_from_existing": ""
    },
    "optimizer": {
        "type": "adamw_optimi",
        "lr": 2e-5,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01,
        "eps": 1e-8,
        "gradient_release": False
    }
}

# t2v (A14B) 默认训练配置（高噪）
t2v_a14b_high_default_train_config = {
    "output_dir": "/data/diffusion_pipe_training_runs/wan2.2_lora_t2v_a14b_high",
    "dataset": "/root/diffusion-pipe/examples/dataset.toml",
    "epochs": 1000,
    "micro_batch_size_per_gpu": 1,
    "pipeline_stages": 1,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "warmup_steps": 100,
    "blocks_to_swap": 0,
    "eval_every_n_epochs": 1,
    "eval_before_first_step": True,
    "eval_micro_batch_size_per_gpu": 1,
    "eval_gradient_accumulation_steps": 1,
    "save_every_n_epochs": 2,
    "checkpoint_every_n_minutes": 120,
    "activation_checkpointing": True,
    "save_dtype": "bfloat16",
    "caching_batch_size": 1,
    "video_clip_mode": "single_beginning",
    "partition_method": "parameters",
    "partition_split": [],
    "disable_block_swap_for_eval": False,
    "eval_datasets": [],
    "model": {
        "type": "wan",
        "ckpt_path": "/root/diffusion-pipe/wan2.2/Wan2.2-T2V-A14B",
        "transformer_path": "/root/diffusion-pipe/wan2.2/Wan2.2-T2V-A14B/high_noise_model",
        "dtype": "bfloat16",
        "transformer_dtype": "float8",
        "timestep_sample_method": "logit_normal",
        "min_t": 0.875,
        "max_t": 1.0,
        "llm_path": ""
    },
    "adapter": {
        "type": "lora",
        "rank": 32,
        "dtype": "bfloat16",
        "init_from_existing": ""
    },
    "optimizer": {
        "type": "adamw_optimi",
        "lr": 2e-5,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01,
        "eps": 1e-8,
        "gradient_release": False
    }
}

# t2v (A14B) 默认训练配置（低噪）
t2v_a14b_low_default_train_config = {
    "output_dir": "/data/diffusion_pipe_training_runs/wan2.2_lora_t2v_a14b_low",
    "dataset": "/root/diffusion-pipe/examples/dataset.toml",
    "epochs": 1000,
    "micro_batch_size_per_gpu": 1,
    "pipeline_stages": 1,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "warmup_steps": 100,
    "blocks_to_swap": 0,
    "eval_every_n_epochs": 1,
    "eval_before_first_step": True,
    "eval_micro_batch_size_per_gpu": 1,
    "eval_gradient_accumulation_steps": 1,
    "save_every_n_epochs": 2,
    "checkpoint_every_n_minutes": 120,
    "activation_checkpointing": True,
    "save_dtype": "bfloat16",
    "caching_batch_size": 1,
    "video_clip_mode": "single_beginning",
    "partition_method": "parameters",
    "partition_split": [],
    "disable_block_swap_for_eval": False,
    "eval_datasets": [],
    "model": {
        "type": "wan",
        "ckpt_path": "/root/diffusion-pipe/wan2.2/Wan2.2-T2V-A14B",
        "transformer_path": "/root/diffusion-pipe/wan2.2/Wan2.2-T2V-A14B/low_noise_model",
        "dtype": "bfloat16",
        "transformer_dtype": "float8",
        "timestep_sample_method": "logit_normal",
        "min_t": 0,
        "max_t": 0.875,
        "llm_path": ""
    },
    "adapter": {
        "type": "lora",
        "rank": 32,
        "dtype": "bfloat16",
        "init_from_existing": ""
    },
    "optimizer": {
        "type": "adamw_optimi",
        "lr": 2e-5,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01,
        "eps": 1e-8,
        "gradient_release": False
    }
}

# t2v (5B) 默认训练配置
t2v_5b_default_train_config = {
    "output_dir": "/data/diffusion_pipe_training_runs/wan2.2_lora_t2v_5b",
    "dataset": "/root/diffusion-pipe/examples/dataset.toml",
    "epochs": 1000,
    "micro_batch_size_per_gpu": 1,
    "pipeline_stages": 1,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "warmup_steps": 100,
    "blocks_to_swap": 0,
    "eval_every_n_epochs": 1,
    "eval_before_first_step": True,
    "eval_micro_batch_size_per_gpu": 1,
    "eval_gradient_accumulation_steps": 1,
    "save_every_n_epochs": 2,
    "checkpoint_every_n_minutes": 120,
    "activation_checkpointing": True,
    "save_dtype": "bfloat16",
    "caching_batch_size": 1,
    "video_clip_mode": "single_beginning",
    "partition_method": "parameters",
    "partition_split": [],
    "disable_block_swap_for_eval": False,
    "eval_datasets": [],
    "model": {
        "type": "wan",
        "ckpt_path": "/data/imagegen_models/Wan2.2-T2V-5B",
        "dtype": "bfloat16",
        "timestep_sample_method": "logit_normal",
        "min_t": 0,
        "max_t": 0.875,
        "llm_path": ""
    },
    "adapter": {
        "type": "lora",
        "rank": 32,
        "dtype": "bfloat16",
        "init_from_existing": ""
    },
    "optimizer": {
        "type": "adamw_optimi",
        "lr": 2e-5,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01,
        "eps": 1e-8,
        "gradient_release": False
    }
}

# t2i (5B) 默认训练配置
t2i_5b_default_train_config = {
    "output_dir": "/data/diffusion_pipe_training_runs/wan2.2_lora_t2i_5b",
    "dataset": "/root/diffusion-pipe/examples/dataset.toml",
    "epochs": 1000,
    "micro_batch_size_per_gpu": 1,
    "pipeline_stages": 1,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "warmup_steps": 100,
    "blocks_to_swap": 0,
    "eval_every_n_epochs": 1,
    "eval_before_first_step": True,
    "eval_micro_batch_size_per_gpu": 1,
    "eval_gradient_accumulation_steps": 1,
    "save_every_n_epochs": 2,
    "checkpoint_every_n_minutes": 120,
    "activation_checkpointing": True,
    "save_dtype": "bfloat16",
    "caching_batch_size": 1,
    "video_clip_mode": "single_beginning",
    "partition_method": "parameters",
    "partition_split": [],
    "disable_block_swap_for_eval": False,
    "eval_datasets": [],
    "model": {
        "type": "wan",
        "ckpt_path": "/data/imagegen_models/Wan2.2-T2I-5B",
        "dtype": "bfloat16",
        "timestep_sample_method": "logit_normal",
        "min_t": 0,
        "max_t": 0.875,
        "llm_path": ""
    },
    "adapter": {
        "type": "lora",
        "rank": 32,
        "dtype": "bfloat16",
        "init_from_existing": ""
    },
    "optimizer": {
        "type": "adamw_optimi",
        "lr": 2e-5,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01,
        "eps": 1e-8,
        "gradient_release": False
    }
}

# 默认数据集配置
default_dataset_config = {
    "resolutions": [512],
    "enable_ar_bucket": True,
    "min_ar": 0.5,
    "max_ar": 2.0,
    "num_ar_buckets": 7,
    "frame_buckets": [1, 33],
    "directory": [{"path": "/home/anon/data/images/grayscale", "num_repeats": 10}]
}

# 根据模型类型和噪点级别选择配置文件路径
def get_train_config_path(model_type, noise_level):
    config_paths = {
        "i2v (A14B)": {
            "high_noise": I2V_A14B_HIGH_TRAIN_CONFIG,
            "low_noise": I2V_A14B_LOW_TRAIN_CONFIG
        },
        "t2v (A14B)": {
            "high_noise": T2V_A14B_HIGH_TRAIN_CONFIG,
            "low_noise": T2V_A14B_LOW_TRAIN_CONFIG
        },
        "t2v (5B)": {"default": T2V_5B_TRAIN_CONFIG},
        "t2i (5B)": {"default": T2I_5B_TRAIN_CONFIG}
    }
    if model_type in ["t2v (5B)", "t2i (5B)"]:
        return config_paths[model_type]["default"]
    return config_paths[model_type][noise_level]

# 根据模型类型和噪点级别选择默认配置
def get_default_train_config(model_type, noise_level):
    default_configs = {
        "i2v (A14B)": {
            "high_noise": i2v_a14b_high_default_train_config,
            "low_noise": i2v_a14b_low_default_train_config
        },
        "t2v (A14B)": {
            "high_noise": t2v_a14b_high_default_train_config,
            "low_noise": t2v_a14b_low_default_train_config
        },
        "t2v (5B)": {"default": t2v_5b_default_train_config},
        "t2i (5B)": {"default": t2i_5b_default_train_config}
    }
    if model_type in ["t2v (5B)", "t2i (5B)"]:
        return default_configs[model_type]["default"]
    return default_configs[model_type][noise_level]

# 加载配置文件函数
def load_config(config_path, default_config):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return toml.load(f)
    return default_config

# 初始化配置
dataset_config = load_config(DATASET_CONFIG, default_dataset_config)
i2v_a14b_high_train_config = load_config(I2V_A14B_HIGH_TRAIN_CONFIG, i2v_a14b_high_default_train_config)
i2v_a14b_low_train_config = load_config(I2V_A14B_LOW_TRAIN_CONFIG, i2v_a14b_low_default_train_config)
t2v_a14b_high_train_config = load_config(T2V_A14B_HIGH_TRAIN_CONFIG, t2v_a14b_high_default_train_config)
t2v_a14b_low_train_config = load_config(T2V_A14B_LOW_TRAIN_CONFIG, t2v_a14b_low_default_train_config)
t2v_5b_train_config = load_config(T2V_5B_TRAIN_CONFIG, t2v_5b_default_train_config)
t2i_5b_train_config = load_config(T2I_5B_TRAIN_CONFIG, t2i_5b_default_train_config)
train_config = i2v_a14b_high_train_config  # 默认使用 i2v (A14B) 高噪配置

# 保存配置文件函数
def save_configs(train_config_dict, dataset_config_dict, model_type, noise_level):
    ckpt_path = train_config_dict["model"].get("ckpt_path", "")
    dataset_path = dataset_config_dict["directory"][0]["path"]
    if not os.path.exists(ckpt_path):
        return f"错误：检查点路径 {ckpt_path} 不存在！"
    if not os.path.exists(dataset_path):
        return f"错误：数据集目录 {dataset_path} 不存在！"
    if "A14B" in model_type:
        transformer_path = train_config_dict["model"].get("transformer_path", "")
        if not os.path.exists(transformer_path):
            return f"错误：Transformer 路径 {transformer_path} 不存在！"
    
    os.makedirs(os.path.dirname(DATASET_CONFIG), exist_ok=True)
    train_config_path = get_train_config_path(model_type, noise_level)
    os.makedirs(os.path.dirname(train_config_path), exist_ok=True)
    with open(train_config_path, "w") as f:
        toml.dump(train_config_dict, f)
    with open(DATASET_CONFIG, "w") as f:
        toml.dump(dataset_config_dict, f)
    return f"配置文件保存成功：\n- 训练配置：{train_config_path}\n- 数据集配置：{DATASET_CONFIG}\nLoRA 权重将保存到 {train_config_dict['output_dir']}/<date_time>/"

# 输入验证函数
def validate_partition_split(value, pipeline_stages, partition_method):
    if partition_method != "manual" or not value:
        return []
    try:
        split = [int(x) for x in value.split(",")]
        if len(split) != pipeline_stages - 1:
            raise ValueError(f"分割点数量 ({len(split)}) 必须等于流水线阶段数 - 1 ({pipeline_stages - 1})")
        return split
    except ValueError as e:
        raise gr.Error(f"无效的分区分割：{str(e)}")

def validate_eval_datasets(value):
    try:
        return toml.loads(value)["eval_datasets"] if value else []
    except Exception as e:
        raise gr.Error(f"无效的评估数据集格式：{str(e)}")

# 更新训练配置函数
def update_train_config(
    output_dir, dataset, epochs, micro_batch_size_per_gpu, pipeline_stages, gradient_accumulation_steps,
    gradient_clipping, warmup_steps, blocks_to_swap, eval_every_n_epochs, eval_before_first_step,
    eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, save_every_n_epochs,
    checkpoint_every_n_minutes, activation_checkpointing, save_dtype, caching_batch_size,
    video_clip_mode, ckpt_path, transformer_path, dtype, transformer_dtype, timestep_sample_method, min_t, max_t,
    rank, adapter_dtype, init_from_existing, optimizer_type, lr, betas_0, betas_1, weight_decay, eps,
    partition_method, partition_split, llm_path, eval_datasets, disable_block_swap_for_eval, gradient_release, model_type, noise_level
):
    output_dir = output_dir.replace("\\", "/")
    dataset = dataset.replace("\\", "/")
    ckpt_path = ckpt_path.replace("\\", "/")
    transformer_path = transformer_path.replace("\\", "/") if transformer_path else ""
    init_from_existing = init_from_existing.replace("\\", "/") if init_from_existing else ""
    llm_path = llm_path.replace("\\", "/") if llm_path else ""

    partition_split = validate_partition_split(partition_split, pipeline_stages, partition_method)
    eval_datasets = validate_eval_datasets(eval_datasets)

    config = {
        "output_dir": output_dir,
        "dataset": dataset,
        "epochs": epochs,
        "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "pipeline_stages": pipeline_stages,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "warmup_steps": warmup_steps,
        "blocks_to_swap": blocks_to_swap,
        "eval_every_n_epochs": eval_every_n_epochs,
        "eval_before_first_step": eval_before_first_step,
        "eval_micro_batch_size_per_gpu": eval_micro_batch_size_per_gpu,
        "eval_gradient_accumulation_steps": eval_gradient_accumulation_steps,
        "save_every_n_epochs": save_every_n_epochs,
        "checkpoint_every_n_minutes": checkpoint_every_n_minutes,
        "activation_checkpointing": activation_checkpointing,
        "save_dtype": save_dtype,
        "caching_batch_size": caching_batch_size,
        "video_clip_mode": video_clip_mode,
        "partition_method": partition_method,
        "partition_split": partition_split,
        "disable_block_swap_for_eval": disable_block_swap_for_eval,
        "eval_datasets": eval_datasets,
        "model": {
            "type": "wan",
            "ckpt_path": ckpt_path,
            "dtype": dtype,
            "timestep_sample_method": timestep_sample_method,
            "min_t": min_t,
            "max_t": max_t,
            "llm_path": llm_path if llm_path else None
        },
        "adapter": {
            "type": "lora",
            "rank": rank,
            "dtype": adapter_dtype,
            "init_from_existing": init_from_existing if init_from_existing else None
        },
        "optimizer": {
            "type": optimizer_type,
            "lr": float(lr),
            "betas": [betas_0, betas_1],
            "weight_decay": weight_decay,
            "eps": eps,
            "gradient_release": gradient_release
        }
    }
    if "A14B" in model_type:
        config["model"]["transformer_path"] = transformer_path if transformer_path else None
        config["model"]["transformer_dtype"] = transformer_dtype if transformer_dtype else None
    if not config["model"].get("transformer_path"):
        config["model"].pop("transformer_path", None)
    if not config["model"].get("transformer_dtype"):
        config["model"].pop("transformer_dtype", None)
    if not config["adapter"]["init_from_existing"]:
        config["adapter"].pop("init_from_existing", None)
    if not config["partition_split"]:
        config.pop("partition_split", None)
    if not config["model"]["llm_path"]:
        config["model"].pop("llm_path", None)
    return config

# 更新数据集配置函数
def update_dataset_config(resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, directory_path, num_repeats):
    directory_path = directory_path.replace("\\", "/")
    config = {
        "resolutions": [int(resolutions)],
        "enable_ar_bucket": enable_ar_bucket,
        "min_ar": min_ar,
        "max_ar": max_ar,
        "num_ar_buckets": num_ar_buckets,
        "frame_buckets": [int(f) for f in frame_buckets.split(",")],
        "directory": [{"path": directory_path, "num_repeats": num_repeats}]
    }
    return config

# 实时读取子进程输出
def read_output(pipe, output_queue):
    for line in iter(pipe.readline, ''):
        output_queue.put(line)
    pipe.close()

# 启动训练函数
def start_training(model_type, num_gpus, noise_level):
    if not os.path.exists(TRAIN_SCRIPT):
        yield f"错误：训练脚本 {TRAIN_SCRIPT} 不存在！"
        return
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    train_config_path = get_train_config_path(model_type, noise_level)
    cmd = ["deepspeed", f"--num_gpus={int(num_gpus)}", TRAIN_SCRIPT, "--deepspeed", f"--config={train_config_path}"]

    try:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        output_queue = queue.Queue()
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, output_queue))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, output_queue))
        stdout_thread.start()
        stderr_thread.start()

        full_output = f"训练开始！LoRA 权重将保存到 output_dir/<date_time>/\n"
        max_lines = 100
        output_lines = [full_output]
        yield full_output

        while True:
            try:
                line = output_queue.get(timeout=1)
                output_lines.append(line)
                if len(output_lines) > max_lines:
                    output_lines.pop(0)
                full_output = "".join(output_lines)
                yield full_output
            except queue.Empty:
                if process.poll() is not None:
                    break

        stdout_thread.join()
        stderr_thread.join()
        return_code = process.returncode
        if return_code == 0:
            output_lines.append("\n训练成功完成！")
        else:
            output_lines.append(f"\n训练异常退出，错误代码：{return_code}")
        if len(output_lines) > max_lines:
            output_lines.pop(0)
        full_output = "".join(output_lines)
        yield full_output
    except Exception as e:
        yield f"训练启动失败：{str(e)}"

# 更新配置函数
def update_configs(model_type, noise_level):
    global train_config
    default_train_config = get_default_train_config(model_type, noise_level)
    train_config = load_config(get_train_config_path(model_type, noise_level), default_train_config)
    dataset_config = load_config(DATASET_CONFIG, default_dataset_config)
    
    # 根据模型类型和噪点级别更新路径和时间步长
    if model_type == "i2v (A14B)":
        ckpt_path = "/root/diffusion-pipe/wan2.2/Wan2.2-I2V-A14B"
        transformer_path = f"/root/diffusion-pipe/wan2.2/Wan2.2-I2V-A14B/{noise_level}_model"
        min_t = 0.9 if noise_level == "high_noise" else 0
        max_t = 1.0 if noise_level == "high_noise" else 0.9
    elif model_type == "t2v (A14B)":
        ckpt_path = "/root/diffusion-pipe/wan2.2/Wan2.2-T2V-A14B"
        transformer_path = f"/root/diffusion-pipe/wan2.2/Wan2.2-T2V-A14B/{noise_level}_model"
        min_t = 0.875 if noise_level == "high_noise" else 0
        max_t = 1.0 if noise_level == "high_noise" else 0.875
    else:
        ckpt_path = train_config["model"]["ckpt_path"]
        transformer_path = ""
        min_t = train_config["model"].get("min_t", 0)
        max_t = train_config["model"].get("max_t", 0.875)
    
    dataset_info = {
        "i2v (A14B)": "数据集配置文件的绝对路径（i2v 训练必须仅包含视频）",
        "t2v (A14B)": "数据集配置文件的绝对路径（t2v 训练可包含视频或图像）",
        "t2v (5B)": "数据集配置文件的绝对路径（t2v 训练可包含视频或图像）",
        "t2i (5B)": "数据集配置文件的绝对路径（t2i 训练必须仅包含图像）"
    }[model_type]
    directory_info = {
        "i2v (A14B)": "仅包含视频文件的目录，用于 i2v 训练",
        "t2v (A14B)": "包含视频或图像文件的目录，用于 t2v 训练",
        "t2v (5B)": "包含视频或图像文件的目录，用于 t2v 训练",
        "t2i (5B)": "仅包含图像文件的目录，用于 t2i 训练"
    }[model_type]
    transformer_visible = "A14B" in model_type
    noise_level_visible = "A14B" in model_type
    return (
        train_config["output_dir"],
        train_config["dataset"],
        train_config["epochs"],
        train_config["micro_batch_size_per_gpu"],
        train_config["pipeline_stages"],
        train_config["gradient_accumulation_steps"],
        train_config["gradient_clipping"],
        train_config["warmup_steps"],
        train_config["blocks_to_swap"],
        train_config["eval_every_n_epochs"],
        train_config["eval_before_first_step"],
        train_config["eval_micro_batch_size_per_gpu"],
        train_config["eval_gradient_accumulation_steps"],
        train_config["save_every_n_epochs"],
        train_config["checkpoint_every_n_minutes"],
        train_config["activation_checkpointing"],
        train_config["save_dtype"],
        train_config["caching_batch_size"],
        train_config["video_clip_mode"],
        ckpt_path,
        transformer_path,
        train_config["model"]["dtype"],
        train_config["model"].get("transformer_dtype", "") if transformer_visible else "",
        train_config["model"]["timestep_sample_method"],
        min_t,
        max_t,
        train_config["adapter"]["rank"],
        train_config["adapter"]["dtype"],
        train_config["adapter"].get("init_from_existing", ""),
        train_config["optimizer"]["type"],
        str(train_config["optimizer"]["lr"]),
        train_config["optimizer"]["betas"][0],
        train_config["optimizer"]["betas"][1],
        train_config["optimizer"]["weight_decay"],
        train_config["optimizer"]["eps"],
        train_config.get("partition_method", "parameters"),
        ",".join(map(str, train_config.get("partition_split", []))),
        train_config["model"].get("llm_path", ""),
        toml.dumps({"eval_datasets": train_config.get("eval_datasets", [])}),
        train_config.get("disable_block_swap_for_eval", False),
        train_config["optimizer"].get("gradient_release", False),
        str(dataset_config["resolutions"][0]),
        dataset_config["enable_ar_bucket"],
        dataset_config["min_ar"],
        dataset_config["max_ar"],
        dataset_config["num_ar_buckets"],
        ",".join(map(str, dataset_config["frame_buckets"])),
        dataset_config["directory"][0]["path"],
        dataset_config["directory"][0]["num_repeats"],
        gr.update(info=dataset_info),
        gr.update(info=directory_info),
        gr.update(visible=transformer_visible),
        gr.update(visible=transformer_visible),
        gr.update(visible=noise_level_visible)
    )

# 动态限制函数
def restrict_blocks_to_swap(pipeline_stages):
    if pipeline_stages > 1:
        return gr.update(value=0, interactive=False, info="交换块数（禁用，因为流水线阶段数 > 1；请验证与 Wan2.2 的兼容性）")
    return gr.update(interactive=True, info="将模型块移到内存以减少显存使用，仅在流水线阶段数=1 时有效（请验证与 Wan2.2 的兼容性）")

def restrict_gradient_clipping(gradient_release):
    if gradient_release:
        return gr.update(value=0, interactive=False, info="梯度裁剪（禁用，因为梯度释放已启用；请验证与 Wan2.2 的兼容性）")
    return gr.update(interactive=True, info="限制最大梯度值，当梯度释放启用时无效（请验证与 Wan2.2 的兼容性）")

# 创建 Gradio 界面
with gr.Blocks(title="Wan2.2 LoRA 训练配置器（B站HooTooH）制作", css=".gradio-container {max-width: 1200px; margin: auto; font-family: Arial, sans-serif;} h1 {color: #f28c38;} h2 {color: #f28c38;} h3 {color: #f28c38;} .section-box {border: 1px solid #dfe6e9; border-radius: 8px; padding: 16px; margin-bottom: 16px; background-color: #f8f9fa;} .gr-button {background-color: #3498db; color: white; border-radius: 6px;} .gr-button:hover {background-color: #2980b9;}") as demo:
    gr.Markdown("## Wan2.2 LoRA 训练配置器（B站HooTooH）制作")
    gr.Markdown("QQ交流群543917943 。确保数据集包含：i2v (A14B) 仅视频，t2v (A14B/5B) 视频或图像，t2i (5B) 仅图像。")

    with gr.Group():
        gr.Markdown("### 模型选择")
        with gr.Row():
            model_type = gr.Dropdown(choices=["i2v (A14B)", "t2v (A14B)", "t2v (5B)", "t2i (5B)"], value="i2v (A14B)", label="模型类型", info="选择模型：i2v (A14B)、t2v (A14B)、t2v (5B) 或 t2i (5B)")
            noise_level = gr.Dropdown(choices=["high_noise", "low_noise"], value="high_noise", label="噪点级别", info="选择高噪或低噪模型（仅限 A14B 模型）", visible=True)

    with gr.Group():
        gr.Markdown("### 常规设置")
        with gr.Row():
            num_gpus = gr.Number(value=1, label="GPU 数量", info="指定用于训练的 GPU 数量", minimum=1, step=1)
            output_dir = gr.Textbox(value=train_config["output_dir"], label="输出目录", info="保存训练结果的路径")
            dataset = gr.Textbox(value=train_config["dataset"], label="数据集配置文件路径", info="数据集配置文件的绝对路径")
        with gr.Row():
            epochs = gr.Number(value=train_config["epochs"], label="训练轮数", info="总训练轮数")
            micro_batch_size_per_gpu = gr.Number(value=train_config["micro_batch_size_per_gpu"], label="每 GPU 微批次大小", info="每个 GPU 的前后向传递批次大小")
            pipeline_stages = gr.Number(value=train_config["pipeline_stages"], label="流水线阶段数", info="将模型拆分到的 GPU 数量")
        with gr.Row():
            partition_method = gr.Dropdown(choices=["parameters", "manual"], value=train_config.get("partition_method", "parameters"), label="分区方法", info="自动参数分配或手动层分割")
            partition_split = gr.Textbox(value=",".join(map(str, train_config.get("partition_split", []))), label="分区分割", info="手动模式下逗号分隔的分割点", visible=False)
        with gr.Row():
            gradient_accumulation_steps = gr.Number(value=train_config["gradient_accumulation_steps"], label="梯度累积步数", info="在更新权重前累积的步数")
            gradient_clipping = gr.Number(value=train_config["gradient_clipping"], label="梯度裁剪", info="限制最大梯度值")
            warmup_steps = gr.Number(value=train_config["warmup_steps"], label="预热步数", info="逐渐增加学习率的步数")
        with gr.Row():
            blocks_to_swap = gr.Number(value=train_config["blocks_to_swap"], label="交换块数", info="将模型块移到内存以减少显存使用（流水线阶段数=1）")
            eval_every_n_epochs = gr.Number(value=train_config["eval_every_n_epochs"], label="每 N 轮评估", info="评估频率")
            eval_before_first_step = gr.Checkbox(value=train_config["eval_before_first_step"], label="训练前评估", info="是否在第一步之前运行评估")
        with gr.Row():
            eval_micro_batch_size_per_gpu = gr.Number(value=train_config["eval_micro_batch_size_per_gpu"], label="评估时每 GPU 微批次大小", info="评估时的微批次大小")
            eval_gradient_accumulation_steps = gr.Number(value=train_config["eval_gradient_accumulation_steps"], label="评估时梯度累积步数", info="评估时的梯度累积步数")
            save_every_n_epochs = gr.Number(value=train_config["save_every_n_epochs"], label="每 N 轮保存", info="模型保存频率")
        with gr.Row():
            checkpoint_every_n_minutes = gr.Number(value=train_config["checkpoint_every_n_minutes"], label="检查点保存间隔（分钟）", info="每隔几分钟保存检查点，0 表示禁用")
            activation_checkpointing = gr.Checkbox(value=train_config["activation_checkpointing"], label="激活检查点", info="保存显存的技术，通常启用")
            save_dtype = gr.Dropdown(choices=["bfloat16", "float16", "float32"], value=train_config["save_dtype"], label="保存数据类型", info="保存模型权重的数据类型")
        with gr.Row():
            caching_batch_size = gr.Number(value=train_config["caching_batch_size"], label="缓存批次大小", info="预缓存时的批次大小")
            video_clip_mode = gr.Dropdown(choices=["single_beginning", "single_middle", "multiple_overlapping"], value=train_config["video_clip_mode"], label="视频帧提取模式", info="提取视频帧的方法")
        with gr.Row():
            eval_datasets = gr.Textbox(value=toml.dumps({"eval_datasets": train_config.get("eval_datasets", [])}), label="评估数据集", info="多个评估数据集配置（TOML 格式）", lines=3)
            disable_block_swap_for_eval = gr.Checkbox(value=train_config.get("disable_block_swap_for_eval", False), label="评估时禁用块交换", info="评估时禁用块交换以加速")

    with gr.Group():
        gr.Markdown("### 模型配置")
        with gr.Row():
            ckpt_path = gr.Textbox(value=train_config["model"]["ckpt_path"], label="检查点路径", info="Wan2.2 模型预训练权重路径（必须存在）")
            transformer_path = gr.Textbox(value=train_config["model"].get("transformer_path", ""), label="Transformer 路径", info="Wan2.2 模型 Transformer 权重路径（仅限 A14B 模型）", visible=True)
            llm_path = gr.Textbox(value=train_config["model"].get("llm_path", ""), label="UMT5 路径", info="自定义 UMT5 权重路径")
        with gr.Row():
            dtype = gr.Dropdown(choices=["bfloat16", "float16", "float32"], value=train_config["model"]["dtype"], label="基础数据类型", info="模型计算的数据类型")
            transformer_dtype = gr.Dropdown(choices=["", "float8", "bfloat16"], value=train_config["model"].get("transformer_dtype", ""), label="Transformer 数据类型", info="Transformer 部分的特殊数据类型（仅限 A14B 模型）", visible=True)
            timestep_sample_method = gr.Dropdown(choices=["logit_normal", "uniform"], value=train_config["model"]["timestep_sample_method"], label="时间步采样方法", info="训练期间时间步的采样策略")
        with gr.Row():
            min_t = gr.Number(value=train_config["model"].get("min_t", 0), label="最小时间步", info="训练的最小时间步（i2v 高噪 0.9，低噪 0；t2v 高噪 0.875，低噪 0）")
            max_t = gr.Number(value=train_config["model"].get("max_t", 0.9 if "i2v" in model_type.value else 0.875), label="最大时间步", info="训练的最大时间步（i2v 高噪 1.0，低噪 0.9；t2v 高噪 1.0，低噪 0.875）")

    with gr.Group():
        gr.Markdown("### 适配器配置")
        with gr.Row():
            rank = gr.Number(value=train_config["adapter"]["rank"], label="LoRA 秩", info="LoRA 秩大小，影响模型容量")
            adapter_dtype = gr.Dropdown(choices=["bfloat16", "float16", "float32"], value=train_config["adapter"]["dtype"], label="LoRA 数据类型", info="LoRA 权重的数据类型")
        with gr.Row():
            init_from_existing = gr.Textbox(value=train_config["adapter"].get("init_from_existing", ""), label="从现有 LoRA 初始化", info="现有 LoRA 权重路径")

    with gr.Group():
        gr.Markdown("### 优化器配置")
        with gr.Row():
            optimizer_type = gr.Dropdown(choices=["adamw_optimi", "AdamW8bitKahan", "Prodigy"], value=train_config["optimizer"]["type"], label="优化器类型", info="用于训练的优化器")
            lr = gr.Textbox(value=str(train_config["optimizer"]["lr"]), label="学习率", info="优化器的学习率（例如 2e-5）")
            gradient_release = gr.Checkbox(value=train_config["optimizer"].get("gradient_release", False), label="梯度释放", info="实验性显存节省选项")
        with gr.Row():
            betas_0 = gr.Number(value=train_config["optimizer"]["betas"][0], label="Beta 1", info="Adam 优化器的第一个 beta 参数")
            betas_1 = gr.Number(value=train_config["optimizer"]["betas"][1], label="Beta 2", info="Adam 优化器的第二个 beta 参数")
            weight_decay = gr.Number(value=train_config["optimizer"]["weight_decay"], label="权重衰减", info="正则化参数")
            eps = gr.Number(value=train_config["optimizer"]["eps"], label="Epsilon", info="Adam 的数值稳定性参数")

    with gr.Group():
        gr.Markdown("### 数据集配置")
        with gr.Row():
            resolutions = gr.Textbox(value=str(dataset_config["resolutions"][0]), label="分辨率", info="训练图像的分分辨率（例如 512）")
            enable_ar_bucket = gr.Checkbox(value=dataset_config["enable_ar_bucket"], label="启用宽高比分组", info="是否按宽高比分组图像")
        with gr.Row():
            min_ar = gr.Number(value=dataset_config["min_ar"], label="最小宽高比", info="宽高比范围的最小值")
            max_ar = gr.Number(value=dataset_config["max_ar"], label="最大宽高比", info="宽高比范围的最大值")
            num_ar_buckets = gr.Number(value=dataset_config["num_ar_buckets"], label="宽高比分组数量", info="宽高比分组数量")
        with gr.Row():
            frame_buckets = gr.Textbox(value=",".join(map(str, dataset_config["frame_buckets"])), label="帧分组", info="视频帧分组，逗号分隔（例如，视频用 1,33，图像用 1）")
            directory_path = gr.Textbox(value=dataset_config["directory"][0]["path"], label="数据目录路径", info="包含训练数据的目录")
            num_repeats = gr.Number(value=dataset_config["directory"][0]["num_repeats"], label="重复次数", info="每个样本的重复次数，影响轮次大小")

    with gr.Group():
        gr.Markdown("### 操作和输出")
        with gr.Row():
            train_config_output = gr.Textbox(label="训练配置（TOML）", lines=10, interactive=False)
            dataset_config_output = gr.Textbox(label="数据集配置（TOML）", lines=10, interactive=False)
        save_message = gr.Textbox(label="保存状态")
        train_output = gr.Textbox(label="训练输出", lines=10)

        with gr.Row():
            save_btn = gr.Button("保存配置文件")
            train_btn = gr.Button("开始训练")

    # 事件绑定
    num_gpus.change(fn=lambda x: x, inputs=num_gpus, outputs=pipeline_stages)
    pipeline_stages.change(fn=restrict_blocks_to_swap, inputs=pipeline_stages, outputs=blocks_to_swap)
    partition_method.change(fn=lambda x: gr.update(visible=x == "manual"), inputs=partition_method, outputs=partition_split)
    gradient_release.change(fn=restrict_gradient_clipping, inputs=gradient_release, outputs=gradient_clipping)

    model_type.change(
        fn=update_configs,
        inputs=[model_type, noise_level],
        outputs=[
            output_dir, dataset, epochs, micro_batch_size_per_gpu, pipeline_stages, gradient_accumulation_steps,
            gradient_clipping, warmup_steps, blocks_to_swap, eval_every_n_epochs, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, save_every_n_epochs,
            checkpoint_every_n_minutes, activation_checkpointing, save_dtype, caching_batch_size,
            video_clip_mode, ckpt_path, transformer_path, dtype, transformer_dtype, timestep_sample_method, min_t, max_t,
            rank, adapter_dtype, init_from_existing, optimizer_type, lr, betas_0, betas_1, weight_decay, eps,
            partition_method, partition_split, llm_path, eval_datasets, disable_block_swap_for_eval, gradient_release,
            resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, directory_path, num_repeats,
            dataset, directory_path, transformer_path, transformer_dtype, noise_level
        ]
    )

    noise_level.change(
        fn=update_configs,
        inputs=[model_type, noise_level],
        outputs=[
            output_dir, dataset, epochs, micro_batch_size_per_gpu, pipeline_stages, gradient_accumulation_steps,
            gradient_clipping, warmup_steps, blocks_to_swap, eval_every_n_epochs, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, save_every_n_epochs,
            checkpoint_every_n_minutes, activation_checkpointing, save_dtype, caching_batch_size,
            video_clip_mode, ckpt_path, transformer_path, dtype, transformer_dtype, timestep_sample_method, min_t, max_t,
            rank, adapter_dtype, init_from_existing, optimizer_type, lr, betas_0, betas_1, weight_decay, eps,
            partition_method, partition_split, llm_path, eval_datasets, disable_block_swap_for_eval, gradient_release,
            resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, directory_path, num_repeats,
            dataset, directory_path, transformer_path, transformer_dtype, noise_level
        ]
    )

    save_btn.click(
        fn=lambda *args: (
            train_config := update_train_config(*args[:41], args[49], args[50]),
            dataset_config := update_dataset_config(*args[41:49]),
            save_message := save_configs(train_config, dataset_config, args[49], args[50]),
            toml.dumps(train_config),
            toml.dumps(dataset_config)
        ),
        inputs=[
            output_dir, dataset, epochs, micro_batch_size_per_gpu, pipeline_stages, gradient_accumulation_steps,
            gradient_clipping, warmup_steps, blocks_to_swap, eval_every_n_epochs, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, save_every_n_epochs,
            checkpoint_every_n_minutes, activation_checkpointing, save_dtype, caching_batch_size,
            video_clip_mode, ckpt_path, transformer_path, dtype, transformer_dtype, timestep_sample_method, min_t, max_t,
            rank, adapter_dtype, init_from_existing, optimizer_type, lr, betas_0, betas_1, weight_decay, eps,
            partition_method, partition_split, llm_path, eval_datasets, disable_block_swap_for_eval, gradient_release,
            resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, directory_path, num_repeats,
            model_type, noise_level
        ],
        outputs=[save_message, train_config_output, dataset_config_output]
    )

    train_btn.click(fn=start_training, inputs=[model_type, num_gpus, noise_level], outputs=train_output)

# 启动 Gradio 界面
demo.launch()