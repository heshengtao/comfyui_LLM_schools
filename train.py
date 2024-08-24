import datetime
import hashlib
import json
import locale
import subprocess
import torch
import os
import torch.nn.functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.ini")
print(config_path)
import configparser
import platform
config = configparser.ConfigParser()
config.read(config_path)
core_path = os.path.join(current_dir, 'train_core.py')

class CausalLM_trainer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "split_datapaths": ("STRING", {"default": ""}),
                "model_name_or_path": ("STRING", {"default": "gpt2"}),
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {
                        "default": "auto",
                    },
                ),
                "dtype": (
                    ["float32", "float16","bfloat16", "int8", "int4"],
                    {
                        "default": "float32",
                    },
                ),
                "args": ("ARGS", {"default": {}}),
                "fine_tuning_method": (["full_fine_tuning","lora","adapter","p-tuning","prompt_tuning","prefix-tuning","IA3"], {"default": "full_fine_tuning"}),
            },
            "optional": {
                "peft_args": ("ARGS", {"default": {}}),
            }
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "log",
    )

    FUNCTION = "call_train_core"

    # OUTPUT_NODE = False

    CATEGORY = "大模型学校（llm_schools）/模型训练（Model Training）"

    def call_train_core(self, split_datapaths, model_name_or_path, device, dtype, args,fine_tuning_method,peft_args="\{\}"):
        if peft_args == "\{\}" or peft_args is None:
            command = [
            'python', core_path,
            '--split_datapaths', split_datapaths,
            '--name_or_path', model_name_or_path,
            '--device', device,
            '--dtype', dtype,
            '--train_args', args,
            '--fine_tuning_method', fine_tuning_method,
        ]
        else:
            command = [
                'python', core_path,
                '--split_datapaths', split_datapaths,
                '--name_or_path', model_name_or_path,
                '--device', device,
                '--dtype', dtype,
                '--train_args', args,
                '--fine_tuning_method', fine_tuning_method,
                '--peft_args', peft_args
            ]
        print(command)
        
        system = platform.system()
        
        if system == 'Windows':
            # 在 Windows 上打开新终端
            process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif system == 'Darwin':
            # 在 macOS 上打开新终端
            process = subprocess.Popen(['open', '-a', 'Terminal.app'] + command)
        elif system == 'Linux':
            # 在 Linux 上打印进程号并提醒用户自行查看
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"子进程 PID: {process.pid}")
            print(f"请在终端中使用 `ps -p {process.pid}` 或 `top` 命令查看进程输出。")
        else:
            raise Exception(f"不支持的操作系统: {system}")

        return (f"训练任务已启动，请查看终端输出。模型检查点将被保存到 comfyui根目录下的{args['output_dir']}，或者是你填入的文件夹绝对路径下。The training task has started, please check the end point output. Model checkpoints will be saved to {args ['output_dir']} in the root directory of comfyui, or in the absolute path of the folder you filled in.",)
    @classmethod
    def IS_CHANGED(s):
        # 生成当前时间的哈希值
        hash_value = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()
        return hash_value

class LLM_Arguments:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_dir": ("STRING", {"default": "results"}),
                "eval_strategy": ("STRING", {"default": "epoch"}),
                "learning_rate": ("FLOAT", {"default": 2e-5, "min": 0.0, "max": 1.0, "step": 0.000001}),
                "per_device_train_batch_size": ("INT", {"default": 1}),
                "per_device_eval_batch_size": ("INT", {"default": 1}),
                "num_train_epochs": ("INT", {"default": 3}),
                "weight_decay": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "save_steps": ("INT", {"default": 1000}),
                "save_total_limit": ("INT", {"default": 2}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("args",)

    FUNCTION = "Argument"

    # OUTPUT_NODE = False

    CATEGORY = "大模型学校（llm_schools）/模型训练（Model Training）"

    def Argument(
            self,
            output_dir,
            eval_strategy,
            learning_rate,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            num_train_epochs,
            weight_decay,
            save_steps,
            save_total_limit,
            is_enable=True):
        if is_enable == False:
            return (None,)
        training_args = {
    "output_dir": output_dir,
    "eval_strategy": eval_strategy,
    "learning_rate": learning_rate,
    "per_device_train_batch_size": per_device_train_batch_size,
    "per_device_eval_batch_size": per_device_eval_batch_size,
    "num_train_epochs": num_train_epochs,
    "weight_decay": weight_decay,
    "save_steps": save_steps,
    "save_total_limit": save_total_limit,
}
        args=json.dumps(training_args, indent=4)
        return (args,)

import json

class Lora_or_adapter_Arguments:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "r": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "lora_alpha": ("FLOAT", {"default": 32.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "target_modules": (
                    [
                        "q_proj, v_proj", "query, value", "c_attn", "query_key_value",
                        "q, v", "in_proj", "query_proj, value_proj", "Wqkv", "qkv_proj"
                    ],
                    {
                        "default": "q_proj, v_proj",
                    },
                ),
                "lora_dropout": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bias": (
                    ["none", "all", "lora_only"],
                    {
                        "default": "none",
                    },
                ),
                "task_type": (
                    [
                        "SEQ_2_SEQ_LM", "CAUSAL_LM", "SEQ_CLASSIFICATION", "TOKEN_CLASSIFICATION", "QUESTION_ANSWERING"
                    ],
                    {
                        "default": "SEQ_2_SEQ_LM",
                    },
                ),
                "init_lora_weights": (
                    [
                        "kaiming_uniform", "gaussian", "pissa", "pissa_niter_[number of iters]", "olora", "loftq"
                    ],
                    {
                        "default": "kaiming_uniform",
                    },
                ),
                "fan_in_fan_out": ("BOOLEAN", {"default": False}),
                "modules_to_save": (
                    [
                        "embed", "norm", "output", "classifier", "pooler"
                    ],
                    {
                        "default": "",
                    },
                ),
                "layers_to_transform": (
                    [
                        "encoder", "decoder", "attention", "feed_forward"
                    ],
                    {
                        "default": "",
                    },
                ),
                "layers_pattern": (
                    [
                        "layers", "h", "blocks", "transformer"
                    ],
                    {
                        "default": "",
                    },
                ),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("peft_args",)

    FUNCTION = "Argument"

    CATEGORY = "大模型学校（llm_schools）/模型训练（Model Training）"

    def Argument(
            self,
            r,
            lora_alpha,
            target_modules,
            lora_dropout,
            bias,
            task_type,
            init_lora_weights,
            fan_in_fan_out,
            modules_to_save,
            layers_to_transform,
            layers_pattern,
            is_enable=True):
        if not is_enable:
            return (None,)
        
        lora_args = {
            "r": r,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules.split(','),  # 将字符串转换为列表
            "lora_dropout": lora_dropout,
            "bias": bias,
            "task_type": task_type,
            "init_lora_weights": init_lora_weights,
            "fan_in_fan_out": fan_in_fan_out,
            "modules_to_save": modules_to_save.split(',') if modules_to_save else None,
            "layers_to_transform": layers_to_transform.split(',') if layers_to_transform else None,
            "layers_pattern": layers_pattern if layers_pattern else None,
        }
        
        peft_args = json.dumps(lora_args, indent=4)
        return (peft_args,)


class Prefix_Arguments:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prefix_length": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "prefix_dropout": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "task_type": (["QUESTION_ANSWERING","SEQ_2_SEQ_LM", "CAUSAL_LM","SEQ_CLASSIFICATION","TOKEN_CLASSIFICATION"], {"default": "QUESTION_ANSWERING"}),
                "num_virtual_tokens": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "token_dim": ("INT", {"default": 768, "min": 1, "max": 4096, "step": 1}),
                "num_transformer_submodules": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "num_attention_heads": ("INT", {"default": 12, "min": 1, "max": 16, "step": 1}),
                "num_layers": ("INT", {"default": 12, "min": 1, "max": 24, "step": 1}),
                "flat": ("BOOLEAN", {"default": False}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("peft_args",)

    FUNCTION = "Argument"

    CATEGORY = "大模型学校（llm_schools）/模型训练（Model Training）"

    def Argument(
            self,
            prefix_length,
            prefix_dropout,
            task_type,
            num_virtual_tokens,
            token_dim,
            num_transformer_submodules,
            num_attention_heads,
            num_layers,
            flat,
            is_enable=True):
        if not is_enable:
            return (None,)
        
        prefix_args = {
            "prefix_length": prefix_length,
            "prefix_dropout": prefix_dropout,
            "task_type": task_type,
            "num_virtual_tokens": num_virtual_tokens,
            "token_dim": token_dim,
            "num_transformer_submodules": num_transformer_submodules,
            "num_attention_heads": num_attention_heads,
            "num_layers": num_layers,
            "flat": flat,
        }
        
        peft_args = json.dumps(prefix_args, indent=4)
        return (peft_args,)

class P_or_Prompt_Arguments:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_virtual_tokens": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "token_dim": ("INT", {"default": 768, "min": 1, "max": 4096, "step": 1}),
                "num_transformer_submodules": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "num_attention_heads": ("INT", {"default": 12, "min": 1, "max": 16, "step": 1}),
                "num_layers": ("INT", {"default": 12, "min": 1, "max": 24, "step": 1}),
                "prompt_tuning_init": (
                    ["RANDOM", "TEXT"],
                    {"default": "RANDOM"}
                ),
                "prompt_tuning_init_text": ("STRING", {"default": ""}),
                "tokenizer_name_or_path": ("STRING", {"default": "bert-base-uncased"}),
                "task_type": (["QUESTION_ANSWERING","SEQ_2_SEQ_LM", "CAUSAL_LM","SEQ_CLASSIFICATION","TOKEN_CLASSIFICATION"], {"default": "QUESTION_ANSWERING"}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("peft_args",)

    FUNCTION = "Argument"

    CATEGORY = "大模型学校（llm_schools）/模型训练（Model Training）"

    def Argument(
            self,
            num_virtual_tokens,
            token_dim,
            num_transformer_submodules,
            num_attention_heads,
            num_layers,
            prompt_tuning_init,
            prompt_tuning_init_text,
            tokenizer_name_or_path,
            task_type,
            is_enable=True):
        if not is_enable:
            return (None,)
        
        prompt_args = {
            "num_virtual_tokens": num_virtual_tokens,
            "token_dim": token_dim,
            "num_transformer_submodules": num_transformer_submodules,
            "num_attention_heads": num_attention_heads,
            "num_layers": num_layers,
            "prompt_tuning_init": prompt_tuning_init,
            "prompt_tuning_init_text": prompt_tuning_init_text,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "task_type": task_type,
        }
        
        peft_args = json.dumps(prompt_args, indent=4)
        return (peft_args,)

import json

class IA3_Arguments:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task_type": (["QUESTION_ANSWERING","SEQ_2_SEQ_LM", "CAUSAL_LM","SEQ_CLASSIFICATION","TOKEN_CLASSIFICATION"], {"default": "QUESTION_ANSWERING"}),
                "num_virtual_tokens": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "token_dim": ("INT", {"default": 768, "min": 1, "max": 4096, "step": 1}),
                "num_transformer_submodules": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
                "num_attention_heads": ("INT", {"default": 12, "min": 1, "max": 16, "step": 1}),
                "num_layers": ("INT", {"default": 12, "min": 1, "max": 24, "step": 1}),
                "ia3_alpha": ("FLOAT", {"default": 32.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "ia3_dropout": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ARGS",)
    RETURN_NAMES = ("peft_args",)

    FUNCTION = "Argument"

    CATEGORY = "大模型学校（llm_schools）/模型训练（Model Training）"

    def Argument(
            self,
            task_type,
            num_virtual_tokens,
            token_dim,
            num_transformer_submodules,
            num_attention_heads,
            num_layers,
            ia3_alpha,
            ia3_dropout,
            is_enable=True):
        if not is_enable:
            return (None,)
        
        ia3_args = {
            "task_type": task_type,
            "num_virtual_tokens": num_virtual_tokens,
            "token_dim": token_dim,
            "num_transformer_submodules": num_transformer_submodules,
            "num_attention_heads": num_attention_heads,
            "num_layers": num_layers,
            "ia3_alpha": ia3_alpha,
            "ia3_dropout": ia3_dropout,
        }
        
        peft_args = json.dumps(ia3_args, indent=4)
        return (peft_args,)


NODE_CLASS_MAPPINGS = {
    "CausalLM_trainer": CausalLM_trainer,
    "LLM_Arguments":LLM_Arguments,
    "Lora_or_adapter_Arguments":Lora_or_adapter_Arguments,
    "Prefix_Arguments":Prefix_Arguments,
    "P_or_Prompt_Arguments":P_or_Prompt_Arguments,
    "IA3_Arguments":IA3_Arguments,
    }

# 获取系统语言
lang = locale.getdefaultlocale()[0]
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.ini")
import configparser
config = configparser.ConfigParser()
config.read(config_path)
try:
    language = config.get("GENERAL", "language")
except:
    language = ""
if language == "zh_CN" or language=="en_US":
    lang=language
if lang == "zh_CN":
    NODE_DISPLAY_NAME_MAPPINGS = {
        "CausalLM_trainer": "因果语言模型训练器",
        "LLM_Arguments": "LLM参数",
        "Lora_or_adapter_Arguments":"LoRA或Adapter参数",
        "Prefix_Arguments":"Prefix参数",
        "P_or_Prompt_Arguments":"P或Prompt参数",
        "IA3_Arguments":"IA3参数",
        }
else:
    NODE_DISPLAY_NAME_MAPPINGS = {
        "CausalLM_trainer": "Causal Language Model Trainer",
        "LLM_Arguments": "LLM Arguments",
        "Lora_or_adapter_Arguments":"LoRA or Adapter Arguments",
        "Prefix_Arguments":"Prefix Arguments",
        "P_or_Prompt_Arguments":"P or Prompt Arguments",
        "IA3_Arguments":"IA3 Arguments",
        }