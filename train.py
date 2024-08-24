import datetime
import gc
import hashlib
import json
import locale
import subprocess
from adapters import AdapterConfig, AdapterTrainer
import torch
from transformers import AutoTokenizer,DataCollatorForLanguageModeling,AutoModelForCausalLM,TrainingArguments,Trainer
if torch.cuda.is_available():
    from transformers import BitsAndBytesConfig
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

    def call_train_core(self, split_datapaths, model_name_or_path, device, dtype, args):
        command = [
            'python', core_path,
            '--split_datapaths', split_datapaths,
            '--name_or_path', model_name_or_path,
            '--device', device,
            '--dtype', dtype,
            '--train_args', args
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

NODE_CLASS_MAPPINGS = {
    "CausalLM_trainer": CausalLM_trainer,
    "LLM_Arguments":LLM_Arguments,
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
        "LLM_Arguments": "LLM 参数",
        }
else:
    NODE_DISPLAY_NAME_MAPPINGS = {
        "CausalLM_trainer": "Causal Language Model Trainer",
        "LLM_Arguments": "LLM Arguments",
        }