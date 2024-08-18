import gc
import json
import locale
import torch
from transformers import AutoTokenizer,DataCollatorWithPadding,AutoModelForCausalLM,TrainingArguments,Trainer
if torch.cuda.is_available():
    from transformers import BitsAndBytesConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.ini")
print(config_path)
import configparser
config = configparser.ConfigParser()
config.read(config_path)
class CausalLM_loader:
    original_IS_CHANGED = None

    def __init__(self):
        self.id = hash(str(self))
        self.device = ""
        self.dtype = ""
        self.model_type = ""
        self.model_path = ""
        self.tokenizer_path = ""
        self.model = ""
        self.tokenizer = ""
        self.is_locked = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {"default": ""}),
                "model_path": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "tokenizer_path": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
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
                "is_locked": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (
        "CUSTOM",
        "CUSTOM",
    )
    RETURN_NAMES = (
        "model",
        "tokenizer",
    )

    FUNCTION = "chatbot"

    # OUTPUT_NODE = False

    CATEGORY = "大模型学校（llm_schools）/加载器（loader）"

    def chatbot(self, model_name, model_path, tokenizer_path, device, dtype, is_locked=False):
        self.is_locked = is_locked
        if CausalLM_loader.original_IS_CHANGED is None:
            # 保存原始的IS_CHANGED方法的引用
            CausalLM_loader.original_IS_CHANGED = CausalLM_loader.IS_CHANGED
        if self.is_locked == False:
            setattr(CausalLM_loader, "IS_CHANGED", CausalLM_loader.original_IS_CHANGED)
        else:
            # 如果方法存在，则删除
            if hasattr(CausalLM_loader, "IS_CHANGED"):
                delattr(CausalLM_loader, "IS_CHANGED")
        if model_path != "" and tokenizer_path != "":
            model_name = ""
        if model_name in config:
            model_path = config[model_name].get("model_path")
            tokenizer_path = config[model_name].get("tokenizer_path")
        elif model_name != "":
            model_path = model_name
            tokenizer_path = model_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        if (
            self.device != device
            or self.dtype != dtype
            or self.model_path != model_path
            or self.tokenizer_path != tokenizer_path
            or is_locked == False
        ):
            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            # 对于 CPU 和 MPS 设备，不需要清空 CUDA 缓存
            elif self.device == "cpu" or self.device == "mps":
                gc.collect()
            self.model = ""
            self.tokenizer = ""
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path
            self.device = device
            self.dtype = dtype

        if self.tokenizer == "":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.model == "":
            if device == "cuda":
                if dtype == "float32":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path, trust_remote_code=True, device_map="cuda"
                    )
                elif dtype == "float16":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path, trust_remote_code=True, device_map="cuda"
                    ).half()
                elif dtype == "bfloat16":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path, trust_remote_code=True, device_map="cuda",torch_dtype=torch.bfloat16
                    )
                elif dtype == "int8":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        device_map="cuda",
                        quantization_config=quantization_config,
                    )
                elif dtype == "int4":
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        device_map="cuda",
                        quantization_config=quantization_config,
                    )
            elif device == "cpu":
                if dtype == "float32":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path, trust_remote_code=True, device_map="cpu"
                    )
                elif dtype in ["float16", "bfloat16", "int8", "int4"]:
                    print(f"{dtype} is not supported on CPU. Using float32 instead.")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path, trust_remote_code=True, device_map="cpu"
                    )
            elif device == "mps":
                if dtype == "float32":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path, trust_remote_code=True, device_map="mps"
                    )
                elif dtype == "float16":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path, trust_remote_code=True, device_map="mps"
                    ).half()
                elif dtype in ["bfloat16", "int8", "int4"]:
                    print(f"{dtype} is not supported on MPS. Using float32 instead.")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path, trust_remote_code=True, device_map="mps"
                    ).half()
            self.model = self.model.eval()
        return (
            self.model,
            self.tokenizer,
        )

class LLM_Arguments:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_dir": ("STRING", {"default": "output"}),
                "evaluation_strategy": ("STRING", {"default": "epoch"}),
                "learning_rate": ("FLOAT", {"default": 1e-5}),
                "per_device_train_batch_size": ("INT", {"default": 1}),
                "per_device_eval_batch_size": ("INT", {"default": 1}),
                "num_train_epochs": ("INT", {"default": 1}),
                "weight_decay": ("FLOAT", {"default": 0.01}),
                "save_steps": ("INT", {"default": 1000}),
                "save_total_limit": ("INT", {"default": 2}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRAINING_ARGS",)
    RETURN_NAMES = ("training_args")

    FUNCTION = "Argument"

    # OUTPUT_NODE = False

    CATEGORY = "大模型学校（llm_schools）/数据预处理（data preprocessing）"

    def Argument(
            self,
            output_dir,
            evaluation_strategy,
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
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=evaluation_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
        )
        return (training_args,)

class LLM_Trainer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("CUSTOM", {}),
                "training_args": ("TRAINING_ARGS", {}),
                "tokenized_datasets": ("TOKENIZED_DATASETS", {}),
                "tokenizer": ("CUSTOM", {}),
                "data_collator": ("DATA_COLLATOR", {}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results")

    FUNCTION = "Trainer"

    # OUTPUT_NODE = False

    CATEGORY = "大模型学校（llm_schools）/数据预处理（data preprocessing）"

    def Trainer(self, model,training_args,tokenized_datasets,tokenizer,data_collator, is_enable=True):
        if is_enable == False:
            return (None,)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        results = trainer.evaluate()
        results_json = json.dumps(results, indent=4)
        return (results_json,)

class LLM_data_collator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tokenizer": ("CUSTOM", {}),
                "datasets": ("DATASETS", {}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TOKENIZED_DATASETS","DATA_COLLATOR")
    RETURN_NAMES = ("tokenized_datasets","data_collator")

    FUNCTION = "Train"

    # OUTPUT_NODE = False

    CATEGORY = "大模型学校（llm_schools）/数据预处理（data preprocessing）"

    def Train(self, tokenizer,datasets, is_enable=True):
        if is_enable == False:
            return (None,)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        tokenized_datasets = datasets.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        return (tokenized_datasets,data_collator,)

NODE_CLASS_MAPPINGS = {
    "LLM_data_collator": LLM_data_collator,
    "CausalLM_loader":CausalLM_loader,
    "LLM_Arguments":LLM_Arguments,
    "LLM_Trainer":LLM_Trainer,
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
        "LLM_data_collator": "数据生成器",
        "CausalLM_loader": "因果语言模型加载器",
        "LLM_Arguments": "LLM训练参数",
        "LLM_Trainer": "LLM训练器",}
else:
    NODE_DISPLAY_NAME_MAPPINGS = {
        "LLM_data_collator": "Data Collator",
        "CausalLM_loader": "Causal Language Model Loader",
        "LLM_Arguments": "LLM Training Arguments",
        "LLM_Trainer": "LLM Trainer",}