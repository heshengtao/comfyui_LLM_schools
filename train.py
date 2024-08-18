import locale
from transformers import AutoTokenizer

class tokenize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tokenizer_name_or_path": ("STRING", {"default": "bert-base-uncased"}),
                "datasets": ("DATASETS", {}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tokenized_datasets",)

    FUNCTION = "token"

    # OUTPUT_NODE = False

    CATEGORY = "大模型学校（llm_schools）/数据预处理（data preprocessing）"

    def token(self, tokenizer_name_or_path,datasets, is_enable=True):
        if is_enable == False:
            return (None,)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        tokenized_datasets = datasets.map(tokenize_function, batched=True)
        return (tokenized_datasets,)


NODE_CLASS_MAPPINGS = {"tokenize": tokenize}
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
    NODE_DISPLAY_NAME_MAPPINGS = {"tokenize": "分词函数"}
else:
    NODE_DISPLAY_NAME_MAPPINGS = {"tokenize": "Tokenize Function"}