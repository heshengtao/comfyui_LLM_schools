import json
import locale
from huggingface_hub import list_datasets,snapshot_download
import os
#获取当前脚本目录下的datasets文件夹路径

class get_dataset_name:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keyword": ("STRING", {"default": "imdb"}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tool",)

    FUNCTION = "get_dataset"

    # OUTPUT_NODE = False

    CATEGORY = "大模型学校（llm_schools）/加载器（loader）"

    def get_dataset(self, keyword, is_enable=True):
        if is_enable == False:
            return (None,)

        datasets_list = list(list_datasets(filter=keyword))

        # 提取数据集名称
        dataset_names = [dataset.id for dataset in datasets_list]

        # 转换为 JSON 字符串
        json_string = json.dumps(dataset_names, ensure_ascii=False,indent=4)    

        return (json_string,)

class download_dataset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "imdb"}),
                "cache_dir": ("STRING", {"default": "datasets"}),
                "token": ("STRING", {"default": "hf_XXX"}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "download"

    OUTPUT_NODE = True

    CATEGORY = "大模型学校（llm_schools）/加载器（loader）"

    def download(self, repo_id,cache_dir,token, is_enable=True):
        if is_enable == False:
            return (None,)

        path = os.path.dirname(os.path.abspath(__file__))
        datasets_path = os.path.join(path, cache_dir)

        # 下载数据集并保存到本地缓存文件夹
        snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=datasets_path, local_dir_use_symlinks=False, resume_download=True, token=token)
        return ()


NODE_CLASS_MAPPINGS = {
    "get_dataset_name": get_dataset_name,
    "download_dataset":download_dataset,
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
        "get_dataset_name": "获取数据集名字",
        "download_dataset": "下载数据集",
        }
else:
    NODE_DISPLAY_NAME_MAPPINGS = {
        "get_dataset_name": "get dataset name",
        "download_dataset": "download dataset",
        }

if __name__ == "__main__":
    print(get_dataset_name().get_dataset("imdb", True))