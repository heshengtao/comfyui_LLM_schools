import json
import locale
from huggingface_hub import list_datasets,snapshot_download
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.ini")
print(config_path)
import configparser
config = configparser.ConfigParser()
config.read(config_path)
HF_token = config.get("TOKEN", "HF_token")

from datasets import load_dataset

def load_datasets_from_cache(local_dir):
    data_files = {'train': [], 'validation': [], 'test': []}
    for root, _, files in os.walk(local_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(('.csv', '.json', '.parquet', '.txt', '.xlsx', '.tsv', '.xml')):
                if 'train' in file:
                    data_files['train'].append(file_path)
                elif 'validation' in file or 'dev' in file:
                    data_files['validation'].append(file_path)
                elif 'test' in file:
                    data_files['test'].append(file_path)

    datasets = {}
    for split, files in data_files.items():
        if files:
            if files[0].endswith('.csv'):
                datasets[split] = load_dataset('csv', data_files=files)
            elif files[0].endswith('.json'):
                datasets[split] = load_dataset('json', data_files=files)
            elif files[0].endswith('.parquet'):
                datasets[split] = load_dataset('parquet', data_files=files)
            elif files[0].endswith('.txt'):
                datasets[split] = load_dataset('text', data_files=files)
            elif files[0].endswith('.xlsx'):
                datasets[split] = load_dataset('excel', data_files=files)
            elif files[0].endswith('.tsv'):
                datasets[split] = load_dataset('csv', data_files=files, delimiter='\t')
            elif files[0].endswith('.xml'):
                datasets[split] = load_dataset('xml', data_files=files)
    return datasets
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
    RETURN_NAMES = ("repo_id_list",)

    FUNCTION = "get_dataset"

    # OUTPUT_NODE = False

    CATEGORY = "大模型学校（llm_schools）/数据预处理（data preprocessing）"

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
                "repo_id": ("STRING", {"default": "RussianNLP/wikiomnia"}),
                "cache_dir": ("STRING", {"default": "datasets"}),
                "token": ("STRING", {"default": "hf_XXX"}),
                "force_download": ("BOOLEAN", {"default": False}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("local_dir",)

    FUNCTION = "download"

    OUTPUT_NODE = True

    CATEGORY = "大模型学校（llm_schools）/数据预处理（data preprocessing）"

    def download(self, repo_id,cache_dir,token, is_enable=True,force_download=False):
        if is_enable == False:
            return (None,)
        if token == "":
            token = HF_token
        path = os.path.dirname(os.path.abspath(__file__))
        datasets_path = os.path.join(path, cache_dir)

        # 下载数据集并保存到本地缓存文件夹
        print("Start download...")
        local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=datasets_path, local_dir_use_symlinks=False, force_download=force_download, token=token)
        print("Download complete")
        
        return (local_dir,)

class split_dataset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "local_dir": ("STRING", {"default": ""}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("DATASETS",)
    RETURN_NAMES = ("datasets",)

    FUNCTION = "split"

    # OUTPUT_NODE = True

    CATEGORY = "大模型学校（llm_schools）/数据预处理（data preprocessing）"

    def split(self, local_dir, is_enable=True):
        if is_enable == False:
            return (None,)
        datasets = load_datasets_from_cache(local_dir)
        
        return (datasets,)


NODE_CLASS_MAPPINGS = {
    "get_dataset_name": get_dataset_name,
    "download_dataset":download_dataset,
    "split_dataset":split_dataset,
    }
# 获取系统语言
lang = locale.getdefaultlocale()[0]


try:
    language = config.get("GENERAL", "language")
except:
    language = ""
if language == "zh_CN" or language=="en_US":
    lang=language
if lang == "zh_CN":
    NODE_DISPLAY_NAME_MAPPINGS = {
        "get_dataset_name": "获取HF数据集repo_id",
        "download_dataset": "下载/加载HF数据集",
        "split_dataset": "分割HF数据集",
        }
else:
    NODE_DISPLAY_NAME_MAPPINGS = {
        "get_dataset_name": "get HF dataset repo_id",
        "download_dataset": "download/load the HF dataset",
        "split_dataset": "split HF dataset",
        }

if __name__ == "__main__":
    # 加载缓存目录中的数据集
    local_dir = 'E:\GitHub\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui_LLM_schools\datasets\datasets--abhi227070--imdb-dataset\snapshots\\331d3ed0738a51bc52d65db7e4d3fc6331fe4d0f'
    datasets = load_datasets_from_cache(local_dir)
    for split, dataset in datasets.items():
        print(f"Loaded {split} dataset:")
        print(dataset)