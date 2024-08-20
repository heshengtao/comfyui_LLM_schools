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

from datasets import load_dataset,DatasetDict,load_from_disk,concatenate_datasets

def split_datasets_from_hf_cache(local_dir, train_ratio, val_ratio, test_ratio):
    # 获取所有文件路径，包括子文件夹中的文件
    all_files = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # 根据文件扩展名加载数据集
    datasets = []
    for file in all_files:
        if file.endswith('.csv'):
            datasets.append(load_dataset('csv', data_files=file)['train'])
        elif file.endswith('.json'):
            datasets.append(load_dataset('json', data_files=file)['train'])
        elif file.endswith('.parquet'):
            datasets.append(load_dataset('parquet', data_files=file)['train'])
        elif file.endswith('.txt'):
            datasets.append(load_dataset('text', data_files=file)['train'])
        elif file.endswith('.xlsx'):
            datasets.append(load_dataset('excel', data_files=file)['train'])
        elif file.endswith('.tsv'):
            datasets.append(load_dataset('csv', data_files=file, delimiter='\t')['train'])
        elif file.endswith('.xml'):
            datasets.append(load_dataset('xml', data_files=file)['train'])
        elif file.endswith('.arrow'):
            datasets.append(load_from_disk(file))
        else:
            pass
    
    # 检查是否成功加载了任何数据集
    if not datasets:
        raise ValueError("No valid datasets found in the specified directory.")
    
    # 合并所有数据集
    combined_dataset = concatenate_datasets(datasets)
    
    # 过滤掉没有答案的样本
    def has_answer(example):
        return len(example['answers']['text']) > 0

    combined_dataset = combined_dataset.filter(has_answer)
    
    # 拆分数据集
    train_test_split = combined_dataset.train_test_split(test_size=1 - train_ratio)
    val_test_split = train_test_split['test'].train_test_split(test_size=test_ratio / (val_ratio + test_ratio))
    
    # 创建一个DatasetDict来保存拆分后的数据集
    split_dataset = DatasetDict({
        'train': train_test_split['train'],
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    })
    
    return split_dataset
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
                "repo_id": ("STRING", {"default": "rajpurkar/squad_v2"}),
                "cache_dir": ("STRING", {"default": "datasets"}),
                "token": ("STRING", {"default": "hf_XXX"}),
                "force_download": ("BOOLEAN", {"default": True}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("local_dir",)

    FUNCTION = "download"

    OUTPUT_NODE = True

    CATEGORY = "大模型学校（llm_schools）/数据预处理（data preprocessing）"

    def download(self, repo_id,cache_dir,token, is_enable=True,force_download=True):
        if is_enable == False:
            return (None,)
        if token == "":
            token = HF_token
        path = os.path.dirname(os.path.abspath(__file__))
        datasets_path = os.path.join(path, cache_dir)
        if not os.path.exists(datasets_path):
            os.makedirs(datasets_path)
        # 下载数据集并保存到本地缓存文件夹
        print("Start download...")
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=datasets_path,
            local_dir_use_symlinks=False,
            force_download=force_download,
            token=token)
        print("Download complete")
        
        return (local_dir,)

class split_dataset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "local_dir": ("STRING", {"default": ""}),
                "train_ratio": ("FLOAT", {"default": 0.8,"min":0.0,"max":1.0,"step":0.1}),
                "val_ratio": ("FLOAT", {"default": 0.1,"min":0.0,"max":1.0,"step":0.1}),
                "test_ratio": ("FLOAT", {"default": 0.1,"min":0.0,"max":1.0,"step":0.1}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("DATASETS","STRING",)
    RETURN_NAMES = ("split_datasets","log",)

    FUNCTION = "split"

    # OUTPUT_NODE = True

    CATEGORY = "大模型学校（llm_schools）/数据预处理（data preprocessing）"

    def split(self, local_dir, train_ratio, val_ratio, test_ratio, is_enable=True):
        log = ""
        if not is_enable:
            return (None,)
        if train_ratio + val_ratio + test_ratio != 1.0:
            # 按比例缩放
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
            log += "The sum of the scale of the dataset is not 1, the scale is scaled.\n"
        datasets = split_datasets_from_hf_cache(local_dir, train_ratio, val_ratio, test_ratio)
        
        # 查看数据集开头部分数据并存储到log变量中
        log += "Train dataset head:\n"
        log += str(datasets['train'].to_pandas().head()) + "\n\n"
        
        log += "Validation dataset head:\n"
        log += str(datasets['validation'].to_pandas().head()) + "\n\n"
        
        log += "Test dataset head:\n"
        log += str(datasets['test'].to_pandas().head()) + "\n\n"
        
        return (datasets, log)


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
    local_dir="D:\AI\AIhuitu\Blender_ComfyUI\Blender_ComfyUI_Mini\ComfyUI\custom_nodes\comfyui_LLM_schools\datasets\datasets--rajpurkar--squad_v2\snapshots\\3ffb306f725f7d2ce8394bc1873b24868140c412"
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    split_datasets = split_datasets_from_hf_cache(local_dir, train_ratio, val_ratio, test_ratio)
    print(split_datasets['train'][0]) 