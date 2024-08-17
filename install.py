import os
import shutil


current_file_path = os.path.abspath(__file__)

# 获取当前文件所在目录的路径
current_dir_path = os.path.dirname(current_file_path)

# 构建config.ini的绝对路径
config_path = os.path.join(current_dir_path, "config.ini")


def copy_config():
    config_path = os.path.join(current_dir_path, "config.ini")
    config_example_path = os.path.join(current_dir_path, "config.ini.example")
    # 判断config.ini是否存在
    if not os.path.exists(config_path):
        # 如果不存在用config_example_path的副本创建config.ini
        shutil.copyfile(config_example_path, config_path)


copy_config()