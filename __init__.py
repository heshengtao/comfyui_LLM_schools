import glob
import importlib
import os
import sys

NODE_CLASS_MAPPINGS= {}
NODE_DISPLAY_NAME_MAPPINGS = {}
def load_custom_tools():
    # 获取当前脚本目录的路径
    current_dir = os.path.dirname(__file__)

    # 找到当前脚本目录下所有的 .py 文件，排除 __init__.py
    files = [f for f in glob.glob(os.path.join(current_dir, "*.py")) if not f.endswith("__init__.py")]

    for file in files:
        # 获取文件名（不包含扩展名）
        name = os.path.splitext(os.path.basename(file))[0]

        try:
            # 创建一个导入规范
            spec = importlib.util.spec_from_file_location(name, file)

            # 根据导入规范创建一个新的模块对象
            module = importlib.util.module_from_spec(spec)

            # 在 sys.modules 中注册这个模块
            sys.modules[name] = module

            # 执行模块的代码，实际加载模块
            spec.loader.exec_module(module)

            # 如果模块有 NODE_CLASS_MAPPINGS 属性，更新字典
            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))

            # 如果模块有 NODE_DISPLAY_NAME_MAPPINGS 属性，更新字典
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

        except Exception as e:
            # 处理导入错误（例如，跳过文件）
            print(f"导入 {name} 时出错：{e}")


# 调用函数来加载 custom_tool 文件夹下的模块
load_custom_tools()