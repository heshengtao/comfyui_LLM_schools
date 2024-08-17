import json
import locale
from datetime import datetime

import pytz
def get_weekday(timezone):
    # 返回当前周几
    timezone = pytz.timezone(timezone)
    now = datetime.now(timezone)
    # 字符串格式输出
    weekday = now.strftime("%A")
    return weekday

class weekday:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "timezone": ("STRING", {"default": "Asia/Shanghai"}),
                "is_enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tool",)

    FUNCTION = "time"

    # OUTPUT_NODE = False

    CATEGORY = "大模型学校（llm_schools）/函数（function）"

    def time(self, timezone, is_enable=True):
        if is_enable == False:
            return (None,)
        output = get_weekday(timezone)
        return (output,)


NODE_CLASS_MAPPINGS = {"weekday": weekday}
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
    NODE_DISPLAY_NAME_MAPPINGS = {"weekday": "星期查询函数"}
else:
    NODE_DISPLAY_NAME_MAPPINGS = {"weekday": "Weekday Query Function"}
