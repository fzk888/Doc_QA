import logging
import os

class LoggerFactory:
    LEVEL = logging.INFO
    
    @staticmethod
    def set_directory(path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
    
def getLogger(name):
    # 返回一个带名前缀的日志记录器
    return logging.getLogger(f"engines.{name}")
