import yaml
import os

def get_base_config(name, default=None):
    try:
        # 尝试从根目录读取 config.yaml
        config_path = os.path.join(os.getcwd(), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get(name, default)
    except Exception:
        pass
    return default

def decrypt_database_config(name):
    # 此项目中直接返回配置即可
    return get_base_config(name, {})
