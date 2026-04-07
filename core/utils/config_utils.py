import os 
import threading

from ruamel.yaml import YAML
from dotenv import load_dotenv


load_dotenv()

yaml = YAML()
yaml.preserve_quotes = True

class Settings:
    def __init__(self, config_path: str = "config.yaml"):
        self.lock = threading.Lock()
        with open(config_path, 'r', encoding='utf-8') as file:
            self.data = yaml.load(file)
        api_key_302 = os.getenv("API_KEY_302")
        inworld_tts_key = os.getenv("INWORLD_TTS_KEY")
        self.data["whisper"]["whisperX_302_api_key"] = api_key_302
        self.data["api"]["key"] = api_key_302
        self.data["inworld_tts"]["api_key"] = inworld_tts_key

        


    def load_key(self, key):
        with self.lock:
            keys = key.split('.')
            value = self.data
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    raise KeyError(f"Key '{k}' not found in configuration")
            return value

    def update_key(self, key, new_value):
        with self.lock:
            keys = key.split('.')
            current = self.data
            for k in keys[:-1]:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return False

            if isinstance(current, dict) and keys[-1] in current:
                current[keys[-1]] = new_value
                return True
            else:
                raise KeyError(f"Key '{keys[-1]}' not found in configuration")
            
settings = Settings()

def load_key(key):
    return settings.load_key(key)


def update_key(key, new_value):
    return settings.update_key(key, new_value)


def get_joiner(language):
    if language in load_key("language_split_with_space"):
        return " "
    if language in load_key("language_split_without_space"):
        return ""
    raise ValueError(f"Unsupported language code: {language}")


if __name__ == "__main__":
    print(load_key("language_split_with_space"))
