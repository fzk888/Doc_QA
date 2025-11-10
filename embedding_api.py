import yaml
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class EmbeddingAPI:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if EmbeddingAPI._instance is not None:
            raise Exception("这是一个单例类。请使用get_instance()方法获取实例。")
        
        model_kwargs = {"device": config['settings']['device']}
        encode_kwargs = {
            "batch_size": config['settings']['batch_size'],
            "normalize_embeddings": config['settings']['normalize_embeddings']
        }
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=config['paths']['model_dir'],
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def embed_documents(self, documents):
        return self.embeddings.embed_documents(documents)

    def embed_query(self, query):
        return self.embeddings.embed_query(query)