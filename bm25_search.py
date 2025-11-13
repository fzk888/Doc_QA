import string
from rank_bm25 import BM25Okapi

# 尝试导入jieba分词，如果失败则使用备用方案
try:
    import jieba
    jieba_initialized = True
except ImportError:
    jieba = None
    jieba_initialized = False
    print("Warning: jieba not found. Using simple tokenization.")

class BM25Search:
    def __init__(self, docs, stopwords_file='stopwords.txt'):
        self.docs = docs
        self.stop_words = self.load_stopwords(stopwords_file)
        self.tokenized_corpus = self.tokenize_corpus()
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def load_stopwords(self, stopwords_file):
        stop_words = set()
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    stop_words.add(line.strip())
        except FileNotFoundError:
            print(f"Warning: Stopwords file {stopwords_file} not found. Using empty stopwords set.")
        return stop_words

    def bm25_tokenizer(self, text):
        tokenized_doc = []
        
        # 使用jieba分词（如果可用）
        if jieba_initialized and jieba is not None:
            tokens = jieba.cut_for_search(text)
        else:
            # 备用方案：使用简单的正则表达式分词
            import re
            tokens = re.findall(r'[\w]+', text)
            
        for token in tokens:
            token = token.strip(string.punctuation)
            if len(token) > 0 and token not in self.stop_words:
                tokenized_doc.append(token)
        return tokenized_doc

    def tokenize_corpus(self):
        # 去除 tqdm 进度条开销，使用列表推导加速初始化
        return [self.bm25_tokenizer(doc.page_content) for doc in self.docs]

    def search(self, query, threshold=0.1):
        # 减少不必要的 I/O 输出，提高查询速度
        bm25_scores = self.bm25.get_scores(self.bm25_tokenizer(query))
        
        bm25_hits = [{'corpus_id': idx, 'score': score} for idx, score in enumerate(bm25_scores) if score >= threshold]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

        bm25_hits_res = [self.docs[hit['corpus_id']] for hit in bm25_hits]

        return bm25_hits_res