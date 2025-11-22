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
    """
    BM25搜索类，用于基于关键词的文档检索
    
    BM25是一种基于概率检索模型的排名函数，常用于信息检索。
    它基于词频、逆文档频率和文档长度等因素来评估查询与文档的相关性。
    """
    
    def __init__(self, docs, stopwords_file='stopwords.txt'):
        """
        初始化BM25搜索器
        
        Args:
            docs (list): 文档列表，每个文档应有page_content属性
            stopwords_file (str): 停用词文件路径
        """
        self.docs = docs
        self.stop_words = self.load_stopwords(stopwords_file)
        self.tokenized_corpus = self.tokenize_corpus()
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def load_stopwords(self, stopwords_file):
        """
        加载停用词列表
        
        Args:
            stopwords_file (str): 停用词文件路径
            
        Returns:
            set: 停用词集合
        """
        stop_words = set()
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    stop_words.add(line.strip())
        except FileNotFoundError:
            print(f"Warning: Stopwords file {stopwords_file} not found. Using empty stopwords set.")
        return stop_words

    def bm25_tokenizer(self, text):
        """
        对文本进行BM25分词处理
        
        Args:
            text (str): 待分词的文本
            
        Returns:
            list: 分词结果列表
        """
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
        """
        对整个语料库进行分词处理
        
        Returns:
            list: 分词后的语料库，每个元素是一个文档的分词结果
        """
        # 去除 tqdm 进度条开销，使用列表推导加速初始化
        return [self.bm25_tokenizer(doc.page_content) for doc in self.docs]

    def search(self, query, threshold=0.1):
        """
        执行BM25搜索
        
        Args:
            query (str): 查询语句
            threshold (float): 相关性得分阈值，低于此值的结果将被过滤
            
        Returns:
            list: 搜索结果文档列表，按相关性得分降序排列
        """
        # 减少不必要的 I/O 输出，提高查询速度
        bm25_scores = self.bm25.get_scores(self.bm25_tokenizer(query))
        
        bm25_hits = [{'corpus_id': idx, 'score': score} for idx, score in enumerate(bm25_scores) if score >= threshold]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

        bm25_hits_res = [self.docs[hit['corpus_id']] for hit in bm25_hits]

        return bm25_hits_res