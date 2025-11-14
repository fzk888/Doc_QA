import string
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import jieba

class BM25Search:
    def __init__(self, docs, stopwords_file='stopwords.txt'):
        self.docs = docs
        self.stop_words = self.load_stopwords(stopwords_file)
        self.tokenized_corpus = self.tokenize_corpus()
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def load_stopwords(self, stopwords_file):
        stop_words = set()
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                stop_words.add(line.strip())
        return stop_words

    def bm25_tokenizer(self, text):
        tokenized_doc = []
        tokens = jieba.cut_for_search(text)
        for token in tokens:
            token = token.strip(string.punctuation)
            if len(token) > 0 and token not in self.stop_words:
                tokenized_doc.append(token)
        return tokenized_doc

    def tokenize_corpus(self):
        tokenized_corpus = []
        for passage in tqdm(self.docs):
            tokenized_corpus.append(self.bm25_tokenizer(passage.page_content))
        return tokenized_corpus

    def search(self, query,threshold):
        print("Input question:", query)

        bm25_scores = self.bm25.get_scores(self.bm25_tokenizer(query))
        
        bm25_hits = [{'corpus_id': idx, 'score': score} for idx, score in enumerate(bm25_scores) if score >= threshold]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

        bm25_hits_res = [self.docs[hit['corpus_id']] for hit in bm25_hits]

        return bm25_hits_res
    
