from typing import List, Tuple
from langchain.docstore.document import Document
import numpy as np

class DocumentReranker:
    def __init__(self, reranker_model):
        """
        初始化DocumentReranker。

        参数:
            reranker_model: 重排模型的实例。
        """
        self.reranker = reranker_model

    def rerank_documents(self, check_item: str, documents: List[Document], top_n: int = 1) -> List[Tuple[Document, float]]:
        """
        根据给定的关键词对Document对象列表进行重排,并返回得分最高的前N个Document对象及其分数。
        
        参数:
            check_item (str): 关键词。
            documents (List[Document]): Document对象列表。
            top_n (int): 返回的Document数量,默认为1。
            
        返回:
            List[Tuple[Document, float]]: 得分最高的前N个Document对象及其分数的列表。
        """
        sentence_pairs = [(check_item, doc.page_content) for doc in documents]
        
        scores = self.reranker.compute_score(sentence_pairs, normalize=True)
        
        # 确保 scores 是一个列表
        if isinstance(scores, (float, np.float64)):
            scores = [scores]
        elif isinstance(scores, np.ndarray):
            scores = scores.tolist()
        
        # 确保 scores 的长度与 documents 相同
        if len(scores) != len(documents):
            raise ValueError(f"Scores length ({len(scores)}) does not match documents length ({len(documents)})")
        
        documents_with_scores = list(zip(documents, scores))
        sorted_documents_with_scores = sorted(documents_with_scores, key=lambda x: x[1], reverse=True)
        
        top_documents_with_scores = sorted_documents_with_scores[:top_n]
        
        return top_documents_with_scores