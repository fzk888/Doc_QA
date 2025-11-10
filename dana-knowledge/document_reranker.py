from typing import List
from langchain.docstore.document import Document

class DocumentReranker:
    def __init__(self, reranker_model):
        """
        初始化DocumentReranker。

        参数:
            reranker_model: 重排模型的实例。
        """
        self.reranker = reranker_model

    def rerank_documents(self, check_item: str, documents: List[Document], top_n: int = 1):
        """
        根据给定的关键词对Document对象列表进行重排,并返回得分最高的前N个Document对象。
        
        参数:
            check_item (str): 关键词。
            documents (List[Document]): Document对象列表。
            top_n (int): 返回的Document数量,默认为1。
            
        返回:
            List[Document]: 得分最高的前N个Document对象。
        """
        sentence_pairs = [(check_item, doc.page_content) for doc in documents]
        
        scores = self.reranker.compute_score(sentence_pairs, normalize=True)
        
        documents_with_scores = zip(documents, scores)
        sorted_documents = sorted(documents_with_scores, key=lambda x: x[1], reverse=True)
        
        top_documents = [doc_with_score[0] for doc_with_score in sorted_documents[:top_n]]
        
        return top_documents