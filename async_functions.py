import os
import json
import shutil
import logging
import random
import re
import uuid
from collections import OrderedDict
import aiofiles
import tiktoken
import yaml
from fastapi import HTTPException
from typing import List
from langchain_core.runnables import RunnablePassthrough
from Knowledge_base import KnowledgeBase
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from FlagEmbedding import FlagReranker
from openai import AsyncOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import AsyncChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from document_reranker import DocumentReranker
from bm25_search import BM25Search
import time
import asyncio

async def setup_environment():
    # Load configuration
    async with aiofiles.open("config.yaml", "r") as config_file:
        config = yaml.safe_load(await config_file.read())

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load embeddings and reranker models
    model_kwargs = {"device": config['settings']['device']}
    encode_kwargs = {
        "batch_size": config['settings']['batch_size'],
        "normalize_embeddings": config['settings']['normalize_embeddings']
    }
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=config['paths']['model_dir'],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    reranker_model = FlagReranker(config['paths']['reranker_model_dir'], use_fp16=config['settings']['use_fp16'])
    
    
class KBState:
    def __init__(self):
        self.kb = None
        self.kb_vectordb = None
        self.current_kb_name = None
        self.history = []
        self.unfilter_context = []
        self.searcher_from_target_doc = None

kb_state = KBState()

async def load_vectordb_and_files():
    global kb_state

    kb_list = os.listdir(config['paths']['kb_dir'])
    if kb_list:
        default_kb_name = kb_list[2]
        print(default_kb_name)
        kb = KnowledgeBase(default_kb_name, embeddings)
        kb_state.kb_vectordb = await asyncio.to_thread(kb.load_vectordb)
        kb_state.unfilter_context = [doc for doc_id, doc in kb_state.kb_vectordb.docstore._dict.items()]
        kb_state.searcher_from_target_doc = await asyncio.to_thread(BM25Search, kb_state.unfilter_context)
    else:
        raise FileNotFoundError("No knowledge bases found.")

async def get_top_documents(query: str):
    logger.info(f"Current KB Name: {kb_state.current_kb_name}")
    if not kb_state.kb_vectordb or not kb_state.searcher_from_target_doc:
        raise ValueError("Knowledge base not loaded.")

    retriever = kb_state.kb_vectordb.as_retriever(search_kwargs={"k": 5})
    bge_context = await asyncio.to_thread(retriever.get_relevant_documents, query)
    bm25_context = await asyncio.to_thread(kb_state.searcher_from_target_doc.search, query, threshold=0.2)
    merged_res = bge_context + bm25_context

    if len(merged_res) <= 1:
        return [(merged_res[0], 0.3)] if merged_res else []

    unique_res = list({doc.page_content: doc for doc in merged_res}.values())
    reranker = DocumentReranker(reranker_model)
    top_documents_with_scores = await asyncio.to_thread(reranker.rerank_documents, query, unique_res, top_n=3)
    unique_top_documents = list({doc.page_content: (doc, round(score, 2)) for doc, score in top_documents_with_scores}.values())
    rr = [i for i in unique_top_documents if i[0].metadata.get('file_path') is not None]
    return rr

async def run_llm_Knowledge_base_file_QA(query: str, keep_history: bool = True):
    openai_api_key = config['paths']['openai_api_keys']
    openai_api_base = config['paths']['openai_api_base']

    llm = AsyncChatOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        model=config['models']['llm_model'],
        temperature=0.2,
        streaming=True
    )

    history_str = "\n".join([str(item) for item in kb_state.history]) + "\n这是以上我和你的对话记录，请参考\n"
    uploaded_files = {os.path.basename(doc.metadata.get('file_path', '')) for doc in kb_state.kb_vectordb.docstore._dict.values()}
    prompt_template = "以上是历史信息{history_str}，您是一位由 Dana AI 开发的大型语言人工智能助手。您将被提供一个用户问题,根据知识库文档列表{uploaded_files},结合问题{query}，撰写一个清晰、简洁且准确的答案。回答："
    prompt = PromptTemplate(template=prompt_template, input_variables=["history_str", "uploaded_files", "query"])
    rag_chain = (
        {"history_str": lambda x: history_str, "uploaded_files": lambda x: uploaded_files, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response_text = ""
    async for chunk in rag_chain.astream(query):
        response_text += chunk
        yield chunk

    if keep_history:
        kb_state.history.append({"query": query, "response": response_text})

async def find_image_links(documents):
    image_info = []
    for doc in documents:
        matches = re.findall(r'!\[.*?\]\((.*?)\)', doc.page_content)
        image_info.extend(matches)
    return image_info