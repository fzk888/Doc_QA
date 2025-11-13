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
from typing import List
from langchain_core.runnables import RunnablePassthrough
from Knowledge_based_async import KnowledgeBase
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from FlagEmbedding import FlagReranker
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from document_reranker import DocumentReranker
from bm25_search import BM25Search
import time
import random
from concurrent.futures import ThreadPoolExecutor
# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docqa")

model_kwargs = {"device": config['settings']['device']}
encode_kwargs = {
    "batch_size": config['settings']['batch_size'],
    "normalize_embeddings": config['settings']['normalize_embeddings']
}
_embeddings = None
_reranker_model = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info("正在初始化嵌入模型（BGE）...")
        _embeddings = HuggingFaceBgeEmbeddings(
            model_name=config['paths']['model_dir'],
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    return _embeddings

def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = FlagReranker(config['paths']['reranker_model_dir'], use_fp16=config['settings']['use_fp16'])
    return _reranker_model


class KBState:
    def __init__(self):
        self.kb = None
        self.kb_vectordb = None
        self.current_kb_name = None
        self.history = []
        self.unfilter_context = []
        self.searcher_from_target_doc = None

kb_state = KBState()
    
def get_top_documents(query: str):
    # global kb_state
    logger.info(f"Current KB Name: {kb_state.current_kb_name}")
    if not kb_state.kb_vectordb or not kb_state.searcher_from_target_doc:
        raise ValueError("Knowledge base not loaded.")

    # 适度降低向量检索返回数量，减少后续重排负载
    logger.info("正在进行向量检索...")
    retriever = kb_state.kb_vectordb.as_retriever(search_kwargs={"k": 4})
    # 并行执行向量检索与 BM25 检索以降低总体等待时间
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_bge = ex.submit(retriever.invoke, query)
        logger.info("正在进行 BM25 检索...")
        fut_bm25 = ex.submit(kb_state.searcher_from_target_doc.search, query, 0.2)
        bge_context = fut_bge.result()
        bm25_context = fut_bm25.result()
    # BM25 已经按得分排序，限制候选数量避免过多文档进入重排
    if len(bm25_context) > 8:
        bm25_context = bm25_context[:8]

    logger.info("正在融合向量检索与 BM25 结果...")
    merged_res = bge_context + bm25_context

    if len(merged_res) <= 1:
        return [(merged_res[0], 0.3)] if merged_res else []

    # 基于 file_path 去重，避免对长文本 page_content 做哈希比较
    unique_by_path = {}
    for doc in merged_res:
        file_path = doc.metadata.get('file_path')
        if file_path and file_path not in unique_by_path:
            unique_by_path[file_path] = doc
    unique_docs = list(unique_by_path.values())

    logger.info("正在重排候选文档...")
    reranker = DocumentReranker(get_reranker_model())
    # 控制进入重排的最大候选数量
    candidates = unique_docs[:8]
    top_documents_with_scores = reranker.rerank_documents(query, candidates, top_n=3)

    # 返回 (Document, score) 且要求存在 file_path 元数据
    return [(doc, round(score, 2)) for doc, score in top_documents_with_scores if doc.metadata.get('file_path')]

def run_llm_Knowlege_baes_file_QA(query: str, keep_history: bool = True):
    openai_api_key = config['paths']['openai_api_keys']
    openai_api_base = config['paths']['openai_api_base']

    llm = ChatOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        model=config['models']['llm_model'],
        temperature=0.2,
        streaming=True
    )

    history_str = "\n".join([str(item) for item in kb_state.history]) + "\n这是以上我和你的对话记录，请参考\n"
    # use global kb_state.kb_vectordb; fall back to empty set if not available
    if getattr(kb_state, 'kb_vectordb', None) is None:
        uploaded_files = set()
    else:
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
    for chunk in rag_chain.stream(query):
        response_text += chunk
        yield chunk

    if keep_history:
        kb_state.history.append({"query": query, "response": response_text})

def find_image_links(documents):
    image_info = []
    for doc in documents:
        matches = re.findall(r'!\[.*?\]\((.*?)\)', doc.page_content)
        image_info.extend(matches)
    return image_info



#使用 OpenAI API 来生成引导性问题。这个函数将遍历知识库中的每个文档，并生成一个与文档内容相关的引导性问题。


def generate_guiding_questions(num_questions_total=3, num_questions_per_doc=2):
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=0.2,
        streaming=True
    )
    # global kb_state
    kb_vectordb = kb_state.kb_vectordb
    doc_groups = {}
    for doc_id, doc in kb_vectordb.docstore._dict.items():
        source = doc.metadata.get('source', 'unknown')
        if source not in doc_groups:
            doc_groups[source] = []
        doc_groups[source].append(doc)
    
    guiding_questions = []
    
    # 如果只有一个文档组，且需要的问题数量大于每个文档的问题数量
    if len(doc_groups) == 1 and num_questions_total > num_questions_per_doc:
        single_source = list(doc_groups.keys())[0]
        docs = doc_groups[single_source]
        
        for i in range(num_questions_total):
            doc = random.choice(docs)  # 随机选择一个文档，可能会重复
            question = generate_single_question(llm, doc)
            guiding_questions.append(question)
    else:
        # 遍历每个文档组并生成引导性问题
        for source, docs in doc_groups.items():
            # 随机抽取部分内容生成问题
            selected_docs = random.sample(docs, min(num_questions_per_doc, len(docs)))
            
            for doc in selected_docs:
                question = generate_single_question(llm, doc)
                guiding_questions.append(question)
    
    # 如果生成的问题不足所需数量，随机选择文档继续生成
    while len(guiding_questions) < num_questions_total:
        random_source = random.choice(list(doc_groups.keys()))
        random_doc = random.choice(doc_groups[random_source])
        question = generate_single_question(llm, random_doc)
        guiding_questions.append(question)
    
    return guiding_questions[:num_questions_total]

def generate_single_question(llm, doc):
    variations = [
        "请根据以下内容提出一个引导性问题,不超过15个字。",
        "根据以下内容，生成一个引导性问题,不超过15个字。",
        "阅读以下内容后，提出一个引导性问题,不超过15个字。",
        "基于给定的文本，创建一个引导性问题,不超过15个字。",
        "考虑以下信息，形成一个引导性问题,不超过15个字。"
    ]
    random_variation = random.choice(variations)
    prompt = f"{random_variation}\n内容: {doc.page_content}\n，引导性问题内容要确保有主体，内容不能太复杂，问题要有引导意义，引导用户提问。只能生成一个问题不能有子问题，一步步思考，直接返回问题，不需要任何开头或解释。例如：深圳北站哪个出口最适合打滴滴？"
    
    result = llm.invoke(prompt).content
    
    return result.strip()

def get_document_snippets(documents, max_length=800):
    snippets = [doc.page_content[:max_length] for doc in documents]
    return "\n".join(snippets)

def document_question_relevance(question, documents):
    #openai_api_key = "EMPTY"
    #openai_api_base = config['paths']['openai_api_base']
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=0,
        streaming=True
    )


    try:
        document = get_document_snippets(documents)

        prompt = PromptTemplate(
            template="""作为文档相关性评估专家，你的任务是判断给定文档是否与用户问题相关。请仔细阅读以下信息：
    
                        用户问题：{question}
    
                        文档内容：
                        {document}
    
                        评估指南：
                        1. 关注文档中与问题相关的关键词、概念或主题。
                        2. 考虑文档是否提供了回答问题所需的信息或背景。
                        3. 即使文档不能完全回答问题，只要包含相关信息也可视为相关。
                        4. 这不是严格的匹配测试，目的是过滤掉明显不相关的文档。
    
                        请根据上述指南，给出你的评估结果。
    
                        输出要求：
                        - 仅返回一个JSON对象，包含一个键"score"
                        - "score"的值必须是"是"或"否"
                        - 不要包含任何其他解释或评论
    
                        示例输出：
                        {{"score": "是"}}
                        或
                        {{"score": "否"}}""",
                                input_variables=["question", "document"],
                            )

        retrieval_grader = prompt | llm | JsonOutputParser()
        result = retrieval_grader.invoke({"question": question, "document": document})
    except:
        document = documents[0].page_content.split('\n')
        prompt = PromptTemplate(
            template="""作为文本相关性评估专家，你的任务是判断给定的两段文本意思是否相近。请仔细阅读以下信息：
    
                        文本一：{question}
    
                        文本二：
                        {document}
    
                        评估指南：
                        1. 关注两段文本的关键词、概念或主题。
                        2. 即使两端段文本不能完全回答问题，只要包含相关信息也可视为相关。
    
                        请根据上述指南，给出你的评估结果。
    
                        输出要求：
                        - 仅返回一个JSON对象，包含一个键"score"
                        - "score"的值必须是"是"或"否"
                        - 不要包含任何其他解释或评论
    
                        示例输出：
                        {{"score": "是"}}
                        或
                        {{"score": "否"}}""",
                                input_variables=["question", "document"],
                            )

        retrieval_grader = prompt | llm | JsonOutputParser()
        result = retrieval_grader.invoke({"question": question, "document": document})
    
    return result['score']

def is_math_question(question):
    #openai_api_key = "EMPTY"
    #openai_api_base = config['paths']['openai_api_base']
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=0.2,
        streaming=True
    )
    prompt = PromptTemplate(
        template="""你是一个问题分类员，评估一个问题是否需要通过计算过程才能回答是否有一系列事实依据。
                    以下是问题:
                    \n ------- 
                    {question}
                    \n ------
                    给出一个二元分数“是”或“否”，以表明答案是否基于支持/问题是否需要通过计算过程才能回答。
                    以JSON形式提供，只有一个关键字‘score’，没有序言或解释。""",
        input_variables=["question"],
    )
    hallucination_grader = prompt | llm | JsonOutputParser()
    response = hallucination_grader.invoke({"question": question})
    return response["score"]

def history_list_to_str(history):
    return "\n".join([str(item) for item in history]) + "\n这是以上我和你的对话记录，请参考\n"

def question_generation_from_last_dialogual(last_dialog: str):
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=0.2,
        streaming=False
    )

    prompt = PromptTemplate(
        template="""你是一名问题引导专家，请根据上一轮的对话，提出3个引导性问题。每个问题之间用换行符分隔。
                    请注意：
                    问题前不需要任何序号。
                    问题内容要简洁明确，确保有明确的主题。
                    引导性问题要能够启发用户进一步思考或提问。
                    示例 1：
                    上一轮对话记录：
                    用户：我最近对机器学习很感兴趣，但不知道从哪里开始。你有什么建议吗？
                    AI：你可以从学习基础概念开始，比如监督学习和无监督学习，然后逐步深入到常见的算法和应用领域。
                    引导性问题：
                    你对监督学习和无监督学习的区别有什么了解吗？
                    你是否有兴趣了解一些常见的机器学习算法，比如决策树和神经网络？
                    你目前有接触到哪些与机器学习相关的项目或资源？
                    示例 2：
                    上一轮对话记录：
                    用户：我最近在学习Python编程，但是遇到了一些困难，特别是在理解面向对象编程的概念时。
                    AI：面向对象编程（OOP）是Python中的一个重要概念，它包括类和对象的使用，封装、继承和多态等特性。
                    引导性问题：
                    你对类和对象的概念有具体的疑问吗？
                    你想了解封装、继承和多态这些特性是如何在Python中实现的吗？
                    你是否遇到了具体的编程问题，可以举个例子吗？
                    这是上一轮对话的记录：{last_dialog}
                    请直接生成问题，不需要任何开头或解释。请根据上述要求生成引导性问题：""",
        input_variables=["last_dialog"]
    )
    rag_chain = (
        {"last_dialog": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    derived_questions_str = rag_chain.invoke(last_dialog)
    derived_questions = [question for question in derived_questions_str.strip().split("\n") if question]
    return derived_questions

def run_llm_MulitDocQA(input_query: str, only_chatKBQA: bool, prompt_template_from_user: str, temperature: float, multiple_dialogue: bool, derivation: bool, show_source: bool):
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=temperature,
        streaming=True
    )
    # global kb_state
    logger.info("enter run_llm_MulitDocQA")
    try:
        logger.info(f"kb_state.current_kb_name={kb_state.current_kb_name} kb_vectordb_is_none={kb_state.kb_vectordb is None}")
    except Exception:
        logger.info(f"kb_state: {kb_state}")
    if len(input_query) == 1:
        query = input_query[0].content
        history_str = ""
    elif len(input_query) == 2:
        query = input_query[-1].content
        history_str = history_list_to_str([query[0]])
    else:
        query = input_query[-1].content
        history_str = history_list_to_str([query[0:-1]])
#    history_str = history_list_to_str(history)
    logger.info(f"input_query: {input_query}")
    
    def generate_prompt(template, input_variables):
        return PromptTemplate(template=template, input_variables=input_variables)
    
    def create_chain(prompt):
        return prompt | llm | StrOutputParser()
    
    if only_chatKBQA:
        top_documents_with_socre = get_top_documents(query)
        top_documents = [doc for doc, score in top_documents_with_socre]

        try:
            top_documents[0].metadata["isQA"]
            if document_question_relevance(query, top_documents[0]) == '是':
                logger.info("问题和问答文档相关")
                answer = "\n".join(top_documents[0].page_content.split('\n')[1:])
                logger.info(f"answer (truncated 200 chars): {answer[:200]}")
                if top_documents[0].metadata["file_url"] != '-':
                    str_l = len(answer[:-1])
                    try:
                        for i in range(str_l // 3):
                            logger.info(answer[:-1][i*3:(i+1)*3])
                            yield stream_type(answer[:-1][i*3:(i+1)*3])
                            time.sleep(0.2)
                        yield stream_type(answer[:-1][(i+1)*3:(i+1)*3 + str_l%3])
                        time.sleep(0.2)
                        yield stream_type_url(answer[-1],top_documents[0].metadata["file_url"])
                    except:
                        yield stream_type_url(answer,top_documents[0].metadata["file_url"])

                else:
                    str_l = len(answer)

                    try:
                        for i in range(str_l // 3):
                            logger.info(answer[i*3:(i+1)*3])
                            yield stream_type(answer[i*3:(i+1)*3])
                            time.sleep(0.2)
                        yield stream_type(answer[(i+1)*3:(i+1)*3 + str_l%3])
                    except:
                        yield stream_type(answer)
            else:
                logger.info("问题和问答文档不相关")
                if document_question_relevance(query, top_documents) == '是':
                    logger.info("文档和问题相关")
                    template = (prompt_template_from_user or "您是一位由 Dana AI 开发的大型语言人工智能助手。请严格根据段落内容分点作答用户问题，请极大程度保留段落的格式与内容进行简要回答。如果涉及计算请按步骤进行计算，注意不要杜撰段落没有提及的要点，且不要重复。如果文档中出现代码相关的信息，可以将完整代码返回，如果给出的段落信息与原文无关") + \
                                (f"\n以下是历史对话记录：{{history}},请参考历史对话记录。" if multiple_dialogue else "") + \
                                "\n以下是相关段落:{top_documents},下面是用户问题：{query} 回答："
                    
                    input_variables = ["query", "top_documents"]
                    if multiple_dialogue:
                        input_variables.append("history")
                    
                    prompt = generate_prompt(template, input_variables)
                    chain = create_chain(prompt)
                    
                    inputs = {"query": query, "top_documents": top_documents}
                    if multiple_dialogue:
                        inputs["history"] = history_str
                    
                    response_text = ""
                    for chunk in chain.stream(inputs):
                        response_text += chunk
                        yield stream_type(chunk)
                    
                    #update_history(query, response_text)
                else:
                    yield f"data: {json.dumps(create_response_dict(content='没有检索到与查询相关的上下文信息,对不起,知识库中没有找到可以回答此问题的相关信息。', image_list=None, documents=None, sources=None), ensure_ascii=False)}\n\n".encode('utf-8')

        ###########################################################################################

        except:
            if document_question_relevance(query, top_documents) == '是':
                logger.info("文档和问题相关")
                template = (prompt_template_from_user or "您是一位由 Dana AI 开发的大型语言人工智能助手。请严格根据段落内容分点作答用户问题，请极大程度保留段落的格式与内容进行简要回答。如果涉及计算请按步骤进行计算，注意不要杜撰段落没有提及的要点，且不要重复。如果文档中出现代码相关的信息，可以将完整代码返回，如果给出的段落信息与原文无关") + \
                            (f"\n以下是历史对话记录：{{history}},请参考历史对话记录。" if multiple_dialogue else "") + \
                            "\n以下是相关段落:{top_documents},下面是用户问题：{query} 回答："
                
                input_variables = ["query", "top_documents"]
                if multiple_dialogue:
                    input_variables.append("history")
                
                prompt = generate_prompt(template, input_variables)
                chain = create_chain(prompt)
                
                inputs = {"query": query, "top_documents": top_documents}
                if multiple_dialogue:
                    inputs["history"] = history_str
                
                response_text = ""
                for chunk in chain.stream(inputs):
                    response_text += chunk
                    yield stream_type(chunk)
                
                #update_history(query, response_text)
            else:
                yield f"data: {json.dumps(create_response_dict(content='没有检索到与查询相关的上下文信息,对不起,知识库中没有找到可以回答此问题的相关信息。', image_list=None, documents=None, sources=None), ensure_ascii=False)}\n\n".encode('utf-8')
    else:
        top_documents_with_socre = get_top_documents(query)
        top_documents = [doc for doc, score in top_documents_with_socre]
        document_question_relevance(query, top_documents[0])
        try:
            top_documents[0].metadata["isQA"]
            logger.info(f"document_question_relevance: {document_question_relevance(query, top_documents[0])}")
            if document_question_relevance(query, top_documents[0]) == '是':
                logger.info("问题和问答文档相关")
                answer = "\n".join(top_documents[0].page_content.split('\n')[1:])
                logger.info(f"file_url: {top_documents[0].metadata.get('file_url')}")
                logger.info("===")
                if top_documents[0].metadata["file_url"] != '-':
                    str_l = len(answer[:-1])
                    try:
                        for i in range(str_l // 3):
                            logger.info(answer[:-1][i*3:(i+1)*3])
                            yield stream_type(answer[:-1][i*3:(i+1)*3])
                            time.sleep(0.2)
                        yield stream_type(answer[:-1][(i+1)*3:(i+1)*3 + str_l%3])
                        time.sleep(0.2)
                        
                        yield stream_type_url(answer[-1],top_documents[0].metadata["file_url"])
                    except:
                        yield stream_type_url(answer,top_documents[0].metadata["file_url"])

                else:
                    str_l = len(answer)

                    try:
                        for i in range(str_l // 3):
                            logger.info(answer[i*3:(i+1)*3])
                            yield stream_type(answer[i*3:(i+1)*3])
                            time.sleep(0.2)
                        yield stream_type(answer[(i+1)*3:(i+1)*3 + str_l%3])
                    except:
                        yield stream_type(answer)

            else:
                logger.info("问题和问答文档不相关")
                if document_question_relevance(query, top_documents) == '是':
                    logger.info("文档和问题相关")
                    template = (prompt_template_from_user or "您是一位由 Dana AI 开发的大型语言人工智能助手。请严格根据段落内容分点作答用户问题，请极大程度保留段落的格式与内容进行简要回答。如果涉及计算请按步骤进行计算，注意不要杜撰段落没有提及的要点，且不要重复。如果文档中出现代码相关的信息，可以将完整代码返回，如果给出的段落信息与原文无关") + \
                                (f"\n以下是历史对话记录：{{history}},请参考历史对话记录。" if multiple_dialogue else "") + \
                                "\n以下是相关段落:{top_documents},下面是用户问题：{query} 回答："
                    
                    input_variables = ["query", "top_documents"]
                    if multiple_dialogue:
                        input_variables.append("history")
                    
                    prompt = generate_prompt(template, input_variables)
                    chain = create_chain(prompt)
                    
                    inputs = {"query": query, "top_documents": top_documents}
                    if multiple_dialogue:
                        inputs["history"] = history_str
                    
                    response_text = ""
                    for chunk in chain.stream(inputs):
                        response_text += chunk
                        yield stream_type(chunk)

                else:
                    if prompt_template_from_user:
                        template = ("以上是历史信息{history_str}，" if multiple_dialogue else "") + \
                        prompt_template_from_user + "根据用户的要求或提问{query},及时给用户满意的反馈和回答。回复："
                    else:
                        template = ("以上是历史信息{history_str}，" if multiple_dialogue else "") + \
                            "您是一位由 Dana AI 开发的大型语言人工智能助手。热情有礼貌的和用户进行交互，根据用户的要求或提问{query},及时给用户满意的反馈和回答。回复："
                
                    input_variables = ["query"]
                    if multiple_dialogue:
                        input_variables.append("history")
                    
                    prompt = generate_prompt(template, input_variables)
                    chain = create_chain(prompt)
                    
                    inputs = {"query": query}
                    if multiple_dialogue:
                        inputs["history_str"] = history_str
                    
                    response_text = ""
                    for chunk in chain.stream(inputs):
                        response_text += chunk
                        yield stream_type(chunk)
                    
                    #update_history(query, response_text)

        ###########################################################################################
        except:
            if document_question_relevance(query, top_documents) == '是':
                logger.info("文档和问题相关")

                template = (prompt_template_from_user or "您是一位由 Dana AI 开发的大型语言人工智能助手。请严格根据段落内容分点作答用户问题，请极大程度保留段落的格式与内容进行简要回答。如果涉及计算请按步骤进行计算，注意不要杜撰段落没有提及的要点，且不要重复。如果文档中出现代码相关的信息，可以将完整代码返回，如果给出的段落信息与原文无关") + \
                            (f"\n以下是历史对话记录：{{history}},请参考历史对话记录。" if multiple_dialogue else "") + \
                            "\n以下是相关段落:{top_documents},下面是用户问题：{query} 回答："
                
                input_variables = ["query", "top_documents"]
                if multiple_dialogue:
                    input_variables.append("history")
                
                prompt = generate_prompt(template, input_variables)
                chain = create_chain(prompt)
                
                inputs = {"query": query, "top_documents": top_documents}
                if multiple_dialogue:
                    inputs["history"] = history_str
                
                response_text = ""
                for chunk in chain.stream(inputs):
                    response_text += chunk
                    yield stream_type(chunk)
                
                #update_history(query, response_text)
        
            else:
                if prompt_template_from_user:
                    template = ("以上是历史信息{history_str}，" if multiple_dialogue else "") + \
                    prompt_template_from_user + "根据用户的要求或提问{query},及时给用户满意的反馈和回答。回复："
                else:
                    template = ("以上是历史信息{history_str}，" if multiple_dialogue else "") + \
                        "您是一位由 Dana AI 开发的大型语言人工智能助手。热情有礼貌的和用户进行交互，根据用户的要求或提问{query},及时给用户满意的反馈和回答。回复："
                                
                input_variables = ["query"]
                if multiple_dialogue:
                    input_variables.append("history")
                
                prompt = generate_prompt(template, input_variables)
                chain = create_chain(prompt)
                
                inputs = {"query": query}
                if multiple_dialogue:
                    inputs["history_str"] = history_str
                
                response_text = ""
                for chunk in chain.stream(inputs):
                    response_text += chunk
                    yield stream_type(chunk)


def only_llm(input_query: str, only_chatKBQA: bool, prompt_template_from_user: str, temperature: float, multiple_dialogue: bool, derivation: bool, show_source: bool):
    llm = ChatOpenAI(
        api_key=config['paths']['openai_api_keys'],
        base_url=config['paths']['openai_api_base'],
        model=config['models']['llm_model'],
        temperature=temperature,
        streaming=True
    )
    if len(input_query) == 1:
        query = input_query[0].content
        history_str = ""
    elif len(input_query) == 2:
        query = input_query[-1].content
        history_str = history_list_to_str([query[0]])
    else:
        query = input_query[-1].content
        history_str = history_list_to_str([query[0:-1]])
#    history_str = history_list_to_str(history)
    logger.info(f"only_llm input_query: {input_query}")
    
    def generate_prompt(template, input_variables):
        return PromptTemplate(template=template, input_variables=input_variables)
    
    def create_chain(prompt):
        return prompt | llm | StrOutputParser()
    
    template = (prompt_template_from_user or "您是一位由 Dana AI 开发的大型语言人工智能助手。请严格根据段落内容分点作答用户问题，请极大程度保留段落的格式与内容进行简要回答。如果涉及计算请按步骤进行计算，注意不要杜撰段落没有提及的要点，且不要重复。如果文档中出现代码相关的信息，可以将完整代码返回，如果给出的段落信息与原文无关") + \
                    (f"\n以下是历史对话记录：{{history}},请参考历史对话记录。" if multiple_dialogue else "") + \
                    "下面是用户问题：{query} 回答："
                
    input_variables = ["query"]
    if multiple_dialogue:
        input_variables.append("history")
                
    prompt = generate_prompt(template, input_variables)
    chain = create_chain(prompt)
                
    inputs = {"query": query}
    if multiple_dialogue:
        inputs["history"] = history_str
    
    response_text = ""
    for chunk in chain.stream(inputs):
        response_text += chunk
        yield stream_type(chunk)

def view_history(history):
    enc = tiktoken.get_encoding("cl100k_base")
    history_str = ""
    total_tokens = 0
    for item in history:
        query = item["query"]
        response = item["response"]
        history_str += f"User: {query}\nAssistant: {response}\n\n"
        total_tokens += len(enc.encode(query)) + len(enc.encode(response))
    return history_str, total_tokens

def get_uploaded_files():
    # Use kb_state.kb_vectordb (global state) instead of undefined kb_vectordb
    if getattr(kb_state, 'kb_vectordb', None) is None:
        return set()
    return {os.path.basename(doc.metadata.get('file_path', '')) for doc in kb_state.kb_vectordb.docstore._dict.values()}

def stream_type(data, model=config['models']['llm_model']):
    return f"data: {json.dumps({'id': str(uuid.uuid4()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': data}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n".encode('utf-8')

def stream_type_url(data,file_url, model=config['models']['llm_model']):
    return f"data: {json.dumps({'id': str(uuid.uuid4()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': data,'file_url': file_url}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n".encode('utf-8')

def create_response_dict(**kwargs):
    return {
        "id": str(uuid.uuid4()),
        "model": config['models']['llm_model'],
        "choices": [{"index": 0, "delta": kwargs, "finish_reason": None}]
    }
#生成最后一个流式回复
def create_final_response(current_dialog, show_source, top_documents_with_score, derivation=False):
    derived_questions = None

    # 如果 derivation 为 True，获取派生问题
    if derivation:
        derived_questions = question_generation_from_last_dialogual(str(current_dialog))

    # 准备最后一个包含所有信息的响应
    final_response = {
        "content": " ",
        "image_list": None,
        "doc": [],
        "derived_questions": derived_questions,
        "user_url": None
    }

    if show_source:
        # 创建一个字典来存储 source 到 document 列表的映射
        doc_dict = OrderedDict()
        for doc, score in top_documents_with_score:
            document_content = doc.page_content
            source = os.path.basename(doc.metadata.get('file_path', 'unknown'))
            
            if source not in doc_dict:
                doc_dict[source] = []
            
            doc_dict[source].append({
                "text": document_content,
                "mate": score
            })
        
        # 将字典转换为所需的 doc 列表格式
        for source, documents in doc_dict.items():
            final_response["doc"].append({
                "source": source, 
                "document": documents
            })
    top_documents = [doc for doc, score in top_documents_with_score]
    # 查找图片信息
    image_info = find_image_links(top_documents)
    if image_info:
        final_response["image_list"] = image_info

    try:
        if top_documents[0].metadata["file_url"] != '-':
            final_response["user_url"] = top_documents[0].metadata["file_url"]
    except:
        pass

    return final_response
