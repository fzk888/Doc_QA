from Knowledge_base import KnowledgeBase
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from FlagEmbedding import FlagReranker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
from PIL import Image
import logging
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI, File, UploadFile, Form, HTTPException,Response
from fastapi.responses import StreamingResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import aiofiles
import asyncio
from document_reranker import DocumentReranker
from bm25_search import BM25Search
from tqdm.autonotebook import tqdm
from FlagEmbedding import FlagReranker
from fastapi.responses import JSONResponse
import base64
import uvicorn
import tiktoken
import shutil
import string
import random
import json
import uuid
import gc
import re
import os
import os
import re
import shutil

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载编码模型和重排模型
model_name = "/root/autodl-tmp/models/BAAI/bge-large-zh-v1.5"
model_kwargs = {"device": "cuda"} #cpu
encode_kwargs = {
    "batch_size": 1024,
    "normalize_embeddings": True
}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
reranker_model = FlagReranker('/root/autodl-tmp/models/BAAI/bge-reranker-large/quietnight/bge-reranker-large', use_fp16=True)

#先指定知识库
kb = None
kb_vectordb = None
current_kb_name = None
history = []
def stream_type(data, model="Qwen1.5-32B-Chat"):
    return f"data: {json.dumps({'id': str(uuid.uuid4()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': data}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n".encode('utf-8')
try:
    # 在全局范围内加载知识库和构建 BM25 向量库
    unfilter_context = [doc for doc_id, doc in kb_vectordb.docstore._dict.items()]
    searcher_from_target_doc = BM25Search(unfilter_context)
except FileNotFoundError:
    print("知识库或 BM25 向量库文件不存在,请确保已经正确构建并保存了知识库和向量库。")
    kb_vectordb = None
    unfilter_context = None
    searcher_from_target_doc = None
except Exception as e:
    print(f"加载知识库或构建 BM25 向量库时出现异常: {str(e)}")
    kb_vectordb = None
    unfilter_context = None
    searcher_from_target_doc = None

def run_llm_Knowlege_baes_file_QA(query: str,keep_history: bool = True):
    openai_api_key = "EMPTY"
    openai_api_base = "http://region-9.autodl.pro:23971/v1"

    llm = ChatOpenAI(api_key=openai_api_key,
                base_url=openai_api_base,
                model='Qwen1.5-32B-Chat',
                temperature=0.2,
                streaming=True)

    def history_list_to_str(history):
        result = ""
        for item in history:
            result += str(item) + "\n"
        return result + "\n这是以上我和你的对话记录，请参考\n" 

    history_str = history_list_to_str(history)
    #先加载知识库
    
    uploaded_files = {os.path.basename(doc.metadata.get('file_path', ''))
                for doc in kb_vectordb.docstore._dict.values()}
    prompt_template = """以上是历史信息{history_str}，您是一位大型语言人工智能助手。您将被提供一个用户问题,根据知识库文档列表{uploaded_files},结合问题{query}，撰写一个清晰、简洁且准确的答案。回答："""
    prompt = PromptTemplate(template=prompt_template, input_variables=["history_str", "uploaded_files","query"])
    rag_chain = (
    {"history_str": lambda x: history_str,"uploaded_files": lambda x: uploaded_files, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
    response_text = ""
    for chunk in rag_chain.stream(query):
        response_text += chunk
        yield chunk
    if keep_history:
            # 将当前查询和回答添加到历史记录列表中
            history.append({"query": query, "response": response_text})

def find_image_links(documents):
    image_info = []
    for doc in documents:
        # 假设图片链接格式为 "![image](http://example.com/image.jpg)"
        matches = re.findall(r'!\[.*?\]\((.*?)\)', doc.page_content)
        for match in matches:
            # 获取图片链接
            image_link = match
            image_info.append(image_link)
    return image_info


def get_top_documents(query: str):
    # 语义检索
    retriever = kb_vectordb.as_retriever(search_kwargs={"k": 5})
    bge_context = retriever.get_relevant_documents(query)
    print("向量检索:\n\n", bge_context)

    # BM25关键词检索
    bm25_threshold = 0.2  # BM25检索的阈值
    bm25_context = searcher_from_target_doc.search(query, threshold=bm25_threshold)
    print("BM25检索:\n\n", bm25_context)

    # 合并两种检索结果
    merged_res = bge_context + bm25_context

    # 检查知识库中文档块数量
    if len(merged_res) <= 1:
        # 如果只有一个文档块，直接返回该文档块
        return merged_res

    # 去重
    unique_res = list({doc.page_content: doc for doc in merged_res}.values())
    # print(unique_res)

    # 检查去重后的文档块数量
    if len(unique_res) == 1:
        # 如果去重后只有一个文档块，直接返回该文档块
        return unique_res

    # 创建DocumentReranker实例
    reranker = DocumentReranker(reranker_model)

    # 对去重后的检索结果进行重排
    top_documents = reranker.rerank_documents(query, unique_res, top_n=3)
    
    # 对重排后的结果进行去重
    unique_top_documents = list({doc.page_content: doc for doc in top_documents}.values())
    print(unique_top_documents)
    return unique_top_documents

#使用 OpenAI API 来生成引导性问题。这个函数将遍历知识库中的每个文档，并生成一个与文档内容相关的引导性问题。
def generate_guiding_questions(num_questions_per_doc = 2):
    openai_api_key = "EMPTY"
    openai_api_base = "http://region-9.autodl.pro:23971/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    
    doc_groups = {}
    for doc_id, doc in kb_vectordb.docstore._dict.items():
        source = doc.metadata.get('source', 'unknown')
        if source not in doc_groups:
            doc_groups[source] = []
        doc_groups[source].append(doc)
    
    guiding_questions = []
    
    # 遍历每个文档组并生成引导性问题
    for source, docs in doc_groups.items():
        # 随机抽取部分内容生成问题
        selected_docs = random.sample(docs, min(num_questions_per_doc, len(docs)))
        
        for doc in selected_docs:
            prompt = f"根据以下内容生成一个引导性问题：\n内容: {doc.page_content}\n，引导性问题内容要确保有主体，内容不能太复杂，问题要有引导意义，引导用户提问。只能生成一个问题不能有子问题，一步步思考，接下来根据上述的内容和要求生成引导性问题:"
            result = client.chat.completions.create(
                temperature=0.5,
                model='Qwen1.5-32B-Chat',
                stream=False,
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
            
            guiding_questions.append({
                "guiding_question": result.strip()
            })
    
    return guiding_questions


def get_document_snippets(documents, max_length=800):
    snippets = []
    for doc in documents:
        content = doc.page_content
        if len(content) >= max_length:
            snippet = content[:max_length]
        else:
            snippet = content
        snippets.append(snippet)
    return "\n".join(snippets)

def document_question_relevance(question, documents):
    openai_api_key = "EMPTY"
    openai_api_base = "http://region-9.autodl.pro:23971/v1"
    # LLM
    llm = ChatOpenAI(api_key=openai_api_key,
                base_url=openai_api_base,
                model='Qwen1.5-32B-Chat',
                temperature=0.2,
                streaming=True)

    prompt = PromptTemplate(
        template="""你是一个评分员，评估检索到的文档与用户问题的相关性。\ n
                    这里是检索到的文档:\n\n {document} \n\n
                    这是用户的问题:{question} \n
                    如果文档包含与用户问题相关的关键字，则将其评为相关。\ n
                    它不需要是一个严格的测试。目标是过滤掉错误的检索。\ n
                    给出一个二元分数“是”或“否”，以表明该文档是否与问题相关。\ n
                    以JSON的形式提供二进制分数，其中只有一个关键字‘score’，不需要任何开头或解释。""",
        input_variables=["question", "document"],
)
    document = get_document_snippets(documents)
    retrieval_grader = prompt | llm | JsonOutputParser()
    
    result = retrieval_grader.invoke({"question": question, "document": document})
    return result['score']
    
def is_math_question(question):
    openai_api_key = "EMPTY"
    openai_api_base = "http://region-9.autodl.pro:23971/v1"
    
    llm = ChatOpenAI(api_key=openai_api_key,
                base_url=openai_api_base,
                model='Qwen1.5-32B-Chat',
                temperature=0.2,
                streaming=True)
    prompt = PromptTemplate(
        template="""你是一个问题分类员，评估一个问题是否需要通过计算过程才能回答是否有一系列事实依据。
                    以下是问题:
                    \n ------- 
                    {question}
                    \n ------
                    给出一个二元分数“是”或“否”，以表明答案是否基于支持/问题是否需要通过计算过程才能回答。
                    以JSON形式提供，只有一个关键字‘score’，没有序言或解释。
                """,
        input_variables=["question"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    response = hallucination_grader.invoke({"question": question})
    return response["score"]

#历史记录
def history_list_to_str(history):
    result = ""
    for item in history:
        result += str(item) + "\n"
    return result + "\n这是以上我和你的对话记录，请参考\n" 

#知识库问答
def run_llm_MulitDocQA(query: str,only_chatKBQA: bool, prompt_template_from_user:str,temperature:float,multiple_dialogue:bool):
    openai_api_key = "EMPTY"
    openai_api_base = "http://region-9.autodl.pro:23971/v1"
    
    llm = ChatOpenAI(api_key=openai_api_key,
                base_url=openai_api_base,
                model='Qwen1.5-32B-Chat',
                temperature=temperature,
                streaming=True)

    # use_history = True  # 设置是否使用历史记录的标志
     # 检索文档
    
    #一开始先判断是否是知识库问答：
    #获取历史计记录
    history_str = history_list_to_str(history)
    if only_chatKBQA:
        #一开始应该先判断问题是否是关于知识库问答的，如果不是，就进行常规问答
        # if is_nomal_question(query) == '是':
        top_documents = get_top_documents(query)
        # print(top_documents)
        #判断检索到的文档是否和问题相关
        if document_question_relevance(query,top_documents) == '是':
            print("文档和问题相关")
            # #如果检索内容和问题相关，判断问题是否为计算类问题
            # # if is_math_question(query) == '是':
            #     # print('问题是计算类问题')
            #     # math_prompt_template = """用户问题和计费统计以及需要结合文档知识计算有关": 
            #                             Role: 计算专家
            #                             Profile:
            #                             Language: 中文
            #                             Description: 你是一位文档分析推理和费用计算方面的专家,精通各种数学算法和逻辑推理。你能够理解文档中描述的算法业务逻辑,并根据用户输入的参数进行计算,给出准确的答案。
            #                             Skill:
            #                             精通数学推理和逻辑分析
            #                             熟悉各种数学算法和计算方法
            #                             能够理解文档中描述的业务算法逻辑
            #                             根据用户输入的参数进行精确计算
            #                             用通俗易懂的语言解释计算过程和结果
            #                             Goals:
            #                             准确理解文档中描述的算法业务逻辑
            #                             提取用户输入的关键数据参数
            #                             给出计算的结果和必要的解释说明
            #                             用通俗易懂的语言与用户沟通
            #                             提供数学推理和计算方面的专业建议
            #                             Constrains:
            #                             计算过程和结果必须准确无误
            #                             不能揣测或编造算法逻辑和计算结果
            #                             不能讨论文档之外的话题
            #                             坚持使用中文与用户沟通
            #                             避免使用过于专业的术语
            #                             OutputFormat:
            #                             总结文档中涉及的算法业务逻辑
            #                             罗列需要用户输入的参数
            #                             给出计算公式或方法
            #                             输出计算结果和必要的解释
            #                             你好!我是一名文档分析推理和费用计算方面的专家,我可以帮助你理解文档中涉及的算法业务逻辑,并根据你输入的参数进行精确计算。

            #                             接下来,我会按照以下步骤与你互动:

            #                             我会仔细阅读文档,理解其中涉及的算法和业务逻辑。
            #                             然后,我会罗列出需要你输入的参数。
            #                             根据算法,我会提供计算公式或方法。
            #                             在你输入参数后,我会进行精确的计算。
            #                             最后,我会用通俗易懂的语言输出计算结果和必要的解释。
            #                             如果你对我的计算结果或解释有任何疑问,欢迎随时提出。我也可以根据需要,提供改进算法或优化计算的建议。

            #                             现在,请向我提供你手头的文档,我会开始分析其中的算法和业务逻辑。文档内容：{top_documents}，用户问题：{query}，回答："""
                # prompt = PromptTemplate(template=math_prompt_template, input_variables=["query", "top_documents"])
                # rag_chain = (
                # {"top_documents": lambda x: top_documents , "query": RunnablePassthrough()}
                # | prompt
                # | llm
                # | StrOutputParser()
                #                 )
                #     # 流式输出结果
                # response_text = ""
                # for chunk in rag_chain.stream(query):
                #     response_text += chunk
                #     yield stream_type(chunk)
                    
                # return  # 直接返回,不执行后面的代码

            if prompt_template_from_user:
                    prompt_template_with_history = prompt_template_from_user + """\n以下是历史对话记录：{history},请参考历史对话记录。下面是用户问题：{query} 回答："""
            else:
                    prompt_template_from_user = "您是一位大型语言人工智能助手。请严格根据段落内容分点作答用户问题，请极大程度保留段落的格式与内容进行简要回答。如果涉及计算请按步骤进行计算，注意不要杜撰段落没有提及的要点，且不要重复。如果文档中出现代码相关的信息，可以将完整代码返回，如果给出的段落信息与原文无关"
            if multiple_dialogue:
                print('加载了历史')
               
        
                prompt = PromptTemplate(template=prompt_template_with_history, input_variables=["query", "top_documents", "history"])
                rag_chain = (
                {"top_documents": lambda x: top_documents , "query": RunnablePassthrough(),"history": lambda x: history_str}
                | prompt
                | llm
                | StrOutputParser()
                                )
            else:
                prompt_template = prompt_template_from_user + """\n以下是相关段落：段落：{top_documents}。下面是用户问题：{query} 回答："""
                

                prompt = PromptTemplate(template=prompt_template, input_variables=["query", "top_documents"])
                rag_chain = (
                {"top_documents": lambda x: top_documents , "query": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
                            )
                
                # 流式输出结果
            response_text = ""
            for chunk in rag_chain.stream(query):
                response_text += chunk
                yield stream_type(chunk)
            # if keep_history:
                # 将当前查询和回答添加到历史记录列表中
            history.append({"query": query, "response": response_text})
            #
            # 输出图片链接信息
            image_info = find_image_links(top_documents)

            if image_info:
                    # 输出最后一条流式响应
                final_response = {
                    "id": str(uuid.uuid4()),
                    "model": "Qwen1.5-32B-Chat",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"以上是我根据知识库中的文档提供的回答。",
                                "image_list": image_info
                            },
                            "finish_reason": None
                        }
                    ]
                }

                yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n".encode('utf-8')
            else:
                final_response = {
                "id": str(uuid.uuid4()),
                "model": "Qwen1.5-32B-Chat",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "以上是我根据知识库中的文档提供的回答。",
                            "image_list": None
                        },
                        "finish_reason": None
                    }
                ]
            }

            yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n".encode('utf-8')

        else:
            final_response = {
                "id": str(uuid.uuid4()),
                "model": "Qwen1.5-32B-Chat",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "没有检索到与查询相关的上下文信息,对不起,知识库中没有找到可以回答此问题的相关信息。",
                            "image_list": None
                        },
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n".encode('utf-8')
            return []
    #如果不是只针对文档问答的话，就进行常规问答
    else:
        
        # #判断检索到的文档是否和问题相关
        # if document_question_relevance(query,top_documents) == '是':
        #     print("文档和问题相关")
        #     #如果检索内容和问题相关，判断问题是否为计算类问题
        #     if is_math_question(query) == '是':
        #         print("问题是计算类问题")
        #         math_prompt_template = """用户问题和计费统计以及需要结合文档知识计算有关": 
        #                                 Role: 计算专家
        #                                 Profile:
        #                                 Language: 中文
        #                                 Description: 你是一位文档分析推理和费用计算方面的专家,精通各种数学算法和逻辑推理。你能够理解文档中描述的算法业务逻辑,并根据用户输入的参数进行计算,给出准确的答案。
        #                                 Skill:
        #                                 精通数学推理和逻辑分析
        #                                 熟悉各种数学算法和计算方法
        #                                 能够理解文档中描述的业务算法逻辑
        #                                 根据用户输入的参数进行精确计算
        #                                 用通俗易懂的语言解释计算过程和结果
        #                                 Goals:
        #                                 准确理解文档中描述的算法业务逻辑
        #                                 提取用户输入的关键数据参数
        #                                 给出计算的结果和必要的解释说明
        #                                 用通俗易懂的语言与用户沟通
        #                                 提供数学推理和计算方面的专业建议
        #                                 Constrains:
        #                                 计算过程和结果必须准确无误
        #                                 不能揣测或编造算法逻辑和计算结果
        #                                 不能讨论文档之外的话题
        #                                 坚持使用中文与用户沟通
        #                                 避免使用过于专业的术语
        #                                 OutputFormat:
        #                                 总结文档中涉及的算法业务逻辑
        #                                 罗列需要用户输入的参数
        #                                 给出计算公式或方法
        #                                 输出计算结果和必要的解释
        #                                 你好!我是一名文档分析推理和费用计算方面的专家,我可以帮助你理解文档中涉及的算法业务逻辑,并根据你输入的参数进行精确计算。

        #                                 接下来,我会按照以下步骤与你互动:

        #                                 我会仔细阅读文档,理解其中涉及的算法和业务逻辑。
        #                                 然后,我会罗列出需要你输入的参数。
        #                                 根据算法,我会提供计算公式或方法。
        #                                 在你输入参数后,我会进行精确的计算。
        #                                 最后,我会用通俗易懂的语言输出计算结果和必要的解释。
        #                                 如果你对我的计算结果或解释有任何疑问,欢迎随时提出。我也可以根据需要,提供改进算法或优化计算的建议。

        #                                 现在,请向我提供你手头的文档,我会开始分析其中的算法和业务逻辑。，文档内容：{top_documents}，用户问题：{query}，回答："""
        #         prompt = PromptTemplate(template=math_prompt_template, input_variables=["query", "top_documents"])
        #         rag_chain = (
        #         {"top_documents": lambda x: top_documents , "query": RunnablePassthrough()}
        #         | prompt
        #         | llm
        #         | StrOutputParser()
        #                         )
        #             # 流式输出结果
        #         response_text = ""
        #         for chunk in rag_chain.stream(query):
        #             response_text += chunk
        #             yield stream_type(chunk)
                    
        #         return  # 直接返回,不执行后面的代码

        #     elif use_history:

        #         prompt_template_with_history = """您是一位大型语言人工智能助手。请严格根据段落内容分点作答用户问题，请极大程度保留段落的格式与内容进行简要回答。如果涉及计算请按步骤进行计算，注意不要杜撰段落没有提及的要点，且不要重复。如果给出的段落信息与原文无关,# 请在相关主题后写上"信息缺失：#"以下是相关段落：段落：{top_documents}。请记住,不要一字不差的重复上下文内容。回答必须使用简体中文,回答的内容一定要清晰，分段落总结。以下是历史对话记录：{history},请参考历史对话记录。下面是用户问题：{query} 回答："""
                
        #         print()
        #         prompt = PromptTemplate(template=prompt_template_with_history, input_variables=["query", "top_documents", "history"])
        #         rag_chain = (
        #         {"top_documents": lambda x: top_documents , "query": RunnablePassthrough(),"history": lambda x: history_str}
        #         | prompt
        #         | llm
        #         | StrOutputParser()
        #                         )
        #     else:
        #         prompt_template = """您是一位大型语言人工智能助手。请严格根据段落内容分点作答用户问题，请极大程度保留段落的格式与内容进行简要回答。如果涉及计算请按步骤进行计算，注意不要杜撰段落没有提及的要点，且不要重复。如果给出的段落信息与原文无关,# 请在相关主题后写上"信息缺失：#"以下是相关段落：段落：{top_documents}。请记住,不要一字不差的重复上下文内容。回答必须使用简体中文,回答的内容一定要清晰，分段落总结。下面是用户问题：{query} 回答："""
                

        #         prompt = PromptTemplate(template=prompt_template, input_variables=["query", "top_documents"])
        #         rag_chain = (
        #         {"top_documents": lambda x: top_documents , "query": RunnablePassthrough()}
        #         | prompt
        #         | llm
        #         | StrOutputParser()
        #                     )
                
        #         # 流式输出结果
        #     response_text = ""
        #     for chunk in rag_chain.stream(query):
        #         response_text += chunk
        #         yield stream_type(chunk)
        #     if keep_history:
        #         # 将当前查询和回答添加到历史记录列表中
        #         history.append({"query": query, "response": response_text})

        #     # 输出图片链接信息
        #     image_info = find_image_links(top_documents)

        #     if image_info:
        #             # 输出最后一条流式响应
        #         final_response = {
        #             "id": str(uuid.uuid4()),
        #             "model": "Qwen1.5-32B-Chat",
        #             "choices": [
        #                 {
        #                     "index": 0,
        #                     "delta": {
        #                         "content": f"以上是我根据知识库中的文档提供的回答。",
        #                         "image_list": image_info
        #                     },
        #                     "finish_reason": None
        #                 }
        #             ]
        #         }

        #         yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n".encode('utf-8')
        #     else:
        #         final_response = {
        #         "id": str(uuid.uuid4()),
        #         "model": "Qwen1.5-32B-Chat",
        #         "choices": [
        #             {
        #                 "index": 0,
        #                 "delta": {
        #                     "content": "以上是我根据知识库中的文档提供的回答。",
        #                     "image_list": None
        #                 },
        #                 "finish_reason": None
        #             }
        #         ]
        #     }

        #         yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n".encode('utf-8')

        # else:
            
        history_str = history_list_to_str(history)
        if multiple_dialogue:
            prompt_template = """以上是历史信息{history_str}，您是一位大型语言人工智能助手。热情有礼貌的和用户进行交互，根据用户的要求或提问{query},及时给用户满意的反馈和回答。回复："""
            prompt = PromptTemplate(template=prompt_template, input_variables=["history_str","query"])
            rag_chain = (
            {"history_str": lambda x: history_str, "query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
                        )
            response_text = ""
            for chunk in rag_chain.stream(query):
                response_text += chunk
                yield stream_type(chunk)
            # if keep_history:
                # 将当前查询和回答添加到历史记录列表中
            history.append({"query": query, "response": response_text})
        else:
            prompt_template = """您是一位大型语言人工智能助手。热情有礼貌的和用户进行交互，根据用户的要求或提问{query},及时给用户满意的反馈和回答。回复："""
            prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
            rag_chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
                        )
            response_text = ""
            for chunk in rag_chain.stream(query):
                response_text += chunk
                yield stream_type(chunk)
            # if keep_history:
                # 将当前查询和回答添加到历史记录列表中
            history.append({"query": query, "response": response_text})
            #

#查看历史
def view_history(history):
    history_str = ""
    total_tokens = 0
    enc = tiktoken.get_encoding("cl100k_base")  # 使用 cl100k_base 编码器
    for item in history:
        query = item["query"]
        response = item["response"]
        history_str += f"User: {query}\n"
        history_str += f"Assistant: {response}\n\n"
        total_tokens += len(enc.encode(query)) + len(enc.encode(response))
    return history_str, total_tokens

#上传文件
def get_uploaded_files():
    uploaded_files = {os.path.basename(doc.metadata.get('file_path', ''))
                      for doc in kb_vectordb.docstore._dict.values()}
    return uploaded_files



KB_DIR = "/root/autodl-tmp/project_/KG-LLM-Doc/Document_test/Doc_QA/Knowledge_based"

def load_vectordb_and_files():
    global kb_vectordb, uploaded_files, files_vectordb

    kb_list = os.listdir(KB_DIR)
    if kb_list:
        default_kb_name = kb_list[0]
        kb = KnowledgeBase(default_kb_name, embeddings)
        kb_vectordb = kb.load_vectordb()
    else:
        default_kb_name = "test"
        kb = KnowledgeBase(default_kb_name, embeddings)
        

    try:
        uploaded_files = set()
        for doc_id, doc in kb_vectordb.docstore._dict.items():
            source = os.path.basename(doc.metadata.get('file_path'))
            if source:  # 确保 source 不为空
                uploaded_files.add(source)
        print(f"知识库 '{default_kb_name}' 的向量数据库和已上传文件加载成功")
    except Exception as e:
        print(f"加载知识库 '{default_kb_name}' 的向量数据库和已上传文件时出错: {str(e)}")
        kb_vectordb = None
        uploaded_files = set()

    files_vectordb = kb_vectordb




from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

import os
UPLOAD_DIRECTORY = "uploads" 
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)



# app = FastAPI(root_path="/chaGLM_api")
app = FastAPI()


# 全局变量
kb_vectordb = None
history = []
searcher_from_target_doc = None
unfilter_context = []
current_kb_name = None  # 当前知识库名称
#开局默认加载第一个知识库
@app.on_event("startup")
async def startup_event():
    load_vectordb_and_files()

#一开始启动的时候先判定是存在知识库，如果存在就不管，如果不纯在

#查看知识库当前已有的知识库
@app.get("/view_kb")
async def view_kb_api():
    KB_dir = "/root/autodl-tmp/project_/KG-LLM-Doc/Document_test/Doc_QA/Knowledge_based"
    try:
        KB_list = os.listdir(KB_dir)
        return {"Knowledge_base_list": KB_list}
    except Exception as e:
        error_message = str(e)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/delete_kb")
async def delete_kb_api(kb_name: str = Form(...)):
    kb_dir = os.path.join(KB_DIR, kb_name)

    if os.path.exists(kb_dir):
        try:
            shutil.rmtree(kb_dir)
            logger.info(f"Knowledge base '{kb_name}' deleted successfully")
            return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' deleted successfully"})
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while deleting knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
    else:
        logger.warning(f"Knowledge base '{kb_name}' does not exist")
        return JSONResponse(status_code=500, content={"code":500, "message": f"Knowledge base '{kb_name}' does not exist"})

#选择知识库
@app.post("/select_kb")
async def select_kb_api(kb_name: str = Form(...)):
    global kb, kb_vectordb, history, searcher_from_target_doc, unfilter_context, current_kb_name

    # 检查是否需要重新加载知识库
    if current_kb_name == kb_name:
        logger.info(f"Knowledge base '{kb_name}' is already loaded")
        return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' is already loaded"})

    kb_dir = os.path.join(KB_DIR, kb_name)

    if os.path.exists(kb_dir):
        try:
            kb = KnowledgeBase(kb_name, embeddings)
            kb_vectordb = kb.load_vectordb()
            history = []
            unfilter_context = [doc for doc_id, doc in kb_vectordb.docstore._dict.items()]
            searcher_from_target_doc = BM25Search(unfilter_context)
            current_kb_name = kb_name  # 更新当前知识库名称
            logger.info(f"Knowledge base '{kb_name}' selected successfully")
            return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' selected successfully"})
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while selecting knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
    else:
        try:
            os.makedirs(kb_dir, exist_ok=True)
            kb = KnowledgeBase(kb_name, embeddings)
            kb_vectordb = None
            history = []
            unfilter_context = []
            searcher_from_target_doc = None
            current_kb_name = kb_name  # 更新当前知识库名称
            logger.info(f"Knowledge base '{kb_name}' created successfully")
            return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' created successfully"})
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while creating knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

#上传文件
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from typing import List
import os
import shutil



@app.post("/update_vectordb")
async def update_vectordb_api(kb_name: str = Form(...), files: List[UploadFile] = File(...)):
    global kb_vectordb, history, searcher_from_target_doc, unfilter_context, current_kb_name

    kb_dir = os.path.join(KB_DIR, kb_name)
    upload_directory = os.path.join(kb_dir, "uploads")
    os.makedirs(upload_directory, exist_ok=True)

    kb = None  # 初始化知识库对象

    # 检查知识库是否存在
    if not os.path.exists(kb_dir):
        try:
            os.makedirs(kb_dir, exist_ok=True)
            kb = KnowledgeBase(kb_name, embeddings)
            current_kb_name = kb_name  # 更新当前知识库名称
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while creating knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
    else:
        try:
            kb = KnowledgeBase(kb_name, embeddings)
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error occurred while loading knowledge base '{kb_name}': {error_message}")
            return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

    # 保存上传的文件
    try:
        for file in files:
            file_path = os.path.join(upload_directory, file.filename)
            async with aiofiles.open(file_path, "wb") as buffer:
                await buffer.write(await file.read())
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error occurred while saving files for knowledge base '{kb_name}': {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

    try:
        files = [os.path.join(upload_directory, filename) for filename in os.listdir(upload_directory)]
        # 更新向量库
        result = kb.update_vectordb(files)
        # 重新加载向量库
        kb_vectordb = kb.load_vectordb()
        # 清空历史记录
        history = []
        # 更新未过滤的上下文
        unfilter_context = [doc for doc_id, doc in kb_vectordb.docstore._dict.items()]
        # 创建新的搜索器
        searcher_from_target_doc = BM25Search(unfilter_context)

        # 清空上传目录
        shutil.rmtree(upload_directory)
        os.makedirs(upload_directory)

        logger.info(f"Knowledge base '{kb_name}' updated successfully")
        return JSONResponse(status_code=200, content={"code": 200, "message": f"Knowledge base '{kb_name}' updated successfully", "result": result})
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error occurred while updating knowledge base '{kb_name}': {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})


#生成引导性问题
@app.get("/view_guiding_questions")
async def view_guiding_questions_api():
    try:
        guiding_questions = generate_guiding_questions()
        logger.info("Guiding questions generated successfully")
        return JSONResponse(status_code=200, content={"code": 200, "guiding_questions": guiding_questions})
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error occurred while generating guiding questions: {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
#查看历史
@app.get("/view_history")
async def view_history_api():
    try:
        history_str, total_tokens = view_history(history)
        logger.info("History viewed successfully")
        return JSONResponse(status_code=200, content={"code": 200, "history": history_str, "total_tokens": total_tokens})
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error occurred while viewing history: {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})

#清空历史
@app.post("/clear_history")
async def clear_history_api():
    try:
        # 清空历史记录
        history.clear()
        logger.info("History cleared successfully")
        return JSONResponse(status_code=200, content={"code": 200, "message": "History cleared successfully"})
    except Exception as e:
        # 捕获清空历史记录过程中的任何异常
        error_message = str(e)
        logger.error(f"Error occurred while clearing history: {error_message}")
        return JSONResponse(status_code=500, content={"code": 500, "message": error_message})
    

#删除文件
@app.post("/remove_file")
async def remove_file_api(kb_name: str = Form(...), file_name: str = Form(...)):
    kb_dir = os.path.join(KB_DIR, kb_name)
    
    if not os.path.exists(kb_dir):
        logger.error(f"Knowledge base '{kb_name}' does not exist")
        raise HTTPException(status_code=500, conten={"code": 500,"message": f"Knowledge base '{kb_name}' does not exist"})
    
    try:
        kb = KnowledgeBase(kb_name, KB_DIR)
        result = kb.remove_file(file_name)
        logger.info(f"File '{file_name}' removed successfully from knowledge base '{kb_name}'")
        return JSONResponse(status_code=200, content={"code": 200, "message": result})
    except HTTPException as e:
        logger.error(f"HTTP exception occurred: {e.detail}")
        raise e
    except Exception as e:
        error_message = str(e)
        logger.error(f"An unexpected error occurred: {error_message}")
        raise HTTPException(status_code=500,content={"code": 500, "message": "An unexpected error occurred"})
    
#查看知识库
@app.get("/view_files")
async def view_files_api():
    try:
        files = kb.view_files()
        logger.info("Files retrieved successfully")
        return JSONResponse(code=200, content={"code": 200, "message": "Files retrieved successfully", "data": files})
    except HTTPException as e:
        logger.error(f"HTTP exception occurred: {e.detail}")
        raise e
    except Exception as e:
        error_message = str(e)
        logger.error(f"An unexpected error occurred: {error_message}")
        raise HTTPException(code=500, content={"code": 500, "message": f"An unexpected error occurred: {error_message}"})



from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: str

class PromptRequest(BaseModel):
    model: str = Field(default="Qwen1.5-32B-Chat")
    messages: List[Message]
    temperature: float = Field(default=0.5)
    n: int = Field(default=1)
    stream: bool = Field(default=True)
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=4086)
    only_chatKBQA:bool = Field(default=True)
    keep_history:bool = Field(default=True)
    presence_penalty: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    repetition_penalty: float = Field(default=1.1)
    kb_name: str = Field(default="test")
    multiple_dialogue:bool = Field(default=False)


@app.post("/mulitdoc_qa")
async def run_llm_mulitdoc_qa_api(request: Request):
    try:
        # 解析请求体中的JSON数据
        request_data = await request.json()
        prompt_request = PromptRequest(**request_data)

        # 访问 prompt_request 的字段
        messages = prompt_request.messages
        temperature = prompt_request.temperature
        prompt_template_from_user = messages[0].content
        query = messages[1].content
        only_chatKBQA = prompt_request.only_chatKBQA
        multiple_dialogue = prompt_request.multiple_dialogue

        # 创建一个生成器，用于流式返回结果
        result_generator = run_llm_MulitDocQA(query, only_chatKBQA, prompt_template_from_user, temperature, multiple_dialogue)

        # 定义一个异步生成器，将结果一起返回
        async def output_generator():
            yield stream_type(None)  # 发送初始的空数据
            for chunk in result_generator:
                yield chunk
                await asyncio.sleep(0)  # 确保其他协程有机会运行
            yield "data: [DONE]\n\n"  # 发送最后的空数据，表示数据传输结束

        logger.info("Request processed successfully")
        # 使用 StreamingResponse 返回流式输出
        return StreamingResponse(output_generator(), media_type="text/event-stream")

    except Exception as e:
        # 捕获运行过程中的任何异常
        error_message = str(e)
        logger.error(f"An unexpected error occurred: {error_message}")
        raise HTTPException(code=500, conten={"code": 500, "message":f"An unexpected error occurred: {error_message}"})

    



@app.get("/display_image")
async def display_image(image_url: str):
    try:
        # 从图片 URL 中提取图片文件名
        image_name = image_url

        # 构建图片文件的完整路径
        image_path = f"/root/autodl-tmp/project_/KG-LLM-Doc/Document_test/Doc_QA/{image_name}"

        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            logger.error(f"Image '{image_name}' does not exist at path '{image_path}'")
            raise HTTPException(code=500, conten={"code": 500, "message": f"Image '{image_name}' does not exist at path '{image_path}'"})

        logger.info(f"Image '{image_name}' retrieved successfully from path '{image_path}'")
        # 返回图片数据
        return FileResponse(image_path)
    except HTTPException as e:
        logger.error(f"HTTP exception occurred: {e.detail}")
        raise e
    except Exception as e:
        error_message = str(e)
        logger.error(f"An unexpected error occurred: {error_message}")
        raise HTTPException(code=500, conten={"code": 500, "message": f"An unexpected error occurred: {error_message}"})



#知识库文件问答
class QueryRequest(BaseModel):
    model: str = Field(default="Qwen1.5-32B-Chat")
    messages: List[Message]
    temperature: float = Field(default=0.5)
    n: int = Field(default=1)
    stream: bool = Field(default=True)
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=4086)
    # only_chatKBQA:bool = Field(default=True)
    keep_history:bool = Field(default=True)
    presence_penalty: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    repetition_penalty: float = Field(default=1.1)


@app.post("/Knowlege_baes_file_QA")
async def run_llm_Knowlege_baes_file_QA_api(request: Request):
    try:
        # 解析请求体中的JSON数据
        request_data = await request.json()
        prompt_request = PromptRequest(**request_data)
        # 访问 prompt_request 的字段
        messages = prompt_request.messages
        temperature = prompt_request.temperature
        query = messages[0].content
        only_chatKBQA = prompt_request.only_chatKBQA
        keep_history = prompt_request.keep_history

        # # 获取上传的文件列表
        # uploaded_files = get_uploaded_files()

        # 创建一个生成器,用于流式返回结果
        result_generator = run_llm_Knowlege_baes_file_QA(query,keep_history)

        # 定义一个异步生成器,使用 stream_type 函数格式化结果
        async def output_generator():
            yield stream_type(None)  # 发送初始的空数据
            # yield stream_type(uploaded_files)
            for chunk in result_generator:
                yield stream_type(chunk)
                await asyncio.sleep(0)  # 确保其他协程有机会运行
            # 发送最后的空数据,表示数据传输结束
            yield "data: [DONE]\n\n"
        # 使用 StreamingResponse 返回流式输出
        return StreamingResponse(output_generator(), media_type="text/event-stream")

    except Exception as e:
        # 捕获运行过程中的任何异常
        error_message = str(e)
        raise HTTPException(code=500, content={"code": 500, "message": f"An unexpected error occurred: {error_message}"})


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="知识库问答应用",
        version="1.0.0",
        description="一个基于知识库的问答应用",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=6006, workers=1)