import asyncio
from concurrent.futures import ProcessPoolExecutor
from documen_processing import process_doc_file, process_md_file, process_txt_file, process_pdf
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
from bm25_search import BM25Search
from langchain_community.vectorstores import FAISS
import shutil
import os
import time
from tqdm import tqdm
import gc
import traceback
import torch
from pathlib import Path
from add.morefile.ppt_processing import process_ppt_file
from add.morefile.html_processing import process_html_file
from add.morefile.excel_processing import process_excel_file, process_csv_file
from add.morefile.pic_processing import process_pic_file
from add.morefile.doc_processing import process_doc2_file

import yaml
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

import logging
logger = logging.getLogger("docqa.kb")

# 从配置文件中获取最大工作进程数，设置默认值为4
MAX_WORKERS = config['system'].get('max_workers', 4)

class KnowledgeBase:
    def __init__(self, kb_name, embeddings):
        self.kb_name = kb_name
        self.embeddings = embeddings
        KB_DIR = config['paths']['kb_dir']
        # 为与其他方法一致，设置 base_directory 指向配置中的知识库根目录
        self.base_directory = KB_DIR
        self.kb_dir = os.path.join(KB_DIR, kb_name)
        self.vectordb = None
        self.uploaded_files = set()
        self.image_directory = os.path.join(self.kb_dir, "images")
        self.markdown_directory = os.path.join(self.kb_dir, "markdown_directory")
        os.makedirs(self.image_directory, exist_ok=True)
        os.makedirs(self.markdown_directory, exist_ok=True)
        # 使用配置中的最大工作进程数
        self.executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)

    async def load_vectordb(self):
        if self.vectordb is None:
            faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
            if os.path.exists(faiss_index_path):
                try:
                    logger.info("正在加载 FAISS 索引...")
                    self.vectordb = await asyncio.to_thread(FAISS.load_local, faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
                    logger.info(f"Loaded vectordb for {self.kb_name}")
                except Exception as e:
                    logger.exception(f"加载向量库失败: {str(e)}")
                    self.vectordb = None
        return self.vectordb

    async def process_files(self, files):
        logger.info("正在解析并分块上传文件...")
        start = time.time()

        # 文件分类
        file_groups = {
            'docx': [], 'doc': [], 'pdf': [], 'md': [], 'txt': [],
            'pptx': [], 'html': [], 'xlsx': [], 'csv': [], 'jpg': [], 'png': []
        }

        supported_extensions = set(file_groups.keys())

        for file in files:
            # 使用 os.path.splitext() 来正确处理文件扩展名
            _, ext = os.path.splitext(file)
            ext = ext.lower().lstrip('.')
            
            # 特殊处理 .doc 和 .docx
            if ext == 'doc' or ext == 'docx':
                if file.lower().endswith('.doc'):
                    ext = 'doc'
                elif file.lower().endswith('.docx'):
                    ext = 'docx'
            
            if ext in supported_extensions:
                file_groups[ext].append(file)

        all_md_header_splits = []

        # 顺序处理文件组，避免并行处理导致的内存问题
        for file in tqdm(file_groups['docx'], desc="Processing DOCX files"):
            try:
                md_header_splits = process_doc_file(file, self.image_directory, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        for file in tqdm(file_groups['doc'], desc="Processing DOC files"):
            try:
                md_header_splits = process_doc2_file(file, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        for file in tqdm(file_groups['pdf'], desc="Processing PDF files"):
            try:
                md_header_splits = process_pdf(file, self.image_directory, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        for file in tqdm(file_groups['md'], desc="Processing Markdown files"):
            try:
                md_header_splits = process_md_file(file, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        for file in tqdm(file_groups['txt'], desc="Processing TXT files"):
            try:
                md_header_splits = process_txt_file(file, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        for file in tqdm(file_groups['pptx'], desc="Processing PPT files"):
            try:
                md_header_splits = process_ppt_file(file, self.image_directory, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        for file in tqdm(file_groups['html'], desc="Processing HTML files"):
            try:
                md_header_splits = process_html_file(file, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        for file in tqdm(file_groups['xlsx'], desc="Processing XLSX files"):
            try:
                md_header_splits = process_excel_file(file, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        for file in tqdm(file_groups['csv'], desc="Processing CSV files"):
            try:
                md_header_splits = process_csv_file(file, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        for file in tqdm(file_groups['jpg'], desc="Processing JPG files"):
            try:
                md_header_splits = process_pic_file(file, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        for file in tqdm(file_groups['png'], desc="Processing PNG files"):
            try:
                md_header_splits = process_pic_file(file, self.markdown_directory)
                all_md_header_splits.extend(md_header_splits)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue

        end = time.time()
        print(f"解析文档总共耗时: {end - start:.2f} 秒")
        return all_md_header_splits

    async def vectorize_documents(self, documents):
        logger.info("正在构建向量索引（FAISS）...")
        start = time.time()
        # 每批处理的文档数（可通过 config.yaml 的 settings.vector_batch_size 调整）
        batch_size = config['settings'].get('vector_batch_size', 100)

        vectordb = None
        batch_index = 0
        for i in range(0, len(documents), batch_size):
            batch_index += 1
            batch = documents[i:i+batch_size]
            batch_start = time.time()

            # 异步创建FAISS向量数据库
            batch_vectordb = await FAISS.afrom_documents(batch, self.embeddings)
            if vectordb is None:
                vectordb = batch_vectordb
            else:
                vectordb.merge_from(batch_vectordb)

            # 清理 GPU 缓存
            self.clean_gpu_cache()
            batch_end = time.time()
            logger.info(f"Vectorized batch {batch_index} size={len(batch)} time={(batch_end-batch_start):.2f}s")

        end = time.time()
        logger.info(f"向量化文档总共耗时: {end - start:.2f} 秒")
        return vectordb

    def clean_gpu_cache(self):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            print("GPU cache cleared")


    async def save_vectordb(self, vectordb):
        logger.info("正在保存向量索引到磁盘...")
        faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
        await asyncio.to_thread(vectordb.save_local, faiss_index_path)

    async def get_faiss_vectordb(self, files):
        documents = await self.process_files(files)
        vectordb = await self.vectorize_documents(documents)
        return vectordb


    async def load_vectordb_and_files(self):
        try:
            faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
            
            if self.embeddings is None:
                raise ValueError("self.embeddings is None. It may not have been initialized properly.")
            
            if not os.path.exists(faiss_index_path):
                raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
            
            self.vectordb = await asyncio.to_thread(FAISS.load_local, faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)

            self.uploaded_files.clear()
            for doc_id, doc in self.vectordb.docstore._dict.items():
                source = os.path.basename(doc.metadata.get('file_path', ''))
                if source:
                    self.uploaded_files.add(source)

            print(f"知识库 {self.kb_name} 的向量数据库和已上传文件加载成功")
        except Exception as e:
            print(f"加载知识库 {self.kb_name} 的向量数据库和已上传文件时出错: {str(e)}")
            print(f"错误类型: {type(e)}")
            print(f"错误堆栈: {traceback.format_exc()}")
            raise  # 重新抛出异常，以便调用者可以处理它
        
    async def update_vectordb(self, files):
        # 将传入的文件列表赋值给 new_files
        new_files = files
        # 获取新文件的文件名列表
        new_files_names = [os.path.basename(file) for file in new_files]
        # 加载现有的向量数据库
        self.vectordb = await self.load_vectordb()
        
        # 如果没有选择任何文件，返回提示信息
        if not new_files:
            return "没有选择任何文件"

        # 如果向量数据库不存在，则创建一个新的
        if self.vectordb is None:
            logger.info("正在首次构建向量库...")
            self.vectordb = await self.get_faiss_vectordb(new_files)
            # 保存新创建的向量库到磁盘，以便后续可以加载
            if self.vectordb is not None:
                try:
                    await self.save_vectordb(self.vectordb)
                except Exception as e:
                    print(f"保存向量库失败: {e}")
            # 更新已上传文件集合
            for file in new_files:
                self.uploaded_files.add(os.path.basename(file))
        else:
            # 如果向量数据库存在，加载现有的向量数据库和文件
            await self.load_vectordb_and_files()
            # 初始化一个列表，用于存储需要删除的文件
            files_to_delete = []
            # 获取文件所在的目录
            directories = os.path.dirname(files[0]) if files else ""
            
            # 检查已上传的文件，如果在新文件列表中，则添加到待删除列表
            for i in self.uploaded_files:
                if i in new_files_names:
                    files_to_delete.append(i)

            # 删除向量数据库中已存在的文件
            if files_to_delete:
                # 创建一个新的列表，包含不在待删除列表中的文档ID
                remaining_docs = [doc_id for doc_id, doc in self.vectordb.docstore._dict.items() 
                                if os.path.basename(doc.metadata.get('file_path', '')) not in files_to_delete]
                
                # 检查是否有剩余文档
                if remaining_docs:
                    # 创建一个新的向量数据库，只包含剩余的文档
                    remaining_vectordb = await asyncio.to_thread(
                        FAISS.from_documents, 
                        [self.vectordb.docstore._dict[doc_id] for doc_id in remaining_docs], 
                        self.embeddings
                    )
                    # 更新当前的向量数据库
                    self.vectordb = remaining_vectordb
                else:
                    # 如果没有剩余文档，则设置向量数据库为None
                    self.vectordb = None
                    # 清空已上传文件列表
                    self.uploaded_files.clear()

            # 处理新文件
            try:
                new_documents = await self.process_files(new_files)
                if new_documents:
                    # 如果有新文档，将它们添加到向量数据库
                    if self.vectordb is not None:
                        # 如果向量数据库存在，添加新文档
                        logger.info("正在增量向量化并合并索引...")
                        new_vectordb = await FAISS.afrom_documents(new_documents, self.embeddings)
                        self.vectordb.merge_from(new_vectordb)
                    else:
                        # 如果向量数据库不存在，创建新的
                        logger.info("正在首次构建向量库...")
                        self.vectordb = await FAISS.afrom_documents(new_documents, self.embeddings)
                
                # 更新已上传文件列表
                for file in new_files:
                    self.uploaded_files.add(os.path.basename(file))
                
                # 保存向量数据库
                if self.vectordb is not None:
                    await self.save_vectordb(self.vectordb)
                
                # 返回处理结果
                return f"已更新 {len(new_files)} 个文件{new_files_names}到知识库 {self.kb_name} 的向量数据库"
            except Exception as e:
                error_msg = f"处理文件时出错: {str(e)}"
                print(error_msg)
                raise Exception(error_msg)
    
        gc.collect()
        self.clean_gpu_cache()
        return f"已更新 {len(new_files)} 个文件{new_files_names}到知识库 {self.kb_name} 的向量数据库"
    
    async def remove_file(self, file_name):
        await self.load_vectordb_and_files()
        if file_name not in self.uploaded_files:
            return {"message": f"文件 {file_name} 不在知识库 {self.kb_name} 中"}
        
        try:
            print(f"开始删除文件 {file_name}...")
            
            doc_ids = [doc_id for doc_id, doc in self.vectordb.docstore._dict.items() if os.path.basename(doc.metadata.get('file_path', '')) == file_name]
            if doc_ids:
                print(f"正在从 faiss_index 中删除与文件 {file_name} 相关的向量...")
                await asyncio.to_thread(self.vectordb.delete, ids=doc_ids)
                faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
                await asyncio.to_thread(self.vectordb.save_local, faiss_index_path)
                print(f"已从 faiss_index 中删除与文件 {file_name} 相关的向量")
            
            self.uploaded_files.remove(file_name)
            print(f"已从已上传文件集合中移除文件 {file_name}")
            
            markdown_file = os.path.join(self.markdown_directory, os.path.splitext(file_name)[0] + ".md")
            if os.path.exists(markdown_file):
                print(f"正在删除 Markdown 文件 {markdown_file}...")
                await asyncio.to_thread(os.remove, markdown_file)
                print(f"已删除 Markdown 文件 {markdown_file}")
        
            image_folder = os.path.join(self.image_directory, os.path.splitext(file_name)[0])
            if os.path.exists(image_folder):
                print(f"正在删除图像文件夹 {image_folder}...")
                await asyncio.to_thread(shutil.rmtree, image_folder)
                print(f"已删除图像文件夹 {image_folder}")
        
            print(f"文件 {file_name} 删除完成")
            return {"message": f"已从知识库 {self.kb_name} 中删除文件 {file_name} 及其相关向量"}
        except Exception as e:
            print(f"删除文件 {file_name} 时出错: {str(e)}")
            return {"error": f"删除文件 {file_name} 时出错: {str(e)}"}
    # async def remove_file(self, file_name):
    #     await self.load_vectordb_and_files()
    #     if file_name not in self.uploaded_files:
    #         return f"文件 {file_name} 不在知识库 {self.kb_name} 中"
        
    #     try:
    #         print(f"开始删除文件 {file_name}...")
            
    #         doc_ids = [doc_id for doc_id, doc in self.vectordb.docstore._dict.items() if os.path.basename(doc.metadata.get('file_path', '')) == file_name]
    #         if doc_ids:
    #             print(f"正在从 faiss_index 中删除与文件 {file_name} 相关的向量...")
    #             await asyncio.to_thread(self.vectordb.delete, ids=doc_ids)
    #             faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
    #             await asyncio.to_thread(self.vectordb.save_local, faiss_index_path)
    #             print(f"已从 faiss_index 中删除与文件 {file_name} 相关的向量")
            
    #         self.uploaded_files.remove(file_name)
    #         print(f"已从已上传文件集合中移除文件 {file_name}")
            
    #         markdown_file = os.path.join(self.markdown_directory, os.path.splitext(file_name)[0] + ".md")
    #         if os.path.exists(markdown_file):
    #             print(f"正在删除 Markdown 文件 {markdown_file}...")
    #             os.remove(markdown_file)
    #             print(f"已删除 Markdown 文件 {markdown_file}")
        
    #         image_folder = os.path.join(self.image_directory, os.path.splitext(file_name)[0])
    #         if os.path.exists(image_folder):
    #             print(f"正在删除图像文件夹 {image_folder}...")
    #             shutil.rmtree(image_folder)
    #             print(f"已删除图像文件夹 {image_folder}")
        
    #         print(f"文件 {file_name} 删除完成")
    #         return f"已从知识库 {self.kb_name} 中删除文件 {file_name} 及其相关向量"
    #     except Exception as e:
    #         print(f"删除文件 {file_name} 时出错: {str(e)}")
    #         return f"删除文件 {file_name} 时出错: {str(e)}"

    async def view_files(self):
        await self.load_vectordb_and_files()
        return f"知识库 {self.kb_name} 中的文件:\n" + "\n".join(self.uploaded_files)
