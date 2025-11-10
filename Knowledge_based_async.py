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

class KnowledgeBase:
    def __init__(self, kb_name, embeddings):
        self.kb_name = kb_name
        self.base_directory = config['paths']['kb_dir']
        self.doc_directory = os.path.join(self.base_directory, kb_name, "doc_directory")
        self.markdown_directory = os.path.join(self.base_directory, kb_name, "markdown_directory")
        self.image_directory = os.path.join(self.base_directory, kb_name, "image_output")
        self.embeddings = embeddings
        
        os.makedirs(self.doc_directory, exist_ok=True)
        os.makedirs(self.markdown_directory, exist_ok=True)
        os.makedirs(self.image_directory, exist_ok=True)
        
        self.vectordb = None
        self.files_vectordb = None
        self.bm25_searcher = None
        self.history = []
        self.uploaded_files = set()
        self.Sreach_load_flag = False

    async def load_vectordb(self):
        if self.vectordb is None:
            faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
            if os.path.exists(faiss_index_path):
                try:
                    self.vectordb = await asyncio.to_thread(FAISS.load_local, faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
                    print(self.vectordb)
                except Exception as e:
                    print(f"加载向量库失败: {str(e)}")
                    self.vectordb = None
        return self.vectordb

    async def process_files(self, files):
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

        async def process_file_group(file_group, process_func, desc, *args):
            for file in tqdm(file_group, desc=desc):
                future = self.executor.submit(process_func, file, *args)
                md_header_splits = await asyncio.wrap_future(future)
                all_md_header_splits.extend(md_header_splits)

        with ProcessPoolExecutor(max_workers=5) as self.executor:
            await asyncio.gather(
                process_file_group(file_groups['docx'], process_doc_file, "Processing DOCX files", self.image_directory, self.markdown_directory),
                process_file_group(file_groups['doc'], process_doc2_file, "Processing DOC files", self.markdown_directory),
                process_file_group(file_groups['pdf'], process_pdf, "Processing PDF files", self.image_directory, self.markdown_directory),
                process_file_group(file_groups['md'], process_md_file, "Processing Markdown files", self.markdown_directory),
                process_file_group(file_groups['txt'], process_txt_file, "Processing TXT files", self.markdown_directory),
                process_file_group(file_groups['pptx'], process_ppt_file, "Processing PPT files", self.image_directory, self.markdown_directory),
                process_file_group(file_groups['html'], process_html_file, "Processing HTML files", self.markdown_directory),
                process_file_group(file_groups['xlsx'], process_excel_file, "Processing XLSX files", self.markdown_directory),
                process_file_group(file_groups['csv'], process_csv_file, "Processing CSV files", self.markdown_directory),
                process_file_group(file_groups['jpg'], process_pic_file, "Processing JPG files", self.markdown_directory),
                process_file_group(file_groups['png'], process_pic_file, "Processing PNG files", self.markdown_directory)
            )

        end = time.time()
        print(f"解析文档总共耗时: {end - start:.2f} 秒")
        return all_md_header_splits
    async def vectorize_documents(self, documents):
        start = time.time()

        # 每批处理的文档数
        batch_size = 100  # 可以根据需要调整这个值

        vectordb = None
        for i in tqdm(range(0, len(documents), batch_size), desc="Vectorizing documents"):
            batch = documents[i:i+batch_size]
            
            # 异步创建FAISS向量数据库
            batch_vectordb = await FAISS.afrom_documents(batch, self.embeddings)
            if vectordb is None:
                vectordb = batch_vectordb
            else:
                vectordb.merge_from(batch_vectordb)
            
            # 清理 GPU 缓存
            self.clean_gpu_cache()

        end = time.time()
        print(f"向量化文档总共耗时: {end - start:.2f} 秒")
        return vectordb

    def clean_gpu_cache(self):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            print("GPU cache cleared")


    async def save_vectordb(self, vectordb):
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
            self.vectordb = await self.get_faiss_vectordb(new_files)
        else:
            # 如果向量数据库存在，加载现有的向量数据库和文件
            await self.load_vectordb_and_files()
            # 初始化一个列表，用于存储需要删除的文件
            files_to_delete = []
            # 获取文件所在的目录
            directories = os.path.dirname(files[0])
            
            # 检查已上传的文件，如果在新文件列表中，则添加到待删除列表
            for i in self.uploaded_files:
                if i in new_files_names:
                    files_to_delete.append(os.path.join(directories, i))

            # 遍历待删除的文件
            for file_name in files_to_delete:
                doc_ids = []
                # 查找与待删除文件相关的文档ID
                for doc_id, doc in self.vectordb.docstore._dict.items():
                    file_path = doc.metadata.get('file_path', '')
                    if file_path == file_name:
                        doc_ids.append(doc_id)

                # 如果找到相关的文档ID，则从向量数据库中删除这些文档
                if doc_ids:
                    await asyncio.to_thread(self.vectordb.delete, ids=doc_ids)
                # 从已上传文件列表中移除该文件
                self.uploaded_files.remove(os.path.basename(file_name))
                
                print("删除文件", [file_name])

            # 为新文件创建一个新的向量数据库
            new_vectordb = await self.get_faiss_vectordb(new_files)
            # 将新的向量数据库合并到现有的向量数据库中
            await asyncio.to_thread(self.vectordb.merge_from, new_vectordb)
        
        # 设置 FAISS 索引的保存路径
        faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
        # 保存更新后的向量数据库到本地
        await asyncio.to_thread(self.vectordb.save_local, faiss_index_path)
    
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