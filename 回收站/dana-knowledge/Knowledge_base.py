from documen_processing import process_doc_file, process_md_file, process_txt_file, process_pdf
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from bm25_search import BM25Search
from langchain_community.vectorstores import FAISS
import shutil
import os
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import gc

class KnowledgeBase:
    def __init__(self, kb_name,embeddings):
        self.kb_name = kb_name
        self.base_directory = "/root/autodl-tmp/project_/KG-LLM-Doc/Document_test/Doc_QA/Knowledge_based"
        self.doc_directory = os.path.join(self.base_directory, kb_name, "doc_directory")
        self.markdown_directory = os.path.join(self.base_directory, kb_name, "markdown_directory")
        self.image_directory = os.path.join(self.base_directory, kb_name, "image_output")
        # self.media_dir = os.path.join(self.base_directory, kb_name, "media_dir")
        self.embeddings = embeddings  # 将传入的编码模型赋值给实例变量
        # self.reranker = reranker  # 将传入的重排模型赋值给实例变量
        
        # 创建知识库目录
        os.makedirs(self.doc_directory, exist_ok=True)
        os.makedirs(self.markdown_directory, exist_ok=True)
        os.makedirs(self.image_directory, exist_ok=True)
        # os.makedirs(self.media_dir, exist_ok=True)
        
        self.vectordb = None
        self.files_vectordb = None
        self.bm25_searcher = None
        self.history = []
        self.uploaded_files = set()
        self.Sreach_load_flag = False
    
    #加载向量库
    def load_vectordb(self):
        if self.vectordb is None:
            faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
            if os.path.exists(faiss_index_path):
                # 如果向量库文件存在,尝试加载现有的向量库
                try:
                    self.vectordb = FAISS.load_local(faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
                except Exception as e:
                    print(f"加载向量库失败: {str(e)}")
                    self.vectordb = None
        return self.vectordb
    
    #加载编码模型
    def load_embeddings(self, model_name, model_kwargs, encode_kwargs):
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    
    #解析文档
    def process_files(self, files):
        start = time.time()

        doc_files = [file for file in files if '.docx' in file]
        pdf_files = [file for file in files if '.pdf' in file]
        markdown_files = [file for file in files if '.md' in file]
        txt_files = [file for file in files if '.txt' in file]

        all_md_header_splits = []

        # Process DOCX files
        if doc_files:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_doc_file, doc_file, self.image_directory, self.markdown_directory) for doc_file in doc_files]
                for future in tqdm(futures, desc="Processing DOCX files"):
                    md_header_splits = future.result()
                    all_md_header_splits.extend(md_header_splits)

        # Process Markdown files
        if markdown_files:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_md_file, md_file) for md_file in markdown_files]
                for future in tqdm(futures, desc="Processing Markdown files"):
                    md_header_splits = future.result()
                    all_md_header_splits.extend(md_header_splits)

        # Process TXT files
        if txt_files:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_txt_file, txt_file,self.markdown_directory) for txt_file in txt_files]
                for future in tqdm(futures, desc="Processing TXT files"):
                    md_header_splits = future.result()
                    all_md_header_splits.extend(md_header_splits)

        if pdf_files:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_pdf, pdf_path, self.image_directory, self.markdown_directory) for pdf_path in pdf_files]
                for future in tqdm(futures, desc="Processing PDF files"):
                    md_header_splits = future.result()
                    all_md_header_splits.extend(md_header_splits)
        end = time.time()
        print(f"解析文档总共耗时: {end - start:.2f} 秒")
        return all_md_header_splits

    #向量化文档
    def vectorize_documents(self, documents):
        start = time.time()

        vectordb = FAISS.from_documents(documents, self.embeddings)

        end = time.time()
        print(f"向量化文档总共耗时: {end - start:.2f} 秒")

        return vectordb
    #保持向量库到本地
    def save_vectordb(self, vectordb):
        faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
        vectordb.save_local(faiss_index_path)

    #获取向量库
    def get_faiss_vectordb(self, files):
        documents = self.process_files(files)
        vectordb = self.vectorize_documents(documents)
        return vectordb

    #加载向量库
    def load_vectordb_and_files(self):
        try:
            faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
            self.vectordb = FAISS.load_local(faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)

            for doc_id, doc in self.vectordb.docstore._dict.items():
                source = os.path.basename(doc.metadata.get('file_path'))
                if source:  # 确保 source 不为空
                    self.uploaded_files.add(source)

            print(f"知识库 {self.kb_name} 的向量数据库和已上传文件加载成功")
        except Exception as e:
            print(f"加载知识库 {self.kb_name} 的向量数据库和已上传文件时出错: {str(e)}")

    #增添文件，更新向量库
    def update_vectordb(self, files):
        new_files = files  # 将所有上传的文件视为新文件
        new_files_names = [os.path.basename(file) for file in new_files]
        self.vectordb = self.load_vectordb()  # 确保在添加文件之前加载或创建向量库
        if not new_files:
            return "没有选择任何文件"

        if self.vectordb is None:
            self.vectordb = self.get_faiss_vectordb(new_files)
        else:
            # 找出需要删除的文件路经
            files_to_delete = [file for file in new_files if file in self.uploaded_files]
            
            # 删除原有知识库中已存在的文件
            for file_name in files_to_delete:
                try:
                    # 删除 faiss_index 中与文件相关的向量内容
                    doc_ids = [doc_id for doc_id, doc in self.vectordb.docstore._dict.items() if os.path.basename(doc.metadata.get('file_path', '')) == file_name]
                    if doc_ids:
                        self.vectordb.delete(ids=doc_ids)
                    
                    # 从已上传文件集合中移除文件名
                    self.uploaded_files.remove(file_name)
                    
                    self.log_operation("删除文件", [file_name])  # 记录删除文件操作
                except Exception as e:
                    print(f"删除文件 {file_name} 时出错: {str(e)}")
            
            # 将新文件添加到知识库中
            new_vectordb = self.get_faiss_vectordb(new_files)
            self.vectordb.merge_from(new_vectordb)
        
        faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
        self.vectordb.save_local(faiss_index_path)
        
        self.uploaded_files.update(new_files_names)
        gc.collect()  # Clear unused memory
        return f"已更新 {len(new_files)} 个文件{new_files_names}到知识库 {self.kb_name} 的向量数据库"

    #删除文件
    def remove_file(self, file_name):
        self.load_vectordb_and_files()
        print(self.uploaded_files)
        if file_name not in self.uploaded_files:
            # print(f"文件 {file_name} 不在知识库 {self.kb_name} 中")
            return f"文件 {file_name} 不在知识库 {self.kb_name} 中"
        
        try:
            print(f"开始删除文件 {file_name}...")
            
            # 删除 faiss_index 中与文件相关的向量内容
            doc_ids = [doc_id for doc_id, doc in self.vectordb.docstore._dict.items() if os.path.basename(doc.metadata.get('file_path', '')) == file_name]
            if doc_ids:
                print(f"正在从 faiss_index 中删除与文件 {file_name} 相关的向量...")
                self.vectordb.delete(ids=doc_ids)
                faiss_index_path = os.path.join(self.base_directory, self.kb_name, "faiss_index")
                self.vectordb.save_local(faiss_index_path)  # 保存更新后的 faiss_index
                print(f"已从 faiss_index 中删除与文件 {file_name} 相关的向量")
            
            # 从已上传文件集合中移除文件名
            self.uploaded_files.remove(file_name)
            print(f"已从已上传文件集合中移除文件 {file_name}")
            
            # self.log_operation("删除文件", [file_name])  # 记录删除文件操作
            #删除 markdown_directory 中对应的 Markdown 文件
            markdown_file = os.path.join(self.markdown_directory, os.path.splitext(file_name)[0] + ".md")
            if os.path.exists(markdown_file):
                print(f"正在删除 Markdown 文件 {markdown_file}...")
                os.remove(markdown_file)
                print(f"已删除 Markdown 文件 {markdown_file}")
        
            # 删除对应的图像文件夹
            image_folder = os.path.join(self.image_directory, os.path.splitext(file_name)[0])
            if os.path.exists(image_folder):
                print(f"正在删除图像文件夹 {image_folder}...")
                shutil.rmtree(image_folder)
                print(f"已删除图像文件夹 {image_folder}")
        
            print(f"文件 {file_name} 删除完成")
            return f"已从知识库 {self.kb_name} 中删除文件 {file_name} 及其相关向量"
        except Exception as e:
            print(f"删除文件 {file_name} 时出错: {str(e)}")
            return f"删除文件 {file_name} 时出错: {str(e)}"

    #查看知识库中的文档
    def view_files(self):
        # 在查看文件前重新加载向量数据库,确保获取最新的文件列表
        self.load_vectordb_and_files()
        
        return f"知识库 {self.kb_name} 中的文件:\n" + "\n".join(self.uploaded_files)
    
  