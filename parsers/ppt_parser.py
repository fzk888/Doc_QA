from parsers.ppt_utils import chunk
import sys
import os
import traceback
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def dummy(a, b):
    pass

def precess_result(r,image_output_dir):
    content = ''
    for n,i in enumerate(r):
        content += i['content_with_weight']
        image_dir = image_output_dir + '/' + str(n) + '.jpg'
        content += str([image_dir])
        if 'image' in i and i['image']:
            i['image'].save(image_dir)
        content += '***ppt换页识别符号***'
    return content

def split_text_preserving_tables(text, max_token_length=3000):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        if current_length + para_length > max_token_length:
            chunks.append("\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks

def process_ppt_file(ppt_file, image_output_dir, markdown_directory):
    file_extension = os.path.splitext(ppt_file)[1].lower()
    md_header_splits = []
    base_name = os.path.basename(ppt_file)
    
    if file_extension == ".pptx":
        try:
            # 获取文档的基文件名（不带路径和扩展名）
            base_name = os.path.splitext(os.path.basename(ppt_file))[0]
            
            # 创建该文档的专属媒体目录
            media_dir = os.path.join(image_output_dir, base_name)
            os.makedirs(media_dir, exist_ok=True)
            #获取PPT解析结果
            res = chunk(ppt_file, callback=dummy)
            ppt_text = precess_result(res,media_dir)
            
            # 生成 Markdown 文件的路径
            markdown_file = os.path.join(markdown_directory, os.path.splitext(os.path.basename(ppt_file))[0] + ".md")
            
            if len(ppt_text.replace(" ","").replace("\n","")) == 0:
                if os.path.exists(ppt_file):
                    os.remove(ppt_file)
                    print(f"文件 '{ppt_file}' 已被删除。")
                else:
                    print(f"文件 '{ppt_file}' 不存在。")
                raise ValueError(f"The file {base_name} does not contain valid content.")
            
            # 将转换后的 Markdown 文本写入文件
            with open(markdown_file, 'w', encoding='utf-8') as file:
                file.write(ppt_text)
            
            # 加载 Markdown 文件
            loader = TextLoader(markdown_file)
            document = loader.load()[0]
            
            # 拆分 Markdown 内容
            text_splitter = RecursiveCharacterTextSplitter(separators=["***ppt换页识别符号***"], chunk_size=200, chunk_overlap=30)
            md_header_splits = text_splitter.create_documents([ppt_text])
            
            if len(md_header_splits) == 0:
                # 如果没有拆分,将整个文档内容作为一个拆分
                md_header_splits = [document]
            else:
                for split in md_header_splits:
                    # 获取每个分块的元数据
                    metadata = split.metadata
                    split.metadata["file_path"] = ppt_file
            
            # 如果只有一个拆分且内容长度超过500 token，按段落进行切割并重新拆分
            if len(md_header_splits) == 1 and len(md_header_splits[0].page_content.split()) > 500:
                original_content = md_header_splits[0].page_content
                split_contents = split_text_preserving_tables(original_content)
                md_header_splits = [Document(page_content=chunk, metadata={"file_path": ppt_file}) for chunk in split_contents]
        
        except Exception as e:
            error_msg = f"转换文件 {ppt_file} 时出错：{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            # 抛出异常以便上层能够捕获和处理
            raise RuntimeError(error_msg)
    
    return md_header_splits