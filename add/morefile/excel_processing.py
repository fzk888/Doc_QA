import pandas as pd
from add.morefile.rag.nlp import rag_tokenizer,tokenize
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document

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

def process_excel_file(file_path,markdown_directory):
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    base_name = os.path.basename(file_path)
    for sheet_name, df in all_sheets.items():
        markdown_str = df.to_markdown()
        save_path = markdown_directory + '/' + sheet_name + '.md'
        with open(save_path, 'w') as file:
            file.write(markdown_str)
        loader = TextLoader(save_path)
        document = loader.load()[0]
        if len(document.page_content.replace(" ","").replace("\n","")) == 0:
            raise ValueError(f"The file {base_name} does not contain valid content.")

        if len(document.page_content) > 500:
            original_content = document.page_content
            split_contents = split_text_preserving_tables(original_content)
            md_header_splits = [Document(page_content=chunk, metadata={"file_path": file_path}) for chunk in split_contents]
        else:
            md_header_splits.append(Document(page_content=document.page_content, metadata={"file_path": file_path}))
    return md_header_splits


def process_csv_file(file_path,markdown_directory):
    df = pd.read_csv(file_path)
    base_name = os.path.basename(file_path)
    markdown_str = df.to_markdown()
    markdown_file = os.path.join(markdown_directory, os.path.splitext(os.path.basename(file_path))[0] + ".md")
    with open(markdown_file, 'w') as file:
        file.write(markdown_str)
    loader = TextLoader(markdown_file)
    document = loader.load()[0]
    if len(document.page_content.replace(" ","").replace("\n","")) == 0:
        if os.path.exists(doc_file):
            os.remove(doc_file)
            print(f"文件 '{doc_file}' 已被删除。")
        else:
            print(f"文件 '{doc_file}' 不存在。")
        raise ValueError(f"The file {base_name} does not contain valid content.")
    
    if len(document.page_content) > 500:
        original_content = document.page_content
        split_contents = split_text_preserving_tables(original_content)
        md_header_splits = [Document(page_content=chunk, metadata={"file_path": file_path}) for chunk in split_contents]
    else:
        md_header_splits.append(Document(page_content=chunk, metadata={"file_path": file_path}))
    return md_header_splits

if __name__ == "__main__":
    process_excel_file('/root/autodl-tmp/project_knowledge/Doc_QA/知识库验证测试.xlsx')