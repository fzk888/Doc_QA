from add.morefile.deepdoc.parser import HtmlParser
from add.morefile.rag.nlp import rag_tokenizer,tokenize

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

import os
import re


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

def process_html_file(html_file,markdown_directory):#, image_output_dir, markdown_directory):
    markdown_file = os.path.join(markdown_directory, os.path.splitext(os.path.basename(html_file))[0] + ".md")
    sections = HtmlParser()(html_file)
    sections = [s for s in sections if s]
    result = ''.join(map(str, sections))
   
    with open(markdown_file, 'w', encoding='utf-8') as file:
        file.write(result)
    loader = TextLoader(markdown_file)
    document = loader.load()[0]

    base_name = os.path.basename(html_file)
    
    if len(document.page_content.replace(" ","").replace("\n","")) == 0:
        if os.path.exists(html_file):
            os.remove(html_file)
            print(f"文件 '{html_file}' 已被删除。")
        else:
            print(f"文件 '{html_file}' 不存在。")

        raise ValueError(f"The file {base_name} does not contain valid content.")
    md_header_splits = [document]
    if len(document.page_content) > 500:
        original_content = md_header_splits[0].page_content
        split_contents = split_text_preserving_tables(original_content)
        md_header_splits = [Document(page_content=chunk, metadata={"file_path": html_file}) for chunk in split_contents]
    return md_header_splits

if __name__ == "__main__":
    process_html_file('/root/autodl-tmp/project_knowledge/Doc_QA/a.html')