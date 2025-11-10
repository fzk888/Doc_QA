import requests
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document

url = 'http://0.0.0.0:8503/pix2text'

data = {
    "file_type": "page",
    "resized_shape": 768,
    "embed_sep": " $,$ ",
    "isolated_sep": "$$\n, \n$$"
}

def process_pic_file(jpg_file, markdown_directory):
    md_header_splits = []
    files = {"image": (jpg_file, open(jpg_file, 'rb'), 'image/jpeg')}
    r = requests.post(url, data=data, files=files)
    output_path = os.path.join(markdown_directory, os.path.splitext(os.path.basename(jpg_file))[0] + ".md")
    outs = r.json()['results']
    out_md_dir = r.json()['output_dir']
    if isinstance(outs, str):
        only_text = outs
    else:
        only_text = '\n'.join([out['text'] for out in outs])
    
    base_name = os.path.basename(jpg_file)
    
    if len(only_text.replace(" ","").replace("\n","")) == 0:
        if os.path.exists(jpg_file):
            os.remove(jpg_file)
            print(f"文件 '{jpg_file}' 已被删除。")
        else:
            print(f"文件 '{jpg_file}' 不存在。")
        raise ValueError(f"The file {base_name} does not contain valid content.")
        
    with open(output_path, 'w') as file:
        file.write(only_text)
    loader = TextLoader(output_path)
    document = loader.load()[0]
    md_header_splits.append(Document(page_content=document.page_content, metadata={"file_path": jpg_file}))
    return md_header_splits