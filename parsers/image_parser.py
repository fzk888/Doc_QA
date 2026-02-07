import requests
import logging
import yaml
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

with open("config.yaml", "r", encoding="utf-8") as _cf:
    _cfg = yaml.safe_load(_cf)
PIC_OCR_PROVIDER = (_cfg.get("settings", {}).get("pic_ocr_provider", "paddle") or "paddle").lower()
URL_OCR = _cfg.get("paths", {}).get("ocr_service_url", "http://127.0.0.1:8001/detection_pic")
URL_PIX2TEXT = _cfg.get("paths", {}).get("pix2text_url", "http://127.0.0.1:8503/pix2text")

data = {
    "file_type": "page",
    "resized_shape": 768,
    "embed_sep": " $,$ ",
    "isolated_sep": "$$\n, \n$$"
}

def process_pic_file(jpg_file, markdown_directory):
    md_header_splits = []
    output_path = os.path.join(markdown_directory, os.path.splitext(os.path.basename(jpg_file))[0] + ".md")
    only_text = ""
    try:
        if PIC_OCR_PROVIDER == "pix2text":
            files = {"image": (jpg_file, open(jpg_file, 'rb'), 'image/jpeg')}
            logging.info(f"[pix2text:image] request {URL_PIX2TEXT} file={jpg_file}")
            r = requests.post(URL_PIX2TEXT, data=data, files=files, timeout=20)
            outs = r.json()['results']
            logging.info(f"[pix2text:image] success file={jpg_file}")
            if isinstance(outs, str):
                only_text = outs
            else:
                only_text = '\n'.join([out['text'] for out in outs])
        else:
            with open(jpg_file, 'rb') as f:
                files = {"file": f}
                logging.info(f"[paddle:image] request {URL_OCR} file={jpg_file}")
                r = requests.post(URL_OCR, files=files, timeout=20)
                r.raise_for_status()
                only_text = r.json().get('detection_result', '')
                logging.info(f"[paddle:image] success file={jpg_file} len={len(only_text)}")
    except Exception as e:
        logging.warning(f"[image-ocr] failed file={jpg_file} err={e}")
        only_text = ""
    base_name = os.path.basename(jpg_file)
    
    if len(only_text.replace(" ","").replace("\n","")) == 0:
        only_text = f"图片文件：{base_name}"
        
    with open(output_path, 'w') as file:
        file.write(only_text)
    loader = TextLoader(output_path)
    document = loader.load()[0]
    md_header_splits.append(Document(page_content=document.page_content, metadata={"file_path": jpg_file}))
    return md_header_splits
