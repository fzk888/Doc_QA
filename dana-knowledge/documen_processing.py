import os
import re
import pdfplumber
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import pypandoc
from pypdf import PdfReader
import requests
def is_valid_table(table_df, min_char_count=120):
    total_char_count = sum(len(str(cell)) for _, row in table_df.iterrows() for cell in row)
    if total_char_count < min_char_count:
        return False
    return True
def extract_all_table_and_textand_image(pdf_path, out_path_md,image_dir):
    pdf = pdfplumber.open(pdf_path)

    def is_sentence_end(text):
        return re.search(r'[。？！.?!]$', text) is not None

    def not_within_bboxes(obj):
        return not any(
            obj["x0"] + obj["x1"] >= 2 * bbox[0] and obj["x0"] + obj["x1"] < 2 * bbox[2] and
            obj["top"] + obj["bottom"] >= 2 * bbox[1] and obj["top"] + obj["bottom"] < 2 * bbox[3]
            for bbox in bboxes
        )

    with open(out_path_md, 'w', encoding='utf-8') as f:
        temp_table = None
        table_name = []
        prev_page_text = ""

        for page in tqdm(pdf.pages):
            page_num = page.page_number
            tables = page.find_tables(
                table_settings={
                    "vertical_strategy": "explicit",
                    "horizontal_strategy": "explicit",
                    "explicit_vertical_lines": page.curves + page.edges,
                    "explicit_horizontal_lines": page.curves + page.edges,
                }
            )
            bboxes = [table.bbox for table in tables]

            page_text = page.filter(not_within_bboxes).extract_text()
            page_text = "\n".join(page_text.split("\n")[1:-1])
            if prev_page_text:
                last_line = prev_page_text.split("\n")[-1]
                first_line = page_text.split("\n")[0]

                if not is_sentence_end(last_line) and is_sentence_end(last_line + first_line):
                    f.write(f"# Page {page_num - 1} - Text\n\n")
                    f.write(prev_page_text + first_line)
                    f.write("\n\n")
                    page_text = "\n".join(page_text.split("\n")[1:])
                else:
                    f.write(f"# Page {page_num - 1} - Text\n\n")
                    f.write(prev_page_text)
                    f.write("\n\n")

            if page_text.strip():
                prev_page_text = page_text
            else:
                prev_page_text = ""
# 保存页面中的图片
            images = page.images
            for image in images:
                image_name = f"page_{page_num}_image_{images.index(image)}.png"
                image_path = os.path.join(image_dir, image_name)
                with open(image_path, "wb") as image_file:
                    image_file.write(image["stream"].get_data())
                       # 将图片信息写入Markdown文档
                f.write(f"# Page {page_num} - Image {images.index(image)}\n\n")
                f.write(f"![Page {page_num} - Image {images.index(image)}]({image_path})\n\n")

            for table_id, table in enumerate(page.extract_tables(), start=1):
                if temp_table is not None:
                    if page.bbox[3] - tables[0].bbox[1] + 40 >= page.chars[0].get('y1'):
                        df = pd.DataFrame(table)
                        temp_table = pd.concat([temp_table, df], axis=0)
                        table_name.append(f"{page_num}-{table_id}")
                        if page.chars[-1].get('y0') < page.bbox[3] - tables[0].bbox[3]:
                            if is_valid_table(temp_table):
                                f.write(f"# Page {page_num} - Table {table_id}\n\n")
                                f.write(temp_table.to_markdown(index=False))
                                f.write("\n\n")
                            temp_table = None
                            table_name = []
                        else:
                            break
                    else:
                        if is_valid_table(temp_table):
                            f.write(f"# Page {page_num} - Table {table_name[0].split('-')[1]}\n\n")
                            f.write(temp_table.to_markdown(index=False))
                            f.write("\n\n")
                        temp_table = None
                        table_name = []
                        if page.chars[-1].get('y0') < page.bbox[3] - tables[table_id - 1].bbox[3]:
                            df = pd.DataFrame(table)
                            if is_valid_table(df):
                                f.write(f"# Page {page_num} - Table {table_id}\n\n")
                                f.write(df.to_markdown(index=False))
                                f.write("\n\n")
                        else:
                            temp_table = pd.DataFrame(table)
                            table_name.append(f"{page_num}-{table_id}")
                else:
                    if page.chars[-1].get('y0') < page.bbox[3] - tables[table_id - 1].bbox[3]:
                        df = pd.DataFrame(table)
                        if is_valid_table(df):
                            f.write(f"# Page {page_num} - Table {table_id}\n\n")
                            f.write(df.to_markdown(index=False))
                            f.write("\n\n")
                    else:
                        temp_table = pd.DataFrame(table)
                        table_name.append(f"{page_num}-{table_id}")
    pdf.close()


#常规md拆分方式-------固定头拆解

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4")
]

#递归拆解方式-----根据各种不同的头的拆解，例如**----**，***----***，****----****等等
separators=["#####","######","#######","####", "###","\n**"]
    

    # 定义拆分器
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


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


def process_doc_file(doc_file, image_output_dir, markdown_directory):
    file_extension = os.path.splitext(doc_file)[1].lower()
    md_header_splits = []
    
    if file_extension == ".docx":
        try:
            # 获取文档的基文件名（不带路径和扩展名）
            base_name = os.path.splitext(os.path.basename(doc_file))[0]
            
            # 创建该文档的专属媒体目录
            media_dir = os.path.join(image_output_dir, base_name)
            os.makedirs(media_dir, exist_ok=True)
            
            # 将 .docx 文件转换为 Markdown，提取的媒体文件保存在 media_dir 中
            markdown_text = pypandoc.convert_file(doc_file, 'markdown', extra_args=['--extract-media=' + media_dir])
            
            # 生成 Markdown 文件的路径
            markdown_file = os.path.join(markdown_directory, os.path.splitext(os.path.basename(doc_file))[0] + ".md")
            
            # 将转换后的 Markdown 文本写入文件
            with open(markdown_file, 'w', encoding='utf-8') as file:
                file.write(markdown_text)
            
            # 加载 Markdown 文件
            loader = TextLoader(markdown_file)
            document = loader.load()[0]
            
            # 拆分 Markdown 内容
            text_splitter = RecursiveCharacterTextSplitter(separators=["#####", "######", "#######", "####", "###", "\n**"], chunk_size=200, chunk_overlap=30)
            md_header_splits = text_splitter.create_documents([markdown_text])
            
            if len(md_header_splits) == 0:
                # 如果没有拆分,将整个文档内容作为一个拆分
                md_header_splits = [document]
            else:
                for split in md_header_splits:
                    # 获取每个分块的元数据
                    metadata = split.metadata
                    
                    # 从分块内容中提取所有以#开头的行
                    lines = split.page_content.strip().split("\n")
                    title_lines = []
                    for line in lines:
                        if line.startswith("#"):
                            title_lines.append(line.strip("#").strip())
                    
                    # 将提取出的标题行作为 docuname
                    if title_lines:
                        metadata["docuname"] = "\n".join(title_lines)
                    
                    # 获取元数据中最后一级的标题
                    last_header = None
                    headers_to_split_on = ["#####", "######", "#######", "####", "###"]
                    for header_level in reversed(headers_to_split_on):
                        if header_level in metadata:
                            last_header = metadata[header_level]
                            break
                    
                    # 将最后一级标题添加到拆分部分的内容前面
                    if last_header:
                        split.page_content = f"## {last_header}\n\n{split.page_content}"
                    split.metadata["file_path"] = doc_file
            
            # 如果只有一个拆分且内容长度超过500 token，按段落进行切割并重新拆分
            if len(md_header_splits) == 1 and len(md_header_splits[0].page_content.split()) > 500:
                original_content = md_header_splits[0].page_content
                split_contents = split_text_preserving_tables(original_content)
                md_header_splits = [Document(page_content=chunk, metadata={"file_path": doc_file}) for chunk in split_contents]
        
        except RuntimeError as e:
            print(f"转换文件 {doc_file} 时出错：{str(e)}")
            print("跳过该文件,继续处理下一个文件。")
    
    return md_header_splits

def process_md_file(md_file):
    loader = TextLoader(md_file)
    document = loader.load()[0]
    # 拆分 Markdown 内容
    text_splitter = RecursiveCharacterTextSplitter(separators=["#####","######","#######","####", "###","\n**"], chunk_size=300, chunk_overlap=30)
    md_header_splits =  text_splitter.create_documents([document.page_content])
    # md_header_splits =  text_splitter.create_documents([markdown_text])
    if len(md_header_splits) == 0:
        # 如果没有拆分,将整个文档内容作为一个拆分
        md_header_splits = [document]
    else:
        for split in md_header_splits:
            # 获取每个分块的元数据
            metadata = split.metadata
            
            # 从分块内容中提取所有以#开头的行
            lines = split.page_content.strip().split("\n")
            title_lines = []
            for line in lines:
                if line.startswith("#"):
                    title_lines.append(line.strip("#").strip())
            
            # 将提取出的标题行作为docuname
            if title_lines:
                metadata["docuname"] = "\n".join(title_lines)
            
            # 获取元数据中最后一级的标题
            last_header = None
            for header_level in reversed(headers_to_split_on):
                if header_level[1] in metadata:
                    last_header = metadata[header_level[1]]
                    break
            
            # 将最后一级标题添加到拆分部分的内容前面
            if last_header:
                split.page_content = f"## {last_header}\n\n{split.page_content}"
            split.metadata["file_path"] = md_file
    return md_header_splits


def process_txt_file(txt_file, markdown_directory):
    with open(txt_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 拆分 Markdown 内容
    text_splitter = RecursiveCharacterTextSplitter(separators=["#####","######","#######","####", "###","\n**","\n\n",""], chunk_size=2000, chunk_overlap=200)
    md_header_splits = text_splitter.create_documents([text])
    
    if len(md_header_splits) == 0:
        # 如果没有拆分,将整个文档内容作为一个拆分
        md_header_splits = [Document(page_content=text)]
    else:
        # 创建 Markdown 文件路径
        markdown_file = os.path.join(markdown_directory, f"{os.path.splitext(os.path.basename(txt_file))[0]}.md")
        
        with open(markdown_file, 'w', encoding='utf-8') as file:
            for index, split in enumerate(tqdm(md_header_splits, desc="Processing splits")):
                # 获取每个分块的元数据
                metadata = split.metadata
                
                # 从分块内容中提取所有以#开头的行
                lines = split.page_content.strip().split("\n")
                title_lines = [line.strip("#").strip() for line in lines if line.startswith("#")]
                
                # 将提取出的标题行作为docuname
                if title_lines:
                    metadata["docuname"] = "\n".join(title_lines)
                
                # 获取元数据中最后一级的标题
                last_header = None
                for header_level in reversed(headers_to_split_on):
                    if header_level[1] in metadata:
                        last_header = metadata[header_level[1]]
                        break
                
                # 将最后一级标题添加到拆分部分的内容前面
                if last_header:
                    split.page_content = f"## {last_header}\n\n{split.page_content}"
                
                split.metadata["file_path"] = txt_file
                
                # 将分块内容写入文件
                file.write(split.page_content)
                file.write("\n\n---\n\n")  # 添加分块分隔标志
    
    return md_header_splits


def pdf_to_markdown(url, pdf_file_path, markdown_file_path, extract_images=False):
    """
    Convert a PDF file to Markdown format by sending a request to the specified URL.

    :param url: The URL to send the POST request to.
    :param pdf_file_path: The file path of the PDF to be converted.
    :param markdown_file_path: The file path where the resulting Markdown will be saved.
    :param extract_images: Optional parameter to extract images from the PDF.
    """
    try:
        # Open and read the PDF file
        with open(pdf_file_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()

        # Prepare the files and parameters for the request
        files = {'pdf_files': (os.path.basename(pdf_file_path), pdf_content, 'application/pdf')}
        params = {'extract_images': extract_images}

        # Send the POST request
        response = requests.post(url, files=files, params=params)
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

        # Get the response content as JSON
        response_json = response.json()[0]['markdown']

        # Convert the JSON response to Markdown format
        markdown_content = json_to_markdown(response_json)

        # Write the formatted Markdown content into the file
        with open(markdown_file_path, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write(markdown_content)

        print(f"Response content has been written to {markdown_file_path}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
    except json.JSONDecodeError:
        print("An error occurred while parsing the response JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def json_to_markdown(data, level=1):
    """
    Convert JSON data to Markdown format.

    :param data: The JSON data to convert.
    :param level: The current heading level.
    :return: The formatted Markdown content.
    """
    md_content = ""
    if isinstance(data, dict):
        for key, value in data.items():
            md_content += f"{'#' * level} {key}\n\n"
            md_content += json_to_markdown(value, level + 1)
    elif isinstance(data, list):
        for item in data:
            md_content += json_to_markdown(item, level)
    else:
        md_content += f"{data}\n\n"
    return md_content

url = "http://localhost:8000/convert"

from pypdf import PdfReader
def process_pdf(pdf_path, image_dir, md_dir, chunk_size=1000):
    try:
        print(f"开始处理文档: {pdf_path}")
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path_md = os.path.join(md_dir, f"{pdf_name}.md") 
        image_dir_file = os.path.join(image_dir, pdf_name)
        os.makedirs(image_dir_file, exist_ok=True)  # 为每个文档创建单独的图片文件夹
        
        try:
            extract_all_table_and_textand_image(pdf_path, out_path_md, image_dir_file)
            use_traditional_parsing = False
        except Exception as e:
            print(f"使用 extract_all_table_and_textand_image 解析文档出错: {e}")
            print("切换到传统的 maker_api 解析方法")
            use_traditional_parsing = True
            
            # # 使用 pypdf 库解析 PDF 文档
            # with open(pdf_path, 'rb') as file:
            #     pdf_reader = PdfReader(file)
            #     text = ""
            #     for page in pdf_reader.pages:
            #         text += page.extract_text()
            
            # # 将解析后的文本保存到 Markdown 文件中
            # with open(out_path_md, 'w', encoding='utf-8') as file:
            #     file.write(text)
            #使用maker_api解析文档：
            pdf_to_markdown(url, pdf_path, out_path_md, extract_images=False)
        # 加载生成的 Markdown 文件
        loader = TextLoader(out_path_md)
        document = loader.load()[0]
        
        if use_traditional_parsing:
            # 使用传统解析方法,按照指定的 chunk 大小分割文本
            text_splits = []
            text = document.page_content
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                metadata = {"document_name": chunk[:20], "file_path": pdf_path}
                text_splits.append(Document(page_content=chunk, metadata=metadata))
        else:
            # 使用原来的分割方式
            md_header_splits = markdown_splitter.split_text(document.page_content)
            
            if len(md_header_splits) > 0:
                # 判断文档名长度
                if len(pdf_name) >= 12:  # 假设6个中文字符等于12个英文字符
                    document_title = pdf_name
                else:
                    # 查找 "# Page 1 - Text" 对应的文本内容
                    page1_text = ""
                    for split in md_header_splits:
                        if "# Page 1 - Text" in split.page_content:
                            page1_text = split.page_content.split("# Page 1 - Text")[-1].strip()
                            break
                    
                    if len(page1_text) > 300:
                        document_title = page1_text[:150]  # 取前50个中文字符作为标题
                    else:
                        document_title = page1_text
                
                document.metadata["document_name"] = document_title
                
                # 给拆分后的每个元素添加文档名和文件路径的 metadata
                for split in md_header_splits:
                    split.metadata["document_name"] = document_title 
                    split.metadata["file_path"] = pdf_path
            
            text_splits = md_header_splits
        
        print(f"文档处理完成: {pdf_path}")
        return text_splits
    
    except Exception as e:
        print(f"处理文档出错: {pdf_path}, 错误信息: {e}")
        return []