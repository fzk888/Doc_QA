import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import logging
import sys
from PIL import Image
import io
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] pix2text_app - %(message)s", force=True)

# 导入 pix2text
try:
    from pix2text import Pix2Text
    logging.info("正在初始化 Pix2Text...")
    p2t = Pix2Text(analyzer_config={'device': 'cpu'})
    logging.info("Pix2Text 初始化完成")
except Exception as e:
    logging.error(f"Pix2Text 初始化失败: {e}")
    p2t = None

app = FastAPI()

@app.post("/pix2text")
async def pix2text_endpoint(
    file: UploadFile = File(...),
    file_type: str = Form("page"),
    resized_shape: int = Form(768),
    embed_sep: str = Form(" $,$ "),
    isolated_sep: str = Form("$$\n, \n$$")
):
    """
    接收图片文件，返回 OCR 识别的 Markdown 格式文本
    """
    logging.info(f"收到请求: file_type={file_type}, resized_shape={resized_shape}")

    if p2t is None:
        logging.error("Pix2Text 服务未初始化")
        return JSONResponse(status_code=200, content={"results": ""})

    try:
        # 读取文件
        contents = await file.read()
        logging.info(f"接收文件: {file.filename}, 大小: {len(contents)}B")

        # 转换为 PIL Image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 如果需要调整大小
        if resized_shape:
            w, h = image.size
            max_side = max(w, h)
            if max_side > resized_shape:
                ratio = resized_shape / max_side
                new_w, new_h = int(w * ratio), int(h * ratio)
                image = image.resize((new_w, new_h))
                logging.info(f"图片缩放: {w}x{h} -> {new_w}x{new_h}")

        # OCR 识别
        logging.info("开始 OCR 识别...")
        results = p2t.recognize(image)

        # 处理不同格式的返回结果
        if isinstance(results, dict):
            markdown_text = results.get('markdown', '')
            if not markdown_text:
                # 尝试其他可能的字段
                for key in ['text', 'content', 'result']:
                    if key in results:
                        markdown_text = str(results[key])
                        break
                if not markdown_text:
                    markdown_text = str(results)
        else:
            markdown_text = str(results)

        logging.info(f"识别成功, 文本长度: {len(markdown_text)}")
        return JSONResponse(status_code=200, content={"results": markdown_text})

    except Exception as e:
        logging.exception(f"识别失败: {e}")
        # 发生错误时返回空字符串而不是错误，避免客户端崩溃
        return JSONResponse(status_code=200, content={"results": ""})

if __name__ == "__main__":
    logging.info("启动 Pix2Text 服务在端口 8503...")
    uvicorn.run(app, host="0.0.0.0", port=8503)
