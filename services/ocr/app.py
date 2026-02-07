import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import time

ocr = PaddleOCR(use_textline_orientation=True, lang="ch")

from fastapi import FastAPI, File, UploadFile
import io
import json
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] ocr_app - %(message)s", force=True)
logging.info("PaddleOCR 初始化完成")



def _extract_texts(obj):
    collected = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str) and k in ("text", "content", "label", "ocr_text"):
                    collected.append(v)
                else:
                    walk(v)
        elif isinstance(x, (list, tuple)):
            for el in x:
                if isinstance(el, str):
                    collected.append(el)
                elif isinstance(el, (list, tuple)) and len(el) > 1:
                    b = el[1]
                    if isinstance(b, (list, tuple)) and len(b) > 0 and isinstance(b[0], str):
                        collected.append(b[0])
                    else:
                        walk(el)
                else:
                    walk(el)
    walk(obj)
    seen = set()
    out = []
    for t in collected:
        s = t.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return "\n".join(out)

def cord_rec(img):
    start_t = time.time()
    try:
        result = ocr.predict(img)
        logging.info("OCR 预测成功")
    except Exception:
        logging.exception("OCR 预测失败，尝试回退")
        try:
            result = ocr.ocr(img, det=True, rec=True)
            logging.info("回退 ocr() 成功")
        except Exception:
            logging.exception("回退 ocr() 失败")
            return ""
    try:
        logging.info(f"OCR 返回类型={type(result)}")
        text = _extract_texts(result)
        logging.info(f"文本抽取 字符数={len(text)} 耗时={(time.time()-start_t):.3f}秒")
    except Exception:
        logging.exception("文本解析失败")
        text = ""
    return text

app = FastAPI()

    
@app.post("/detection_pic")
async def detection_card(file: UploadFile = File(...)):
    contents = await file.read()
    logging.info(f"接收 文件名={getattr(file,'filename',None)} 大小={len(contents)}B")
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    w, h = image.size
    logging.info(f"打开图片 尺寸={w}x{h}")
    max_side = 3000
    if max(w, h) > max_side:
        r = max_side / float(max(w, h))
        image = image.resize((int(w * r), int(h * r)))
        nw, nh = image.size
        logging.info(f"缩放比例={r:.3f} 新尺寸={nw}x{nh}")
    arr = np.array(image, dtype=np.uint8)
    logging.info(f"预处理 数组形状={arr.shape} 类型={arr.dtype}")
    try:
        st = time.time()
        r1 = cord_rec(arr)
        logging.info(f"识别完成 字符数={len(r1)} 耗时={(time.time()-st):.3f}秒")
        return JSONResponse(status_code=200, content={"code": 200, "detection_result": r1})
    except Exception as e:
        logging.exception("识别失败")
        return JSONResponse(status_code=500, content={"code": 500, "message": str(e), "detection_result": ""})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
