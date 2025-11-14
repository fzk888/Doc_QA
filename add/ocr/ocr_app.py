import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

ocr = PaddleOCR(use_textline_orientation=True, lang="ch")

from fastapi import FastAPI, File, UploadFile
import io
import json
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import logging



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
    try:
        result = ocr.predict(img)
    except Exception:
        logging.exception("[ocr_app] ocr predict failed")
        try:
            result = ocr.ocr(img, det=True, rec=True)
        except Exception:
            logging.exception("[ocr_app] ocr fallback ocr() failed")
            return ""
    try:
        logging.info(f"[ocr_app] predict type={type(result)}")
        text = _extract_texts(result)
    except Exception:
        logging.exception("[ocr_app] parse predict failed")
        text = ""
    return text

app = FastAPI()

    
@app.post("/detection_pic")
async def detection_card(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    logging.info(f"[ocr_app] received image filename={getattr(file,'filename',None)} size={len(contents)}")
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    w, h = image.size
    max_side = 3000
    if max(w, h) > max_side:
        r = max_side / float(max(w, h))
        image = image.resize((int(w * r), int(h * r)))
    image = np.array(image, dtype=np.uint8)
    try:
        r1 = cord_rec(image)
        logging.info(f"[ocr_app] ocr done chars={len(r1)}")
        return JSONResponse(status_code=200, content={"code": 200, "detection_result": r1})
    except Exception as e:
        logging.exception("[ocr_app] detection failed")
        return JSONResponse(status_code=500, content={"code": 500, "message": str(e), "detection_result": ""})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
