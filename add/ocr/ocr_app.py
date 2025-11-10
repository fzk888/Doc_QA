from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

from fastapi import FastAPI, File, UploadFile
import io
import json
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse



def cord_rec(img):
    result = ocr.ocr(img, cls=True)
    text = ""
    try:
        for i in result[0]:

                text += str(i[1][0])
                text += "\n"
    except:
        pass
    return text

app = FastAPI()

    
@app.post("/detection_pic")
async def detection_card(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    image = np.array(image)
    r1 = cord_rec(image)
    return JSONResponse(status_code=200, content={"code": 200, "detection_result": r1})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)