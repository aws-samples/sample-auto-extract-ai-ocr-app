import os
import json
import logging
import traceback
import base64
import numpy as np
import cv2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import uvicorn
import sys

# Logging configuration
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s:%(name)s: %(message)s"
    ))
    logger.addHandler(h)

app = FastAPI()
ocr_instance = None


def load_ocr_models():
    """Initialize PaddleOCR model"""
    global ocr_instance

    logger.info("Initializing PaddleOCR...")

    try:
        ocr_instance = PaddleOCR(
            lang='japan',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        logger.info("PaddleOCR initialization completed")
        return ocr_instance

    except Exception as e:
        logger.error(f"PaddleOCR initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def parse_request_data(request_body: bytes, content_type: str):
    """Parse request data and extract image data"""
    logger.info(f"Parsing request - Content-Type: {content_type}")

    try:
        if content_type == 'application/json':
            # Expect base64 encoded image in JSON format
            input_data = json.loads(request_body)
            if 'image' in input_data:
                image_data = base64.b64decode(input_data['image'])
                return {'image_data': image_data}
            else:
                return {'error': 'No image field in JSON request'}
        elif content_type and content_type.startswith('image/'):
            # Direct image binary data
            return {'image_data': request_body}
        else:
            return {'error': f'Unsupported Content-Type: {content_type}'}
    except Exception as e:
        logger.error(f"Request parsing error: {str(e)}")
        return {'error': str(e)}


def perform_ocr(input_data, ocr_model):
    """Perform OCR processing and return results"""
    logger.info("Starting OCR processing")

    try:
        # Error checking
        if 'error' in input_data:
            return {'error': input_data['error'], 'words': []}

        if 'image_data' not in input_data:
            return {'error': 'No image data available', 'words': []}

        # Load image data with OpenCV
        image_data = input_data['image_data']
        img = np.frombuffer(image_data, dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            return {'error': 'Failed to decode image', 'words': []}

        logger.info("Running OCR...")

        # Execute OCR with PaddleOCR
        results = ocr_model.predict(img)
        logger.info("OCR processing completed")

        # Convert results to unified format
        json_data = {"words": []}

        for result in results:
            if isinstance(result, dict) and 'rec_texts' in result and 'rec_polys' in result and 'rec_scores' in result:
                for i, (text, poly, score) in enumerate(zip(
                    result['rec_texts'],
                    result['rec_polys'], 
                    result['rec_scores']
                )):
                    if text.strip():  # Skip empty strings
                        word_dict = {
                            "id": i,
                            "content": text,
                            "rec_score": float(score),
                            "points": poly.tolist() if hasattr(poly, 'tolist') else poly
                        }
                        json_data["words"].append(word_dict)

        logger.info(f"OCR completed: {len(json_data['words'])} words detected")
        return json_data

    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "words": []}


@app.get("/ping")
async def ping():
    """Health check endpoint"""
    logger.info("Health check requested")
    health = ocr_instance is not None
    status = 200 if health else 404
    return JSONResponse(
        content={"status": "healthy" if health else "unhealthy"},
        status_code=status
    )


@app.post("/invocations")
async def invocations(request: Request):
    """Main OCR inference endpoint"""
    logger.info("Inference request received")

    try:
        content_type = request.headers.get('content-type')
        request_body = await request.body()

        logger.info(f"Content-Type: {content_type}, Data size: {len(request_body)} bytes")

        # Parse request data
        input_data = parse_request_data(request_body, content_type)

        # Execute OCR processing
        prediction = perform_ocr(input_data, ocr_instance)

        logger.info("Returning OCR results")
        return JSONResponse(content=prediction)

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"error": str(e), "words": []},
            status_code=500
        )


if __name__ == '__main__':
    logger.info("Starting application...")

    # Initialize OCR model
    logger.info("Loading OCR model...")
    load_ocr_models()

    # Start FastAPI server
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host='0.0.0.0', port=port)
