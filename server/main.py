"""
감정인식 FastAPI 백엔드

엔드포인트:
  GET  /api/health              서버·모델 상태
  GET  /api/models              모델 목록 + 성능 지표
  POST /api/analyze             이미지(파일) → 단일 모델 감정 분석
  POST /api/analyze/compare     이미지(파일) → 전체 모델 비교 분석
  POST /api/analyze/base64      base64 이미지 → 단일 모델 (웹캠용)
  GET  /api/pipeline/{name}     시각화 이미지 제공

Usage:
  python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
"""
import base64
import logging
import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from server.predictor import ModelManager, PIPELINE_IMAGES, BASE_DIR, detect_and_crop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Face EMG Emotion API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

manager = ModelManager()


@app.on_event('startup')
async def startup():
    manager.load_all()
    logger.info(f'서버 준비. 로드 모델: {list(manager.predictors.keys())}')


# ── 헬스체크 ──────────────────────────────────────────────────────────────────

@app.get('/api/health')
def health():
    import torch
    return {
        'status':        'ok',
        'loaded_models': list(manager.predictors.keys()),
        'device':        str(next(iter(manager.predictors.values())).device)
                         if manager.predictors else 'none',
        'cuda':          torch.cuda.is_available(),
    }


# ── 모델 목록 + 성능 지표 ─────────────────────────────────────────────────────

@app.get('/api/models')
def get_models():
    return {'models': manager.available_models()}


# ── 이미지 공통 전처리 ────────────────────────────────────────────────────────

def _decode_image(contents: bytes) -> np.ndarray:
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail='이미지 디코딩 실패')
    # 긴 변 1280px 제한
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return img_bgr


# ── 단일 모델 분석 ────────────────────────────────────────────────────────────

@app.post('/api/analyze')
async def analyze(
    file:     UploadFile = File(...),
    model_id: str        = Form(default='densenet121'),
):
    """
    이미지 업로드 → 지정 모델로 감정 분석.
    Returns: emotion, confidence, scores, face_b64, face_detected, infer_ms
    """
    import traceback as tb
    try:
        if model_id not in manager.predictors:
            loaded = list(manager.predictors.keys())
            if not loaded:
                raise HTTPException(status_code=503, detail='로드된 모델 없음')
            model_id = loaded[0]

        contents = await file.read()
        logger.info(f'analyze: model={model_id} file={file.filename} size={len(contents)}')
        img_bgr = _decode_image(contents)
        logger.info(f'  decoded: shape={img_bgr.shape}')
        bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)
        logger.info(f'  face: bbox={bbox} face_shape={face_rgb.shape}')

        result = manager.predict_one(model_id, face_rgb)
        if result is None:
            raise HTTPException(status_code=503, detail=f'모델 추론 실패: {model_id}')

        return {
            **result,
            'face_b64':      face_b64,
            'face_detected': bbox is not None,
            'bbox':          bbox,
            'model_id':      model_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'analyze 500:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


# ── 전체 모델 비교 분석 ───────────────────────────────────────────────────────

@app.post('/api/analyze/compare')
async def analyze_compare(file: UploadFile = File(...)):
    """
    이미지 업로드 → 로드된 모든 모델로 비교 분석.
    Returns: results (모델별 예측), face_b64, face_detected
    """
    import traceback as tb
    try:
        contents = await file.read()
        logger.info(f'compare: file={file.filename} size={len(contents)}')
        img_bgr = _decode_image(contents)
        bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)

        results = manager.predict_all(face_rgb)
        if not results:
            raise HTTPException(status_code=503, detail='로드된 모델 없음')

        return {
            'results':       results,
            'face_b64':      face_b64,
            'face_detected': bbox is not None,
            'bbox':          bbox,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'compare 500:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


# ── base64 분석 (웹캠용) ──────────────────────────────────────────────────────

@app.post('/api/analyze/base64')
async def analyze_base64(payload: dict):
    """
    payload: { image_b64, model_id (optional), compare (optional bool) }
    """
    image_b64 = payload.get('image_b64', '')
    model_id  = payload.get('model_id', 'densenet121')
    compare   = payload.get('compare', False)

    if not image_b64:
        raise HTTPException(status_code=400, detail='image_b64 없음')

    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    img_bytes = base64.b64decode(image_b64)
    img_bgr   = _decode_image(img_bytes)

    bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)

    if compare:
        results = manager.predict_all(face_rgb)
        return {
            'results':       results,
            'face_b64':      face_b64,
            'face_detected': bbox is not None,
        }
    else:
        if model_id not in manager.predictors:
            loaded = list(manager.predictors.keys())
            if not loaded:
                raise HTTPException(status_code=503, detail='로드된 모델 없음')
            model_id = loaded[0]

        result = manager.predict_one(model_id, face_rgb)
        return {
            **result,
            'face_b64':      face_b64,
            'face_detected': bbox is not None,
            'model_id':      model_id,
        }


# ── 내 모델 테스트 ────────────────────────────────────────────────────────────
#
# 사용자가 직접 학습한 .pth 파일을 업로드해 즉석 추론.
# 메모리에만 유지 (서버 재시작 시 초기화).

_custom_models: dict = {}   # token -> EmotionPredictor


@app.post('/api/custom-model/upload')
async def upload_custom_model(file: UploadFile = File(...)):
    """
    .pth 파일 업로드 → 로드 후 token 반환.
    token을 /api/custom-model/analyze 에 사용.
    """
    import traceback as tb, tempfile, uuid

    if not file.filename.endswith('.pth'):
        raise HTTPException(status_code=400, detail='.pth 파일만 허용됩니다.')

    contents = await file.read()
    if len(contents) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail='파일 크기 200MB 초과')

    try:
        import torch
        # 임시 파일에 저장 후 로드
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        ckpt = torch.load(tmp_path, map_location='cpu')
        os.unlink(tmp_path)

        backbone    = ckpt.get('backbone', 'densenet121')
        num_classes = ckpt.get('num_classes', 4)
        in_channels = ckpt.get('in_channels', 3)
        use_clahe   = ckpt.get('use_clahe', False)
        use_edge    = ckpt.get('use_edge', False)
        emotions    = ckpt.get('emotions', ['기쁨', '당황', '분노', '상처'])

        from model import EmotionClassifier
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EmotionClassifier(num_classes, backbone, pretrained=False,
                                  in_channels=in_channels).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        token = str(uuid.uuid4())[:8]
        _custom_models[token] = {
            'model':       model,
            'device':      device,
            'backbone':    backbone,
            'in_channels': in_channels,
            'use_clahe':   use_clahe,
            'use_edge':    use_edge,
            'emotions':    emotions,
            'filename':    file.filename,
        }

        return {
            'token':       token,
            'backbone':    backbone,
            'num_classes': num_classes,
            'in_channels': in_channels,
            'use_clahe':   use_clahe,
            'use_edge':    use_edge,
            'emotions':    emotions,
            'filename':    file.filename,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'upload error:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=f'모델 로드 실패: {e}')


@app.post('/api/custom-model/analyze')
async def analyze_custom(
    file:  UploadFile = File(...),
    token: str        = Form(...),
):
    """
    token에 해당하는 업로드 모델로 이미지 감정 분석.
    """
    import traceback as tb, time
    import torch, torch.nn.functional as F

    if token not in _custom_models:
        raise HTTPException(status_code=404, detail='토큰이 없거나 만료됐습니다. 모델을 다시 업로드하세요.')

    try:
        m        = _custom_models[token]
        model    = m['model']
        device   = m['device']
        emotions = m['emotions']

        img_bgr = _decode_image(await file.read())
        bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)

        # 전처리 (server/predictor.py 동일 로직)
        from dataset import apply_clahe, extract_edge
        face = cv2.resize(face_rgb, (224, 224))
        if m['use_clahe']:
            face = apply_clahe(face)
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (face.astype(np.float32) / 255.0 - MEAN) / STD
        import torch
        tensor = torch.from_numpy(norm.transpose(2, 0, 1))
        if m['use_edge']:
            edge = extract_edge(face).astype(np.float32) / 255.0
            tensor = torch.cat([tensor, torch.from_numpy(edge).unsqueeze(0)], dim=0)
        tensor = tensor.unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
        elapsed = (time.time() - t0) * 1000

        pred = int(probs.argmax())
        EMOJI = {'기쁨': '😄', '당황': '😳', '분노': '😡', '상처': '😢',
                 '불안': '😰', '슬픔': '😢', '중립': '😐'}
        emotion = emotions[pred]
        return {
            'emotion':    emotion,
            'emoji':      EMOJI.get(emotion, '🤔'),
            'confidence': float(probs[pred]),
            'scores':     {e: float(probs[i]) for i, e in enumerate(emotions)},
            'infer_ms':   round(elapsed, 1),
            'face_b64':   face_b64,
            'face_detected': bbox is not None,
            'backbone':   m['backbone'],
            'emotions':   emotions,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'custom analyze error:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/api/custom-model/{token}')
def delete_custom_model(token: str):
    """업로드된 커스텀 모델 메모리에서 제거."""
    if token in _custom_models:
        del _custom_models[token]
    return {'status': 'deleted', 'token': token}


# ── 파이프라인 시각화 이미지 ──────────────────────────────────────────────────

@app.get('/api/pipeline/{name}')
def get_pipeline_image(name: str):
    """
    name: edge_samples | gradcam_samples | class_gradcam | tsne | comparison
    """
    if name not in PIPELINE_IMAGES:
        raise HTTPException(status_code=404, detail=f'알 수 없는 이미지: {name}')
    path = os.path.join(BASE_DIR, PIPELINE_IMAGES[name])
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f'이미지 없음: {path}')
    return FileResponse(path, media_type='image/png')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server.main:app', host='0.0.0.0', port=8000, reload=False)
