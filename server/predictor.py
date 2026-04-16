from __future__ import annotations
"""
감정인식 멀티모델 추론 관리자 (ONNX Runtime 기반).
Vercel 배포용: torch 불필요, onnxruntime + opencv + numpy만 사용.
"""
import base64
import logging
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# torch 의존성 없이 직접 정의
EMOTIONS   = ['기쁨', '당황', '분노', '상처']                           # 4클래스
EMOTIONS_7 = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']   # 7클래스

EMOTION_EMOJI = {
    '기쁨': '😄', '당황': '😳', '분노': '😡', '상처': '😢',
    '불안': '😨', '슬픔': '😢', '중립': '😐',
}

# ── 모델 레지스트리 ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # ── 7클래스 팀원 모델 ──────────────────────────────────────────────────────
    'resnet18': {
        'label':       'ResNet-18 (강민구)',
        'description': '7개 감정 분류 · 실시간 추론 모델',
        'onnx':        'api/models/resnet18.onnx',
        'color':       '#22C55E',
        'val_acc':     0.82,
        'f1_per':      {e: 0.80 for e in EMOTIONS_7},
        'emotions':    EMOTIONS_7,
        'use_edge':    False,
    },
    'mobilenet_v2': {
        'label':       'MobileNet-V2 (한유승)',
        'description': '7개 감정 분류 · 경량 모바일 모델',
        'onnx':        'api/models/mobilenet_v2.onnx',
        'color':       '#F59E0B',
        'val_acc':     0.0,
        'f1_per':      {e: 0.0 for e in EMOTIONS_7},
        'emotions':    EMOTIONS_7,
        'use_edge':    False,
    },
    # ── 4클래스 DenseNet 모델 (본인) ───────────────────────────────────────────
    'densenet121': {
        'label':       'DenseNet121',
        'description': '4개 감정 분류 · 기본 전처리',
        'onnx':        'api/models/densenet121.onnx',
        'color':       '#4F86C6',
        'val_acc':     0.8762,
        'f1_per':      {'기쁨': 0.968, '당황': 0.902, '분노': 0.860, '상처': 0.828},
        'emotions':    EMOTIONS,
        'use_edge':    False,
    },
    'densenet121_new': {
        'label':       'DenseNet121 New',
        'description': '4개 감정 분류 · 개선 학습',
        'onnx':        'api/models/densenet121_new.onnx',
        'color':       '#2E86AB',
        'val_acc':     0.0,
        'f1_per':      {e: 0.0 for e in EMOTIONS},
        'emotions':    EMOTIONS,
        'use_edge':    False,
    },
    'densenet121_clahe_edge': {
        'label':       'DenseNet121 + Edge',
        'description': '4개 감정 분류 · CLAHE + Canny 엣지 채널',
        'onnx':        'api/models/densenet121_clahe_edge.onnx',
        'color':       '#57B894',
        'val_acc':     0.8476,
        'f1_per':      {'기쁨': 0.959, '당황': 0.881, '분노': 0.813, '상처': 0.807},
        'emotions':    EMOTIONS,
        'use_edge':    True,
    },
}

# 파이프라인 시각화 이미지
PIPELINE_IMAGES = {
    'edge_samples':    'output/viz/edge_samples.png',
    'gradcam_samples': 'output/viz/gradcam_samples.png',
    'class_gradcam':   'output/viz/class_gradcam.png',
    'tsne':            'output/viz/tsne.png',
    'comparison':      'output/comparison.png',
}


# ── 단일 모델 추론기 ──────────────────────────────────────────────────────────

class EmotionPredictor:
    def __init__(self, model_id: str):
        self.model_id  = model_id
        self.info      = MODEL_REGISTRY[model_id]
        self.session   = None
        self.emotions  = self.info.get('emotions', EMOTIONS)
        self.use_clahe = self.info.get('use_clahe', False)
        self.use_edge  = self.info.get('use_edge', False)

    def load(self) -> bool:
        onnx_path = os.path.join(BASE_DIR, self.info['onnx'])
        if not os.path.isfile(onnx_path):
            logger.warning(f'[{self.model_id}] ONNX 파일 없음: {onnx_path}')
            return False
        try:
            self.session = ort.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider'],
            )
            logger.info(f'[{self.model_id}] 로드 완료')
            return True
        except Exception as e:
            logger.error(f'[{self.model_id}] 로드 실패: {e}')
            return False

    def predict(self, face_rgb: np.ndarray) -> dict:
        """face_rgb: (H, W, 3) uint8 → 감정 예측 결과."""
        face = cv2.resize(face_rgb, (224, 224))

        face_f    = face.astype(np.float32) / 255.0
        face_norm = (face_f - MEAN) / STD
        chw       = face_norm.transpose(2, 0, 1)  # (3, H, W)

        if self.use_edge:
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            edge = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
            chw  = np.concatenate([chw, edge[np.newaxis]], axis=0)  # (4, H, W)

        inp = chw[np.newaxis].astype(np.float32)  # (1, C, H, W)

        t0 = time.time()
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: inp})[0][0]  # (num_classes,)
        elapsed = (time.time() - t0) * 1000

        # softmax
        e = np.exp(logits - logits.max())
        probs = e / e.sum()

        pred_idx = int(probs.argmax())
        emo_list = self.emotions
        return {
            'emotion':     emo_list[pred_idx],
            'emoji':       EMOTION_EMOJI.get(emo_list[pred_idx], '🤔'),
            'confidence':  float(probs[pred_idx]),
            'scores':      {em: float(probs[i]) for i, em in enumerate(emo_list)},
            'infer_ms':    round(elapsed, 1),
            'num_classes': len(emo_list),
        }


# ── 얼굴 검출 ─────────────────────────────────────────────────────────────────

def detect_and_crop(img_bgr: np.ndarray):
    """
    Haar Cascade로 가장 큰 얼굴 검출 + 10% 패딩 크롭.
    반환: (bbox_or_None, face_rgb_or_None, face_b64_or_None)
    얼굴 미검출 시 bbox=None, face_rgb=None, face_b64=None 반환.
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, None, None
    else:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(fw * 0.1)
        pad_y = int(fh * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_bgr.shape[1], x + fw + pad_x)
        y2 = min(img_bgr.shape[0], y + fh + pad_y)
        face_bgr = img_bgr[y1:y2, x1:x2]
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    _, buf = cv2.imencode('.jpg', face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    face_b64 = base64.b64encode(buf).decode('utf-8')
    face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return bbox, face_rgb, face_b64


# ── 멀티모델 관리자 ───────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self):
        self.predictors: dict[str, EmotionPredictor] = {}

    def load_all(self):
        for mid in MODEL_REGISTRY:
            p = EmotionPredictor(mid)
            if p.load():
                self.predictors[mid] = p
        logger.info(f'로드된 모델: {list(self.predictors.keys())}')

    def available_models(self) -> list:
        result = []
        for mid, info in MODEL_REGISTRY.items():
            result.append({
                'id':          mid,
                'label':       info['label'],
                'description': info['description'],
                'color':       info['color'],
                'loaded':      mid in self.predictors,
                'val_acc':     info['val_acc'],
                'f1_per':      info['f1_per'],
            })
        return result

    def predict_one(self, model_id: str, face_rgb: np.ndarray) -> dict | None:
        if model_id not in self.predictors:
            return None
        return self.predictors[model_id].predict(face_rgb)

    def predict_all(self, face_rgb: np.ndarray) -> list:
        results = []
        for mid in MODEL_REGISTRY:
            if mid not in self.predictors:
                continue
            res = self.predictors[mid].predict(face_rgb)
            res['model_id']    = mid
            res['model_label'] = MODEL_REGISTRY[mid]['label']
            res['color']       = MODEL_REGISTRY[mid]['color']
            results.append(res)
        return results
