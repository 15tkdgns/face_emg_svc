from __future__ import annotations
"""
감정인식 멀티모델 추론 관리자.
output/ 하위 4개 학습 결과를 모두 로드해 단일/비교 추론 지원.
"""
import base64
import logging
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset import EMOTIONS as ALL_EMOTIONS, SAMPLE_EMOTIONS, apply_clahe, extract_edge
from model import EmotionClassifier

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 4클래스 모델용
EMOTIONS = SAMPLE_EMOTIONS  # ['기쁨', '당황', '분노', '상처']
# 7클래스 모델용
EMOTIONS_7 = ALL_EMOTIONS   # ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']

EMOTION_EMOJI = {
    '기쁨': '😄', '당황': '😳', '분노': '😡', '상처': '😢',
    '불안': '😨', '슬픔': '😢', '중립': '😐',
}

# ── 모델 레지스트리 ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'resnet18': {
        'label':       'ResNet-18 (강민구)',
        'description': '7개 감정 분류 · 실시간 추론 모델',
        'ckpt':        'kang_mingoo/resnet18_emotion_best.pth',
        'color':       '#22C55E',
        'val_acc':     0.82,
        'f1_per':      {e: 0.80 for e in EMOTIONS_7},
        'num_classes': 7,
        'emotions':    EMOTIONS_7,
        'backbone':    'resnet18',
    },
    'mobilenet_v2': {
        'label':       'MobileNet-V2 (한유승)',
        'description': '7개 감정 분류 · 경량 모바일 모델',
        'ckpt':        '한유승/best_emotion_model.pth',
        'color':       '#F59E0B',
        'val_acc':     0.0,
        'f1_per':      {e: 0.0 for e in EMOTIONS_7},
        'num_classes': 7,
        'emotions':    EMOTIONS_7,
        'backbone':    'mobilenet_v2',
    },
    'efficientnet_v2_s': {
        'label':       'EfficientNetV2-S (신희원)',
        'description': '7개 감정 분류 · Acc 91.4%',
        'ckpt':        '신희원/best_efficientnet_v2_s_clean.pth',
        'color':       '#EC4899',
        'val_acc':     0.914,
        'f1_per':      {e: 0.91 for e in EMOTIONS_7},
        'num_classes': 7,
        'emotions':    EMOTIONS_7,
        'backbone':    'efficientnet_v2_s',
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
        self.model_id    = model_id
        self.info        = MODEL_REGISTRY[model_id]
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model       = None
        self.use_clahe   = False
        self.use_edge    = False
        self.in_channels = 3
        # 모델별 감정 리스트 (7클래스 vs 4클래스)
        self.emotions    = self.info.get('emotions', EMOTIONS)

    def load(self) -> bool:
        ckpt_path = os.path.join(BASE_DIR, self.info['ckpt'])
        if not os.path.isfile(ckpt_path):
            logger.warning(f'[{self.model_id}] 체크포인트 없음: {ckpt_path}')
            return False
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            # ── raw state_dict 처리 (메타데이터 없는 .pth) ────────
            is_raw = isinstance(ckpt, dict) and 'state_dict' not in ckpt \
                     and any(k.startswith(('conv', 'layer', 'bn', 'features')) for k in ckpt.keys())

            if is_raw:
                backbone    = self.info.get('backbone', 'resnet18')
                num_classes = self.info.get('num_classes', len(self.emotions))
                self.in_channels = 3
                self.use_clahe   = False
                self.use_edge    = False

                from model import build_model
                self.model = build_model(
                    num_classes, backbone, pretrained=False, in_channels=3
                ).to(self.device)

                # build_model converts fc → Sequential(Dropout, Linear)
                # but raw ckpt has fc.weight / fc.bias → remap to fc.1.weight / fc.1.bias
                remapped = {}
                for k, v in ckpt.items():
                    if k == 'fc.weight':
                        remapped['fc.1.weight'] = v
                    elif k == 'fc.bias':
                        remapped['fc.1.bias'] = v
                    else:
                        remapped[k] = v
                self.model.load_state_dict(remapped)
            else:
                # ── wrapped checkpoint 처리 ────────
                backbone         = ckpt.get('backbone', 'densenet121')
                num_classes      = ckpt.get('num_classes', len(self.emotions))
                self.in_channels = ckpt.get('in_channels', 3)
                self.use_clahe   = ckpt.get('use_clahe', False)
                self.use_edge    = ckpt.get('use_edge', False)

                self.model = EmotionClassifier(
                    num_classes, backbone, pretrained=False, in_channels=self.in_channels
                ).to(self.device)
                self.model.load_state_dict(ckpt['state_dict'])

            self.model.eval()
            logger.info(f'[{self.model_id}] 로드 완료 (classes={len(self.emotions)}, raw={is_raw})')
            return True
        except Exception as e:
            logger.error(f'[{self.model_id}] 로드 실패: {e}')
            import traceback; traceback.print_exc()
            return False

    def predict(self, face_rgb: np.ndarray) -> dict:
        """face_rgb: (H, W, 3) uint8 → 감정 예측 결과."""
        face = cv2.resize(face_rgb, (224, 224))

        if self.use_clahe:
            face = apply_clahe(face)

        face_f   = face.astype(np.float32) / 255.0
        face_norm = (face_f - MEAN) / STD
        rgb_tensor = torch.from_numpy(face_norm.transpose(2, 0, 1))  # (3, H, W)

        if self.use_edge:
            edge = extract_edge(face).astype(np.float32) / 255.0
            edge_t = torch.from_numpy(edge).unsqueeze(0)
            tensor = torch.cat([rgb_tensor, edge_t], dim=0)
        else:
            tensor = rgb_tensor

        tensor = tensor.unsqueeze(0).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
        elapsed = (time.time() - t0) * 1000

        emo_list = self.emotions
        pred_idx = int(probs.argmax())
        return {
            'emotion':    emo_list[pred_idx],
            'emoji':      EMOTION_EMOJI.get(emo_list[pred_idx], '🤔'),
            'confidence': float(probs[pred_idx]),
            'scores':     {e: float(probs[i]) for i, e in enumerate(emo_list)},
            'infer_ms':   round(elapsed, 1),
            'num_classes': len(emo_list),
        }


# ── 얼굴 검출 ─────────────────────────────────────────────────────────────────

def detect_and_crop(img_bgr: np.ndarray):
    """
    Haar Cascade로 가장 큰 얼굴 검출 + 10% 패딩 크롭.
    반환: (bbox_or_None, face_bgr, face_b64)
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )

    if len(faces) == 0:
        # 얼굴 미검출 → 중앙 정사각형 크롭
        h, w = img_bgr.shape[:2]
        s = min(h, w)
        x1, y1 = (w - s) // 2, (h - s) // 2
        face_bgr = img_bgr[y1:y1 + s, x1:x1 + s]
        bbox = None
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
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
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
        for mid in MODEL_REGISTRY:   # 등록 순서 유지
            if mid not in self.predictors:
                continue
            res = self.predictors[mid].predict(face_rgb)
            res['model_id']    = mid
            res['model_label'] = MODEL_REGISTRY[mid]['label']
            res['color']       = MODEL_REGISTRY[mid]['color']
            results.append(res)
        return results
