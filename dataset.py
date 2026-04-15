"""
AI Hub 감정인식 데이터셋 로더 (데이터셋 82)

전처리 옵션:
  use_clahe  : CLAHE 히스토그램 평활화 (조도 불균형 보정)
  use_edge   : Canny 엣지를 4번째 채널로 추가 (RGB → RGBE)
  use_align  : mediapipe로 눈 위치 기반 얼굴 정렬
"""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

EMOTIONS = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
SAMPLE_EMOTIONS = ['기쁨', '당황', '분노', '상처']

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def _consensus_box(item: dict):
    keys = ['annot_A', 'annot_B', 'annot_C']
    boxes = [item[k]['boxes'] for k in keys if k in item and item[k]]
    if not boxes:
        return None
    return (
        np.mean([b['minX'] for b in boxes]),
        np.mean([b['minY'] for b in boxes]),
        np.mean([b['maxX'] for b in boxes]),
        np.mean([b['maxY'] for b in boxes]),
    )


def _load_json(path: str) -> list:
    """UTF-8 / UTF-8-BOM / unicode escape 모두 지원."""
    with open(path, 'rb') as f:
        raw = f.read()
    return json.loads(raw)


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE를 LAB L채널에 적용해 조도 불균형 보정."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def extract_edge(img_rgb: np.ndarray) -> np.ndarray:
    """Canny 엣지맵 반환 (H, W), uint8 0~255."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150)


def align_face(img_rgb: np.ndarray) -> np.ndarray:
    """
    mediapipe FaceMesh로 눈 랜드마크를 검출해 얼굴 정렬.
    검출 실패 시 원본 반환.
    """
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_mesh
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                               refine_landmarks=True) as fm:
            res = fm.process(img_rgb)
            if not res.multi_face_landmarks:
                return img_rgb
            lm = res.multi_face_landmarks[0].landmark
            h, w = img_rgb.shape[:2]
            # 왼눈 중심(133), 오른눈 중심(362) 인덱스
            le = np.array([lm[133].x * w, lm[133].y * h])
            re = np.array([lm[362].x * w, lm[362].y * h])
            angle = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
            cx, cy = w / 2, h / 2
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            return cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR)
    except Exception:
        return img_rgb


class EmotionDataset(Dataset):
    """
    한국인 감정인식 이미지 데이터셋.

    Args:
        data_root  : New_sample_3 폴더 경로
        emotions   : 사용할 감정 목록
        split      : 'train' | 'val' | 'all'
        val_ratio  : validation 비율
        image_size : 크롭 후 리사이즈 크기
        augment    : 학습 데이터 증강 여부
        use_clahe  : CLAHE 히스토그램 평활화
        use_edge   : 엣지 채널 추가 (출력 4ch)
        use_align  : mediapipe 얼굴 정렬
        seed       : random seed
    """

    def __init__(
        self,
        data_root: str,
        emotions: list = None,
        split: str = 'train',
        val_ratio: float = 0.2,
        image_size: int = 224,
        augment: bool = True,
        use_clahe: bool = False,
        use_edge: bool = False,
        use_align: bool = False,
        seed: int = 42,
        new_label_root: str = None,
    ):
        """
        data_root       : New_sample_3 폴더 경로 (원천데이터/라벨링데이터 포함)
        new_label_root  : 새 구조 레이블 루트 (예: 라벨링데이터-.../라벨링데이터/)
                          지정 시 JSON은 여기서, 이미지는 data_root/원천데이터/EMOIMG_{e}_SAMPLE/ 우선 검색
        """
        self.data_root      = data_root
        self.new_label_root = new_label_root
        self.emotions       = emotions or SAMPLE_EMOTIONS
        self.split          = split
        self.image_size     = image_size
        self.use_clahe      = use_clahe
        self.use_edge       = use_edge
        self.use_align      = use_align

        self.samples = self._build_samples(val_ratio, seed)

        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
        ]
        aug_transforms = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.Resize((image_size, image_size)),
        ]
        tail = [transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]

        self.rgb_transform = transforms.Compose(
            (aug_transforms if split == 'train' and augment else base_transforms) + tail
        )

    def _find_label_and_img_dir(self, emotion: str):
        """(json_path, img_dir) 반환. 없으면 (None, None)."""
        # ── 새 구조: new_label_root/{emotion}/img_emotion_training_data({emotion}).json ──
        if self.new_label_root:
            new_json = os.path.join(self.new_label_root, emotion,
                                    f'img_emotion_training_data({emotion}).json')
            # 이미지: data_root/원천데이터/EMOIMG_{emotion}_SAMPLE 우선,
            #         없으면 new_label_root/../원천데이터/{emotion}
            img_dir_legacy = os.path.join(self.data_root, '원천데이터', f'EMOIMG_{emotion}_SAMPLE')
            img_dir_new    = os.path.join(os.path.dirname(self.new_label_root), '원천데이터', emotion)
            img_dir = img_dir_legacy if os.path.isdir(img_dir_legacy) else (
                      img_dir_new    if os.path.isdir(img_dir_new)    else None)
            if os.path.isfile(new_json) and img_dir:
                return new_json, img_dir

        # ── 기존 구조: data_root/라벨링데이터/EMOIMG_{emotion}_SAMPLE/*.json ──
        label_dir = os.path.join(self.data_root, '라벨링데이터', f'EMOIMG_{emotion}_SAMPLE')
        img_dir   = os.path.join(self.data_root, '원천데이터',   f'EMOIMG_{emotion}_SAMPLE')
        if os.path.isdir(label_dir) and os.path.isdir(img_dir):
            json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
            if json_files:
                return os.path.join(label_dir, json_files[0]), img_dir

        return None, None

    def _build_samples(self, val_ratio, seed):
        all_samples = []
        for emotion in self.emotions:
            json_path, img_dir = self._find_label_and_img_dir(emotion)
            if json_path is None:
                print(f'  [{emotion}] 데이터 없음 → 건너뜀')
                continue
            records  = _load_json(json_path)
            existing = set(os.listdir(img_dir))
            added = 0
            for rec in records:
                fname = rec.get('filename', '')
                if fname not in existing:
                    continue
                box = _consensus_box(rec)
                if box is None:
                    continue
                all_samples.append({
                    'image_path': os.path.join(img_dir, fname),
                    'label': self.emotions.index(emotion),
                    'emotion': emotion,
                    'box': box,
                    'gender': rec.get('gender', ''),
                    'age': rec.get('age', -1),
                })
                added += 1
            print(f'  [{emotion}] {added}장 로드')

        rng = np.random.RandomState(seed)
        idx = np.arange(len(all_samples))
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - val_ratio))
        if self.split == 'train':
            idx = idx[:cut]
        elif self.split == 'val':
            idx = idx[cut:]
        return [all_samples[i] for i in idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 한글 경로 대응
        with open(item['image_path'], 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"이미지 읽기 실패: {item['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 얼굴 크롭 (10% 패딩)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = item['box']
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * 0.1))
        y1 = max(0, int(y1 - bh * 0.1))
        x2 = min(w, int(x2 + bw * 0.1))
        y2 = min(h, int(y2 + bh * 0.1))
        face = img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img

        # 얼굴 정렬
        if self.use_align:
            face = align_face(face)

        # CLAHE
        if self.use_clahe:
            face = apply_clahe(face)

        # RGB 텐서 (3, H, W)
        rgb_tensor = self.rgb_transform(face)  # (3, H, W)

        # 엣지 채널 추가 → (4, H, W)
        if self.use_edge:
            face_resized = cv2.resize(face, (self.image_size, self.image_size))
            edge = extract_edge(face_resized).astype(np.float32) / 255.0
            edge_tensor = torch.from_numpy(edge).unsqueeze(0)  # (1, H, W)
            tensor = torch.cat([rgb_tensor, edge_tensor], dim=0)
        else:
            tensor = rgb_tensor

        return tensor, item['label']

    def class_counts(self) -> dict:
        from collections import Counter
        return dict(Counter(s['emotion'] for s in self.samples))


class BibleDataset(Dataset):
    """
    바이블코딩 감정 데이터셋.
    이미지가 이미 224x224 크롭 완료 → 얼굴 크롭 없이 전체 이미지 사용.

    Args:
        data_root  : 바이블코딩 폴더 경로 (샘플데이터/라벨링데이터 포함)
        emotions   : 사용할 감정 목록 (None → EMOTIONS 7개)
        split      : 'train' | 'val' | 'test' | 'all'
        train_ratio: 학습 비율
        val_ratio  : 검증 비율 (나머지는 test)
        augment    : 학습 증강 여부
        use_edge   : 엣지 채널 추가 (4ch)
        seed       : random seed
    """

    def __init__(
        self,
        data_root: str,
        emotions: list = None,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        augment: bool = True,
        use_edge: bool = False,
        seed: int = 42,
    ):
        self.data_root   = data_root
        self.emotions    = emotions or EMOTIONS
        self.split       = split
        self.use_edge    = use_edge
        self.image_size  = 224

        self.samples = self._build_samples(train_ratio, val_ratio, seed)

        aug_transforms = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
        base_transforms = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
        self.transform = transforms.Compose(
            aug_transforms if split == 'train' and augment else base_transforms
        )

    def _build_samples(self, train_ratio, val_ratio, seed):
        all_samples = []
        img_base = os.path.join(self.data_root, '샘플데이터')
        for emotion in self.emotions:
            img_dir = os.path.join(img_base, emotion)
            if not os.path.isdir(img_dir):
                print(f'  [{emotion}] 이미지 없음 → 건너뜀')
                continue
            files = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            for fn in files:
                all_samples.append({
                    'image_path': os.path.join(img_dir, fn),
                    'label':      self.emotions.index(emotion),
                    'emotion':    emotion,
                })
            print(f'  [{emotion}] {len(files)}장 로드')

        rng = np.random.RandomState(seed)
        # 클래스별로 분할 (stratified)
        train, val, test = [], [], []
        for emotion in self.emotions:
            cls = [s for s in all_samples if s['emotion'] == emotion]
            idx = np.arange(len(cls))
            rng.shuffle(idx)
            n_train = int(len(idx) * train_ratio)
            n_val   = int(len(idx) * val_ratio)
            train += [cls[i] for i in idx[:n_train]]
            val   += [cls[i] for i in idx[n_train:n_train + n_val]]
            test  += [cls[i] for i in idx[n_train + n_val:]]

        return {'train': train, 'val': val, 'test': test, 'all': all_samples}[self.split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        with open(item['image_path'], 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"이미지 읽기 실패: {item['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rgb_tensor = self.transform(img)

        if self.use_edge:
            edge = extract_edge(img).astype(np.float32) / 255.0
            edge_tensor = torch.from_numpy(edge).unsqueeze(0)
            return torch.cat([rgb_tensor, edge_tensor], dim=0), item['label']

        return rgb_tensor, item['label']

    def class_counts(self) -> dict:
        from collections import Counter
        return dict(Counter(s['emotion'] for s in self.samples))


class AiHubDataset(Dataset):
    """
    AI Hub 한국인 감정인식 데이터셋 (aihub/prepare.py로 전처리 완료된 데이터 사용).

    전처리 후 구조:
        {data_root}/train/{emotion}/*.jpg
        {data_root}/val/{emotion}/*.jpg

    Args:
        data_root      : prepare.py --out 경로 (예: G:/aihub_cropped)
        emotions       : 사용할 감정 목록 (None → EMOTIONS 7개)
        split          : 'train' | 'val' | 'test'
                         ※ val/test는 val 폴더를 val_test_ratio로 분리
        val_test_ratio : val 폴더를 검증:테스트로 나누는 비율 (기본 0.5)
        max_per_class  : 클래스당 최대 샘플 수 (None=전체)
        augment        : 학습 증강 여부
        use_edge       : 엣지 채널 추가 (4ch)
        seed           : random seed
    """

    def __init__(
        self,
        data_root: str,
        emotions: list = None,
        split: str = 'train',
        val_test_ratio: float = 0.5,
        max_per_class: int = None,
        augment: bool = True,
        use_edge: bool = False,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.emotions  = emotions or EMOTIONS
        self.split     = split
        self.use_edge  = use_edge

        self.samples = self._build_samples(val_test_ratio, max_per_class, seed)

        aug_transforms = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
        base_transforms = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ]
        self.transform = transforms.Compose(
            aug_transforms if split == 'train' and augment else base_transforms
        )

    def _build_samples(self, val_test_ratio, max_per_class, seed):
        rng = np.random.RandomState(seed)
        folder = 'train' if self.split == 'train' else 'val'

        all_samples = []
        for emotion in self.emotions:
            img_dir = os.path.join(self.data_root, folder, emotion)
            if not os.path.isdir(img_dir):
                print(f'  [{emotion}] 없음 ({img_dir})')
                continue
            files = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            label = self.emotions.index(emotion)
            for fn in files:
                all_samples.append({
                    'image_path': os.path.join(img_dir, fn),
                    'label':      label,
                    'emotion':    emotion,
                })
            print(f'  [{emotion}] {len(files)}장')

        result = []
        for emotion in self.emotions:
            cls = [s for s in all_samples if s['emotion'] == emotion]
            idx = np.arange(len(cls))
            rng.shuffle(idx)
            if max_per_class:
                idx = idx[:max_per_class]
            cls = [cls[i] for i in idx]

            if self.split == 'train':
                result.extend(cls)
            else:
                n_val = int(len(cls) * val_test_ratio)
                if self.split == 'val':
                    result.extend(cls[:n_val])
                elif self.split == 'test':
                    result.extend(cls[n_val:])

        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = cv2.imread(item['image_path'])
        if img is None:
            raise IOError(f"이미지 읽기 실패: {item['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_tensor = self.transform(img)

        if self.use_edge:
            edge = extract_edge(img).astype(np.float32) / 255.0
            edge_tensor = torch.from_numpy(edge).unsqueeze(0)
            return torch.cat([rgb_tensor, edge_tensor], dim=0), item['label']

        return rgb_tensor, item['label']

    def class_counts(self) -> dict:
        from collections import Counter
        return dict(Counter(s['emotion'] for s in self.samples))
