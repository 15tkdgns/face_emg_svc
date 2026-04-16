"""
Vercel Python 서버리스 진입점.

Vercel은 api/ 디렉토리의 Python 파일을 서버리스 함수로 실행.
FastAPI ASGI 앱을 그대로 노출.

주의: PyTorch 모델(.pth)은 Vercel 250MB 제한으로 직접 배포 불가.
      ML 추론은 외부 백엔드(Railway, Render 등)에서 운영하고
      VITE_API_BASE 환경변수로 프론트엔드와 연결하세요.
"""
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from server.main import app  # noqa: F401
except ImportError as e:
    # torch/cv2 등 ML 라이브러리가 없을 때 경량 앱으로 폴백
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title='Face EMG API (stub)')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

    @app.get('/api/health')
    def health():
        return {'status': 'unavailable', 'reason': f'ML libs not installed on this runtime: {e}'}

    @app.get('/api/models')
    def models():
        return {'models': []}
