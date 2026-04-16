"""
Vercel Python 서버리스 진입점.
app 변수를 모듈 최상위에 선언해야 @vercel/python이 인식함.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 모듈 최상위에 app 선언 (Vercel 정적 분석 요구사항)
app = FastAPI(title='Face EMG Emotion API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

try:
    # 전체 서버 로드 (onnxruntime + 모델)
    from server.main import app as _full_app
    app = _full_app
except Exception as _e:
    import logging
    logging.getLogger(__name__).warning(f'Full server unavailable: {_e}')

    @app.get('/api/health')
    def health():
        return {'status': 'unavailable', 'reason': str(_e)}

    @app.get('/api/models')
    def models():
        return {'models': []}
