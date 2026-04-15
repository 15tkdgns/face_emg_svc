import { useRef, useState } from 'react'
import { api } from '../api'

const EMOTION_EMOJI  = { 기쁨: '😄', 당황: '😳', 분노: '😡', 상처: '😢', 불안: '😰', 슬픔: '😢', 중립: '😐' }
const EMOTION_COLOR  = { 기쁨: '#f59e0b', 당황: '#f97316', 분노: '#ef4444', 상처: '#6366f1', 불안: '#8b5cf6', 슬픔: '#3b82f6', 중립: '#6b7280' }

function ConfidenceBar({ emotion, score, highlight }) {
  const color = EMOTION_COLOR[emotion] || '#6366f1'
  return (
    <div className="conf-bar-row">
      <span className="conf-bar-label">{EMOTION_EMOJI[emotion] ?? '🤔'} {emotion}</span>
      <div className="conf-bar-track">
        <div
          className="conf-bar-fill"
          style={{ width: `${(score * 100).toFixed(1)}%`, background: color, opacity: highlight ? 1 : 0.45 }}
        />
      </div>
      <span className="conf-bar-pct">{(score * 100).toFixed(1)}%</span>
    </div>
  )
}

export default function CustomModelTab() {
  // ── 모델 업로드 상태 ──────────────────────────────────────────────
  const [modelFile, setModelFile]     = useState(null)
  const [uploading, setUploading]     = useState(false)
  const [modelInfo, setModelInfo]     = useState(null)   // 업로드 성공 시 서버 응답
  const [token, setToken]             = useState(null)
  const [uploadError, setUploadError] = useState(null)

  // ── 이미지 분석 상태 ──────────────────────────────────────────────
  const [imgFile, setImgFile]         = useState(null)
  const [preview, setPreview]         = useState(null)
  const [analyzing, setAnalyzing]     = useState(false)
  const [result, setResult]           = useState(null)
  const [analyzeError, setAnalyzeError] = useState(null)
  const [faceB64, setFaceB64]         = useState(null)
  const [faceDetected, setFaceDetected] = useState(null)

  const modelInputRef = useRef(null)
  const imgInputRef   = useRef(null)

  // ── 모델 파일 선택 ────────────────────────────────────────────────
  const onModelFileChange = (e) => {
    const f = e.target.files[0]
    if (!f) return
    setModelFile(f)
    setModelInfo(null)
    setToken(null)
    setUploadError(null)
    // 모델 변경 시 분석 결과도 초기화
    setResult(null)
    setFaceB64(null)
    setFaceDetected(null)
  }

  // ── 모델 업로드 ───────────────────────────────────────────────────
  const uploadModel = async () => {
    if (!modelFile) return
    setUploading(true)
    setUploadError(null)
    setModelInfo(null)
    setToken(null)
    try {
      const res = await api.uploadCustomModel(modelFile)
      setModelInfo(res.data)
      setToken(res.data.token)
    } catch (e) {
      const detail = e?.response?.data?.detail
      setUploadError(detail ?? '모델 업로드 실패. 서버 콘솔을 확인하세요.')
      console.error('[upload error]', e)
    } finally {
      setUploading(false)
    }
  }

  // ── 모델 삭제 ─────────────────────────────────────────────────────
  const removeModel = async () => {
    if (!token) return
    try { await api.deleteCustomModel(token) } catch (_) {}
    setToken(null)
    setModelInfo(null)
    setModelFile(null)
    setResult(null)
    setFaceB64(null)
    setFaceDetected(null)
    setUploadError(null)
  }

  // ── 이미지 선택 ───────────────────────────────────────────────────
  const onImgChange = (e) => {
    const f = e.target.files[0]
    if (!f) return
    setImgFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setFaceB64(null)
    setAnalyzeError(null)
  }

  // ── 분석 실행 ─────────────────────────────────────────────────────
  const analyze = async () => {
    if (!imgFile || !token) return
    setAnalyzing(true)
    setAnalyzeError(null)
    setResult(null)
    setFaceB64(null)
    try {
      const res = await api.analyzeCustomModel(imgFile, token)
      setResult(res.data)
      setFaceB64(res.data.face_b64)
      setFaceDetected(res.data.face_detected)
    } catch (e) {
      if (e?.response?.status === 404) {
        setAnalyzeError('토큰이 만료됐습니다. 모델을 다시 업로드하세요.')
        setToken(null)
        setModelInfo(null)
      } else {
        const detail = e?.response?.data?.detail
        setAnalyzeError(detail ?? `서버 오류 (${e?.response?.status ?? 'unknown'})`)
      }
      console.error('[custom analyze error]', e)
    } finally {
      setAnalyzing(false)
    }
  }

  const emotions = result?.emotions ?? (modelInfo?.emotions ?? [])

  return (
    <div>
      {/* ── 1. 모델 업로드 섹션 ──────────────────────────────────── */}
      <div className="section">
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 12 }}>
          직접 학습한 <code>.pth</code> 파일을 업로드하고 감정을 분석해보세요.<br />
          모델은 서버 메모리에만 유지되며 재시작 시 초기화됩니다.
        </p>

        {!token ? (
          <>
            <label className="btn btn-outline btn-full" style={{ cursor: 'pointer', marginBottom: 8 }}>
              {modelFile ? `📦 ${modelFile.name}` : '📦 .pth 파일 선택'}
              <input
                ref={modelInputRef}
                type="file"
                accept=".pth"
                style={{ display: 'none' }}
                onChange={onModelFileChange}
              />
            </label>

            <button
              className="btn btn-primary btn-full"
              disabled={!modelFile || uploading}
              onClick={uploadModel}
            >
              {uploading ? <><div className="spinner" /> 업로드 중...</> : '⬆️ 모델 업로드'}
            </button>
          </>
        ) : (
          <div style={{
            background: 'var(--surface)', border: '1.5px solid var(--border)',
            borderRadius: 12, padding: '14px 16px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <p style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>✅ 모델 로드 완료</p>
                <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>{modelInfo?.filename}</p>
              </div>
              <button
                className="btn btn-outline"
                style={{ padding: '4px 10px', fontSize: 12 }}
                onClick={removeModel}
              >
                제거
              </button>
            </div>
            <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {[
                ['백본', modelInfo?.backbone],
                ['클래스', modelInfo?.num_classes],
                ['채널', modelInfo?.in_channels],
                ['CLAHE', modelInfo?.use_clahe ? 'ON' : 'OFF'],
                ['Edge', modelInfo?.use_edge ? 'ON' : 'OFF'],
              ].map(([k, v]) => (
                <span key={k} style={{
                  background: 'var(--border)', borderRadius: 6,
                  padding: '2px 8px', fontSize: 11, color: 'var(--text-muted)',
                }}>
                  {k}: <strong style={{ color: 'var(--text)' }}>{v}</strong>
                </span>
              ))}
            </div>
            {modelInfo?.emotions && (
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
                감정 클래스: {modelInfo.emotions.join(' · ')}
              </p>
            )}
          </div>
        )}

        {uploadError && <div className="notice" style={{ marginTop: 8 }}>{uploadError}</div>}
      </div>

      {/* ── 2. 이미지 분석 섹션 (모델 로드 후 표시) ─────────────── */}
      {token && (
        <>
          <div className="section" style={{ paddingTop: 0 }}>
            <div className="preview-wrap">
              {preview
                ? <img src={preview} alt="미리보기" />
                : <span style={{ fontSize: 40 }}>🖼️</span>
              }
            </div>
          </div>

          <div className="section" style={{ paddingTop: 0 }}>
            <label className="btn btn-outline btn-full" style={{ cursor: 'pointer', marginBottom: 8 }}>
              {preview ? '🔄 다른 이미지 선택' : '📂 이미지 선택'}
              <input
                ref={imgInputRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={onImgChange}
              />
            </label>

            <button
              className="btn btn-primary btn-full"
              disabled={!imgFile || analyzing}
              onClick={analyze}
            >
              {analyzing ? <><div className="spinner" /> 분석 중...</> : '🔍 감정 분석'}
            </button>
          </div>

          {analyzeError && (
            <div className="section" style={{ paddingTop: 0 }}>
              <div className="notice">{analyzeError}</div>
            </div>
          )}

          {faceDetected !== null && (
            <div className="section" style={{ paddingTop: 0 }}>
              <div className={`notice${faceDetected ? ' info' : ''}`}>
                {faceDetected
                  ? '✅ 얼굴이 검출되었습니다'
                  : '⚠️ 얼굴 미검출 — 중앙 크롭으로 분석했습니다'}
              </div>
            </div>
          )}

          {faceB64 && (
            <div className="section" style={{ paddingTop: 0 }}>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>분석에 사용된 얼굴 영역</p>
              <img
                src={`data:image/jpeg;base64,${faceB64}`}
                alt="분석 얼굴"
                style={{ width: 100, height: 100, objectFit: 'cover', borderRadius: 12, border: '1px solid var(--border)' }}
              />
            </div>
          )}

          {result && (
            <div className="slide-up">
              <div style={{ textAlign: 'center', padding: '20px 0 16px' }}>
                <div style={{ fontSize: 56, lineHeight: 1.1, marginBottom: 8 }}>
                  {result.emoji ?? EMOTION_EMOJI[result.emotion] ?? '🤔'}
                </div>
                <div style={{ fontSize: 26, fontWeight: 800, color: EMOTION_COLOR[result.emotion] ?? '#6366f1' }}>
                  {result.emotion}
                </div>
                <div style={{ fontSize: 14, color: '#71717a', marginTop: 4 }}>
                  신뢰도 {(result.confidence * 100).toFixed(1)}% · {result.infer_ms}ms · {result.backbone}
                </div>
              </div>
              <div style={{ padding: '0 16px 16px' }}>
                {emotions.map(e => (
                  <ConfidenceBar
                    key={e}
                    emotion={e}
                    score={result.scores?.[e] ?? 0}
                    highlight={e === result.emotion}
                  />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
