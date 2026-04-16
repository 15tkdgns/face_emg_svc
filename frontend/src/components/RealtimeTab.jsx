import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../api'

// ── 감정 정의 (백엔드 7클래스와 매핑) ─────────────────────────────
// 백엔드 EMOTIONS_7 = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
const EMOTIONS = [
  { key: '기쁨', label: '기쁨' },
  { key: '당황', label: '당황' },
  { key: '분노', label: '분노' },
  { key: '불안', label: '불안' },
  { key: '상처', label: '상처' },
  { key: '슬픔', label: '슬픔' },
  { key: '중립', label: '중립' },
]

const ZERO_SCORES = EMOTIONS.reduce((a, e) => ({ ...a, [e.key]: 0 }), {})

// ── 개별 감정 Progress Bar ────────────────────────────────────────
function EmotionBar({ emotion, value, rank }) {
  const isDominant = rank <= 1
  const pct = Math.round(value * 100)

  return (
    <div className="flex items-center gap-3 py-1.5">
      <span className="w-8 text-xs font-medium text-white/50 shrink-0">{emotion.label}</span>

      <div className="flex-1 min-w-0">
        <div className="h-2 w-full rounded-full overflow-hidden bg-white/[0.06]">
          <motion.div
            className="h-full rounded-full bg-white"
            initial={{ width: 0 }}
            animate={{ width: `${pct}%`, opacity: isDominant ? 1 : 0.35 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          />
        </div>
      </div>

      <motion.span
        className="w-10 text-right text-[11px] tabular-nums text-white/40 shrink-0"
        key={pct}
        initial={{ opacity: 0.5 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        {pct}%
      </motion.span>
    </div>
  )
}

// ── 메인 컴포넌트 ─────────────────────────────────────────────────
export default function RealtimeTab() {
  const [cameraOn, setCameraOn]       = useState(false)
  const [scores, setScores]           = useState(ZERO_SCORES)
  const [inferMs, setInferMs]         = useState(null)
  const [error, setError]             = useState(null)
  const [modelInfo, setModelInfo]     = useState(null)
  const [selectedModel, setSelectedModel] = useState('resnet18')
  const [availableModels, setAvailableModels] = useState([])
  const [showModelPicker, setShowModelPicker] = useState(false)
  const selectedModelRef = useRef('resnet18')

  const intervalRef  = useRef(null)
  const videoRef     = useRef(null)
  const streamRef    = useRef(null)
  const canvasRef    = useRef(document.createElement('canvas'))
  const processingRef = useRef(false)

  // 로드 가능한 모델 목록 가져오기
  useEffect(() => {
    api.models().then(res => {
      const all = res.data.models || []
      setAvailableModels(all)
      const loaded = all.filter(m => m.loaded)
      if (loaded.length > 0 && !all.find(m => m.id === selectedModel && m.loaded)) {
        setSelectedModel(loaded[0].id)
        selectedModelRef.current = loaded[0].id
      }
    }).catch(() => {})
  }, [])

  // selectedModel 변경 시 ref도 동기화
  useEffect(() => {
    selectedModelRef.current = selectedModel
  }, [selectedModel])

  // 순위 계산
  const sorted = [...EMOTIONS]
    .map(e => ({ key: e.key, value: scores[e.key] ?? 0 }))
    .sort((a, b) => b.value - a.value)
  const rankMap = {}
  sorted.forEach((s, i) => { rankMap[s.key] = i })

  const topEmotion = sorted[0]
  const topEmotionDef = EMOTIONS.find(e => e.key === topEmotion?.key)

  // ── 프레임 캡처 → API 호출 ──────────────────────────────────────
  const captureAndAnalyze = useCallback(async () => {
    if (processingRef.current) return // 이전 요청 아직 진행 중
    const video = videoRef.current
    if (!video || video.readyState < 2) return

    processingRef.current = true
    try {
      const canvas = canvasRef.current
      canvas.width  = video.videoWidth
      canvas.height = video.videoHeight
      canvas.getContext('2d').drawImage(video, 0, 0)

      const imageB64 = canvas.toDataURL('image/jpeg', 0.7)
      const res = await api.analyzeBase64(imageB64, selectedModelRef.current)
      const data = res.data

      // scores는 { '기쁨': 0.95, '당황': 0.02, ... } 형태
      if (data.scores) {
        setScores(data.scores)
        setInferMs(data.infer_ms)
        setError(null)
        if (!modelInfo) {
          setModelInfo({
            model: data.model_id,
            classes: data.num_classes,
          })
        }
      }
    } catch (e) {
      if (e?.code === 'ERR_NETWORK') {
        setError('백엔드 서버에 연결할 수 없습니다')
      }
      console.error('[realtime] analyze error:', e)
    } finally {
      processingRef.current = false
    }
  }, [modelInfo])

  // 카메라 시작
  const startCamera = useCallback(async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      })
      streamRef.current = stream
      setCameraOn(true)
    } catch (e) {
      setError('카메라 접근 권한이 필요합니다')
      console.error('[realtime] camera error:', e)
    }
  }, [])

  // cameraOn 상태 변경 시 video에 stream 연결
  useEffect(() => {
    if (cameraOn && streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current
      videoRef.current.play().catch(() => {})

      // 1초마다 프레임 캡처 → AI 분석
      intervalRef.current = setInterval(() => {
        captureAndAnalyze()
      }, 1000)
    }
    return () => clearInterval(intervalRef.current)
  }, [cameraOn, captureAndAnalyze])

  // 카메라 정지
  const stopCamera = useCallback(() => {
    setCameraOn(false)
    clearInterval(intervalRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    setScores(ZERO_SCORES)
    setInferMs(null)
  }, [])

  useEffect(() => {
    return () => {
      clearInterval(intervalRef.current)
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [])

  return (
    <div className="px-4 py-5 flex flex-col items-center">
      <div className="w-full max-w-md rounded-2xl overflow-hidden border border-white/[0.08] bg-white/[0.03]">

        {/* Header */}
        <div className="px-5 pt-5 pb-3 flex items-center justify-between border-b border-white/[0.06]">
          <div>
            <h2 className="text-sm font-bold text-white tracking-tight">실시간 감정 분석</h2>
            {modelInfo && (
              <p className="text-[10px] text-white/30 mt-0.5 font-mono">
                {modelInfo.model} · {modelInfo.classes}cls
              </p>
            )}
          </div>
          <AnimatePresence>
            {cameraOn && topEmotionDef && topEmotion.value > 0.05 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-sm font-bold text-white"
              >
                {topEmotionDef.label}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Webcam Area */}
        <div className="px-4 pt-4 pb-3">
          <div className="relative aspect-[4/3] rounded-xl overflow-hidden bg-black">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`absolute inset-0 w-full h-full object-cover -scale-x-100 ${
                cameraOn ? 'block' : 'hidden'
              }`}
            />

            {cameraOn ? (
              <>
                <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-black/60 backdrop-blur-sm rounded-full px-2.5 py-1 z-10">
                  <motion.div
                    className="w-1.5 h-1.5 rounded-full bg-white"
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                  />
                  <span className="text-white text-[10px] font-bold tracking-wider">LIVE</span>
                </div>
                {inferMs && (
                  <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm rounded-full px-2.5 py-1 z-10">
                    <span className="text-white/60 text-[10px] font-mono">{inferMs}ms</span>
                  </div>
                )}
              </>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <svg className="w-10 h-10 text-white/10 mb-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                  <circle cx="12" cy="13" r="4"/>
                </svg>
                <p className="text-white/20 text-xs">카메라 꺼짐</p>
              </div>
            )}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mx-5 mb-3 rounded-xl bg-white/[0.04] border border-white/10 p-3">
            <p className="text-white/60 text-xs">{error}</p>
          </div>
        )}

        {/* Emotion Bars */}
        <div className="px-5 pb-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[10px] font-bold text-white/20 uppercase tracking-widest">Emotion</span>
            {cameraOn && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-1">
                <div className="w-1 h-1 rounded-full bg-white/40 animate-pulse" />
                <span className="text-[9px] text-white/30">분석중</span>
              </motion.div>
            )}
          </div>
          <div className="space-y-1">
            {EMOTIONS.map(emotion => (
              <EmotionBar
                key={emotion.key}
                emotion={emotion}
                value={scores[emotion.key] ?? 0}
                rank={rankMap[emotion.key]}
              />
            ))}
          </div>
        </div>

        {/* Model Selector */}
        {availableModels.length > 0 && (
          <div className="px-5 pb-4">
            <label className="text-[10px] font-bold text-white/20 uppercase tracking-widest block mb-2">Model</label>
            <select
              value={selectedModel}
              onChange={(e) => { setSelectedModel(e.target.value); setModelInfo(null) }}
              className="w-full appearance-none bg-white/[0.04] border border-white/[0.08] rounded-xl px-4 py-2.5 text-sm font-medium text-white focus:outline-none focus:border-white/20 transition-all cursor-pointer"
            >
              {availableModels.map(m => (
                <option key={m.id} value={m.id} disabled={!m.loaded}>
                  {m.label}{!m.loaded ? ' (미로드)' : ''}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Camera Toggle */}
        <div className="px-5 pb-5">
          <button
            onClick={cameraOn ? stopCamera : startCamera}
            className={`w-full py-3.5 rounded-xl font-bold text-sm transition-all duration-200 flex items-center justify-center gap-2 active:scale-[0.98] ${
              cameraOn
                ? 'bg-white/[0.06] text-white/60 hover:bg-white/[0.09] border border-white/10'
                : 'bg-white text-black hover:bg-white/90'
            }`}
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            {cameraOn ? '카메라 끄기' : '카메라 켜기'}
          </button>
        </div>
      </div>
    </div>
  )
}
