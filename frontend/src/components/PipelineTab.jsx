import { useState } from 'react'
import { api } from '../api'

import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

const STEPS = [
  {
    icon: '📂', color: '#60A5FA', bg: 'from-blue-500/15 to-blue-500/5',
    title: '이미지 입력',
    desc: 'AI Hub 한국인 감정인식 데이터셋. 7개 감정 클래스, ~488K 이미지.',
    tag: 'Dataset',
  },
  {
    icon: '✂️', color: '#FBBF24', bg: 'from-amber-500/15 to-amber-500/5',
    title: '얼굴 크롭',
    desc: '3인 어노테이터 bounding box 평균 좌표 + 10% 패딩 적용.',
    tag: 'Preprocessing',
  },
  {
    icon: '🌅', color: '#34D399', bg: 'from-emerald-500/15 to-emerald-500/5',
    title: 'CLAHE 평활화',
    desc: 'LAB L채널 CLAHE. clipLimit=2.0, tileGrid=8×8.',
    tag: 'Optional',
  },
  {
    icon: '🔲', color: '#C084FC', bg: 'from-purple-500/15 to-purple-500/5',
    title: '엣지 채널',
    desc: 'Canny(50,150) → RGB+Edge 4채널. Conv 3→4ch 확장.',
    tag: 'Optional',
  },
  {
    icon: '🔀', color: '#FB923C', bg: 'from-orange-500/15 to-orange-500/5',
    title: '데이터 증강',
    desc: 'HFlip + ColorJitter(±0.3) + Rotation(±15°).',
    tag: 'Training',
  },
  {
    icon: '🧠', color: '#F87171', bg: 'from-red-500/15 to-red-500/5',
    title: '모델 추론',
    desc: 'ImageNet pretrained DenseNet/EfficientNet. AdamW + CosineAnnealing.',
    tag: 'Inference',
  },
  {
    icon: '📊', color: '#60A5FA', bg: 'from-sky-500/15 to-sky-500/5',
    title: '결과 출력',
    desc: '4클래스 Softmax. Best: DenseNet121 87.6% Acc.',
    tag: 'Output',
  },
]

const VIZ_ITEMS = [
  { key: 'edge_samples', label: '엣지맵', emoji: '🔲', desc: '원본 vs Canny vs Sobel-X vs Sobel-Y 비교' },
  { key: 'gradcam_samples', label: 'Grad-CAM', emoji: '🔥', desc: '역전파 활성화 히트맵 — 모델의 시선' },
  { key: 'class_gradcam', label: '클래스 CAM', emoji: '🎯', desc: '감정별 평균 Grad-CAM 영역 비교' },
  { key: 'tsne', label: 't-SNE', emoji: '🌐', desc: 'Classifier 피처 2D 투영 클러스터링' },
  { key: 'comparison', label: '모델 비교', emoji: '📈', desc: '4개 모델 Accuracy + F1 비교 차트' },
]

function StepList() {
  return (
    <div className="relative">
      {/* Timeline line */}
      <div className="absolute left-[19px] top-8 bottom-8 w-px bg-gradient-to-b from-[#7C65F6]/30 via-[#7C65F6]/10 to-transparent" />
      
      <div className="space-y-1">
        {STEPS.map((s, i) => (
          <div 
            key={i} 
            className="flex items-start gap-4 group relative animate-slide-up"
            style={{ animationDelay: `${i * 60}ms` }}
          >
            {/* Node */}
            <div className="relative z-10 shrink-0">
              <div 
                className={`w-10 h-10 rounded-xl flex items-center justify-center text-base bg-gradient-to-br ${s.bg} border border-white/[0.06] group-hover:border-white/10 transition-all duration-300 group-hover:scale-110 group-hover:shadow-lg`}
                style={{ boxShadow: `0 0 0 rgba(255,255,255,0)` }}
              >
                {s.icon}
              </div>
            </div>
            
            {/* Content */}
            <div className="flex-1 pb-5 pt-1">
              <div className="flex items-center gap-2 mb-1">
                <h4 className="text-sm font-bold text-foreground/90">{s.title}</h4>
                <Badge className="bg-white/[0.04] text-muted-foreground/50 border-white/[0.06] text-[8px] px-1.5 py-0 font-mono uppercase">{s.tag}</Badge>
              </div>
              <p className="text-xs text-muted-foreground/45 leading-relaxed">{s.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function VizViewer() {
  const [selected, setSelected] = useState('edge_samples')
  const [imgError, setImgError] = useState(false)

  const current = VIZ_ITEMS.find(v => v.key === selected)
  const url = api.pipelineImageUrl(selected)

  return (
    <div className="space-y-4 animate-slide-up">
      <div className="grid grid-cols-2 gap-2">
        {VIZ_ITEMS.map(v => (
          <button
            key={v.key}
            className={`py-3 px-3 rounded-xl text-xs font-semibold text-center transition-all duration-300 border ${
              selected === v.key
                ? 'glass border-[#7C65F6]/30 text-[#A78BFA] shadow-lg shadow-[#7C65F6]/10'
                : 'border-white/[0.04] text-muted-foreground/40 hover:text-muted-foreground/60 hover:border-white/[0.08] bg-white/[0.02]'
            }`}
            onClick={() => { setSelected(v.key); setImgError(false) }}
          >
            <span className="text-lg block mb-1">{v.emoji}</span>
            {v.label}
          </button>
        ))}
      </div>

      {current && (
        <p className="text-xs text-muted-foreground/40 leading-relaxed px-1 font-medium">
          {current.desc}
        </p>
      )}

      <Card className="glass overflow-hidden glow-neon">
        <div className="min-h-[200px] flex items-center justify-center bg-gradient-to-br from-white/[0.02] to-transparent">
          {imgError ? (
            <div className="py-12 text-center">
              <div className="text-5xl mb-4 opacity-20">🖼️</div>
              <p className="text-sm font-semibold text-muted-foreground/30">이미지 미생성</p>
              <p className="text-[10px] mt-1.5 text-muted-foreground/20">
                <code className="bg-white/[0.04] px-1.5 py-0.5 rounded text-[9px]">python visualize.py</code> 실행 후 생성됩니다
              </p>
            </div>
          ) : (
            <img
              key={url}
              src={url}
              alt={current?.label}
              className="w-full block"
              onError={() => setImgError(true)}
            />
          )}
        </div>
      </Card>
    </div>
  )
}

export default function PipelineTab() {
  const [view, setView] = useState('steps')

  return (
    <div className="px-5 py-4 space-y-5">
      <div className="glass rounded-2xl p-1 flex gap-1">
        {[
          { key: 'steps', label: '🔄 처리 단계' },
          { key: 'viz', label: '🖼️ 시각화 결과' },
        ].map(v => (
          <button
            key={v.key}
            className={`flex-1 py-2.5 text-xs font-semibold rounded-xl transition-all duration-300 ${
              view === v.key ? 'bg-[#7C65F6]/20 text-[#A78BFA]' : 'text-muted-foreground/50'
            }`}
            onClick={() => setView(v.key)}
          >
            {v.label}
          </button>
        ))}
      </div>

      {view === 'steps' && <StepList />}
      {view === 'viz'   && <VizViewer />}
    </div>
  )
}
