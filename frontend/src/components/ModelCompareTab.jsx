import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell,
} from 'recharts'
import { api } from '../api'

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

const EMOTIONS = ['기쁨', '당황', '분노', '상처']
const EMOTION_EMOJI = { 기쁨: '😄', 당황: '😳', 분노: '😡', 상처: '😢' }

function accColor(acc) {
  if (acc >= 0.87) return '#34D399'
  if (acc >= 0.83) return '#FBBF24'
  return '#F87171'
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-lg p-3 text-xs shadow-2xl border border-white/10">
      <p className="font-semibold mb-1.5 text-foreground">{label}</p>
      {payload.map(p => (
        <p key={p.name} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: p.fill }} />
          <span className="text-muted-foreground">{p.name}:</span>
          <span className="font-bold text-foreground">{(p.value * 100).toFixed(1)}%</span>
        </p>
      ))}
    </div>
  )
}

function shortLabel(label) {
  return label
    .replace('EfficientNet-B0', 'Eff-B0')
    .replace('DenseNet121', 'Dense121')
    .replace(' + CLAHE + Edge', '+CE')
}

export default function ModelCompareTab() {
  const [models, setModels]   = useState([])
  const [view, setView]       = useState('table')
  const [chartBy, setChartBy] = useState('acc')
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  useEffect(() => {
    api.models()
      .then(r => setModels(r.data.models))
      .catch(e => setError(
        e?.code === 'ERR_NETWORK'
          ? '서버에 연결할 수 없습니다.'
          : `오류: ${e?.response?.data?.detail ?? e?.message}`
      ))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-4">
        <div className="w-8 h-8 border-2 border-[#7C65F6]/30 border-t-[#7C65F6] rounded-full animate-spin" />
        <p className="text-xs text-muted-foreground/50 font-medium">Loading models...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="px-5 py-4">
        <div className="rounded-xl bg-red-500/10 border border-red-500/20 p-4">
          <p className="text-red-400 text-sm font-medium">{error}</p>
        </div>
      </div>
    )
  }

  const accData = models.map(m => ({
    name: shortLabel(m.label), 전체Acc: m.val_acc, color: m.color, loaded: m.loaded,
  }))

  const f1Data = EMOTIONS.map(emo => {
    const row = { name: `${EMOTION_EMOJI[emo]} ${emo}` }
    models.forEach(m => { row[shortLabel(m.label)] = m.f1_per[emo] ?? 0 })
    return row
  })

  return (
    <div className="px-5 py-4 space-y-5">
      {/* View Switcher */}
      <div className="glass rounded-2xl p-1 flex gap-1">
        {[
          { key: 'table', label: '📋 성능표' },
          { key: 'chart', label: '📊 차트' },
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

      {/* Table View */}
      {view === 'table' && (
        <Card className="glass overflow-hidden animate-slide-up">
          <CardHeader className="pb-3 pt-5 px-5">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-bold">모델 벤치마크</CardTitle>
              <Badge className="bg-[#7C65F6]/15 text-[#A78BFA] border-[#7C65F6]/20 text-[10px]">
                {models.length} Models
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="overflow-x-auto px-3 pb-4">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/[0.06]">
                  <th className="px-2 py-2.5 text-left font-semibold text-muted-foreground/60 text-[10px] uppercase tracking-wider">모델</th>
                  <th className="px-2 py-2.5 text-left font-semibold text-muted-foreground/60 text-[10px] uppercase tracking-wider">Acc</th>
                  {EMOTIONS.map(e => (
                    <th key={e} className="px-2 py-2.5 text-center font-semibold text-muted-foreground/60 text-[10px]">
                      {EMOTION_EMOJI[e]}
                    </th>
                  ))}
                  <th className="px-2 py-2.5 text-center font-semibold text-muted-foreground/60 text-[10px] uppercase tracking-wider">상태</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr key={m.id} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors animate-slide-up" style={{ animationDelay: `${i * 60}ms` }}>
                    <td className="px-2 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-6 rounded-full" style={{ background: `linear-gradient(180deg, ${m.color}, ${m.color}50)` }} />
                        <span className="text-[11px] font-semibold text-foreground/90">{m.label}</span>
                      </div>
                    </td>
                    <td className="px-2 py-3">
                      <span className="px-2 py-0.5 rounded-md text-[11px] font-black text-white" style={{ backgroundColor: accColor(m.val_acc) }}>
                        {(m.val_acc * 100).toFixed(1)}
                      </span>
                    </td>
                    {EMOTIONS.map(e => (
                      <td key={e} className="px-2 py-3 text-center text-muted-foreground/70 font-mono text-[10px]">
                        {m.f1_per[e]?.toFixed(2) ?? '-'}
                      </td>
                    ))}
                    <td className="px-2 py-3 text-center">
                      {m.loaded
                        ? <span className="w-2 h-2 inline-block rounded-full bg-emerald-400 shadow-sm shadow-emerald-400/50" />
                        : <span className="w-2 h-2 inline-block rounded-full bg-red-400/50" />}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[9px] text-muted-foreground/30 mt-3 font-medium">
              F1 scores on validation set (420 images). Color: 🟢 ≥87% 🟡 ≥83% 🔴 &lt;83%
            </p>
          </CardContent>
        </Card>
      )}

      {/* Chart View */}
      {view === 'chart' && (
        <Card className="glass overflow-hidden animate-slide-up">
          <CardContent className="pt-5 pb-4">
            <div className="glass rounded-xl p-1 flex gap-1 mb-5">
              <button
                className={`flex-1 py-1.5 text-[10px] font-semibold rounded-lg transition-all duration-300 ${chartBy === 'acc' ? 'bg-[#7C65F6]/20 text-[#A78BFA]' : 'text-muted-foreground/40'}`}
                onClick={() => setChartBy('acc')}
              >Overall Accuracy</button>
              <button
                className={`flex-1 py-1.5 text-[10px] font-semibold rounded-lg transition-all duration-300 ${chartBy === 'f1' ? 'bg-[#7C65F6]/20 text-[#A78BFA]' : 'text-muted-foreground/40'}`}
                onClick={() => setChartBy('f1')}
              >Per-Class F1</button>
            </div>

            {chartBy === 'acc' && (
              <>
                <p className="text-[10px] text-muted-foreground/40 mb-3 font-semibold uppercase tracking-wider">Validation Accuracy</p>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={accData} margin={{ top: 4, right: 8, left: -20, bottom: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="name" tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.3)' }} angle={-25} textAnchor="end" interval={0} />
                    <YAxis domain={[0.75, 1.0]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.3)' }} />
                    <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} contentStyle={{ background: 'rgba(15,15,35,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                    <Bar dataKey="전체Acc" radius={[6, 6, 0, 0]}>
                      {accData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}

            {chartBy === 'f1' && (
              <>
                <p className="text-[10px] text-muted-foreground/40 mb-3 font-semibold uppercase tracking-wider">F1-Score by Emotion</p>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={f1Data} margin={{ top: 4, right: 8, left: -20, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.4)' }} />
                    <YAxis domain={[0.6, 1.0]} tickFormatter={v => v.toFixed(1)} tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.3)' }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 9 }} />
                    {models.map(m => (
                      <Bar key={m.id} dataKey={shortLabel(m.label)} fill={m.color} radius={[4, 4, 0, 0]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* Model Cards */}
      <div>
        <h3 className="text-[10px] font-semibold text-muted-foreground/40 uppercase tracking-widest mb-3 px-1">Registered Models</h3>
        <div className="space-y-2">
          {models.map((m, i) => (
            <Card
              key={m.id}
              className="glass overflow-hidden hover:border-white/10 transition-all duration-300 group cursor-default animate-slide-up"
              style={{ animationDelay: `${i * 60}ms` }}
            >
              <CardContent className="p-3.5 flex items-center gap-3">
                <div className="w-1 h-10 rounded-full shrink-0 transition-all duration-300 group-hover:h-12" style={{ background: `linear-gradient(180deg, ${m.color}, ${m.color}40)` }} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-bold text-foreground/90 truncate">{m.label}</div>
                  <div className="text-[10px] text-muted-foreground/40 truncate">{m.description}</div>
                </div>
                <div className="text-right shrink-0">
                  <div className="text-xl font-black tabular-nums font-mono" style={{ color: accColor(m.val_acc) }}>
                    {(m.val_acc * 100).toFixed(1)}
                    <span className="text-xs font-semibold text-muted-foreground/40">%</span>
                  </div>
                  <div className="text-[9px] text-muted-foreground/30 font-medium">
                    {m.loaded ? '● Active' : '○ Offline'}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
