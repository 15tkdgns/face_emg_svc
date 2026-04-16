import { useState } from 'react'
import AnalyzeTab      from './components/AnalyzeTab'
import ModelCompareTab from './components/ModelCompareTab'
import PipelineTab     from './components/PipelineTab'
import RealtimeTab     from './components/RealtimeTab'
import CustomModelTab  from './components/CustomModelTab'

const TABS = [
  {
    id: 'realtime',
    label: '실시간',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
        <circle cx="12" cy="13" r="4"/>
      </svg>
    ),
  },
  {
    id: 'analyze',
    label: '분석',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
        <circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
      </svg>
    ),
  },
  {
    id: 'models',
    label: '성능',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
        <path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>
      </svg>
    ),
  },
  {
    id: 'pipeline',
    label: '파이프라인',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
    ),
  },
  {
    id: 'custom',
    label: '내 모델',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
        <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
        <line x1="12" y1="22.08" x2="12" y2="12"/>
      </svg>
    ),
  },
]

const TAB_HEADER = {
  realtime: { title: 'Face-Actor', sub: 'Real-time Detection' },
  analyze:  { title: 'Face-Actor', sub: 'Face Emotion Recognition' },
  models:   { title: 'Face-Actor', sub: 'Performance Benchmark' },
  pipeline: { title: 'Face-Actor', sub: 'Data Pipeline' },
  custom:   { title: 'Face-Actor', sub: 'My Model Test' },
}

export default function App() {
  const [tab, setTab] = useState('realtime')

  const header = TAB_HEADER[tab]

  return (
    <div className="flex flex-col h-[100dvh] max-w-[480px] mx-auto relative overflow-hidden animated-bg">

      {/* Header */}
      <header className="relative z-10 px-5 pt-4 pb-3 flex items-center justify-between border-b border-white/[0.06]">
        <div>
          <h1 className="text-xl font-extrabold text-white tracking-tight">
            {header.title}
          </h1>
          <p className="text-[10px] text-white/30 font-medium mt-0.5 tracking-wide uppercase">
            {header.sub}
          </p>
        </div>
        <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center text-black text-xs font-bold">
          FA
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden pb-24 hide-scrollbar relative z-10">
        {tab === 'realtime'  && <RealtimeTab />}
        {tab === 'analyze'   && <AnalyzeTab />}
        {tab === 'models'    && <ModelCompareTab />}
        {tab === 'pipeline'  && <PipelineTab />}
        {tab === 'custom'    && <CustomModelTab />}
      </main>

      {/* Bottom Nav */}
      <nav className="fixed bottom-0 left-1/2 -translate-x-1/2 w-full max-w-[480px] z-[100] px-4 pb-[max(env(safe-area-inset-bottom),8px)] pt-2">
        <div className="rounded-2xl flex items-stretch h-14 bg-[#111111] border border-white/[0.08]">
          {TABS.map(t => {
            const active = tab === t.id
            return (
              <button
                key={t.id}
                className={`flex-1 flex flex-col items-center justify-center gap-0.5 bg-transparent border-none cursor-pointer text-[10px] font-semibold transition-all duration-200 relative ${
                  active ? 'text-white' : 'text-white/25 hover:text-white/50'
                }`}
                onClick={() => setTab(t.id)}
              >
                {active && (
                  <div className="absolute -top-0.5 w-6 h-0.5 rounded-full bg-white" />
                )}
                <div className={`transition-transform duration-200 ${active ? 'scale-110' : ''}`}>
                  {t.icon}
                </div>
                <span>{t.label}</span>
              </button>
            )
          })}
        </div>
      </nav>
    </div>
  )
}
