import './ProgressBar.css'

export default function ProgressBar({ current, total, color }) {
  const pct = total > 0 ? (current / total) * 100 : 0

  return (
    <div className="progress-bar-container">
      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="progress-bar-label">{current} / {total}</div>
    </div>
  )
}
