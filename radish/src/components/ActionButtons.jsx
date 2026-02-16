import './ActionButtons.css'

export default function ActionButtons({ onSwipe }) {
  return (
    <div className="action-buttons">
      <button
        className="action-btn action-btn-no"
        onClick={() => onSwipe('left')}
        aria-label="Out of stock"
      >
        ✕
      </button>
      <button
        className="action-btn action-btn-yes"
        onClick={() => onSwipe('right')}
        aria-label="In stock"
      >
        ✓
      </button>
    </div>
  )
}
