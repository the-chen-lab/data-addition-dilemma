import { useState, useRef, useCallback } from 'react'
import { useItems } from './hooks/useItems'
import SwipeCard from './components/SwipeCard'
import ActionButtons from './components/ActionButtons'
import ProgressBar from './components/ProgressBar'
import LocationHeader from './components/LocationHeader'
import ShoppingList from './components/ShoppingList'

function ModeSelect({ onSelect }) {
  return (
    <div className="mode-select">
      <h1 className="mode-title">Radish Food Inventory</h1>
      <div className="mode-buttons">
        <button className="mode-btn" onClick={() => onSelect('short')}>
          Quick Check
        </button>
        <button className="mode-btn" onClick={() => onSelect('long')}>
          Full Check
        </button>
      </div>
    </div>
  )
}

export default function App() {
  const [mode, setMode] = useState(null)
  const { items, loading, error } = useItems(mode)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [outOfStock, setOutOfStock] = useState([])
  const cardRef = useRef(null)

  const handleSwipe = useCallback((direction) => {
    if (currentIndex >= items.length) return
    const item = items[currentIndex]
    if (direction === 'left') {
      setOutOfStock((prev) => [...prev, item])
    }
    setCurrentIndex((prev) => prev + 1)
  }, [currentIndex, items])

  const swipe = useCallback((direction) => {
    if (cardRef.current) {
      cardRef.current.swipe(direction)
    }
  }, [])

  const handleStartOver = () => {
    setMode(null)
    setCurrentIndex(0)
    setOutOfStock([])
  }

  if (!mode) {
    return (
      <div className="app">
        <ModeSelect onSelect={setMode} />
      </div>
    )
  }

  if (loading) {
    return (
      <div className="app">
        <div className="loading">Loading items...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app">
        <div className="error">
          <p>Failed to load items</p>
          <p className="error-detail">{error}</p>
        </div>
      </div>
    )
  }

  const done = currentIndex >= items.length && items.length > 0

  if (done) {
    return (
      <div className="app">
        <ShoppingList items={outOfStock} onStartOver={handleStartOver} />
      </div>
    )
  }

  const currentItem = items[currentIndex]
  const prevItem = items[currentIndex - 1]
  const isNewLocation = currentItem && (!prevItem || prevItem.location !== currentItem.location)

  return (
    <div className="app">
      <ProgressBar current={currentIndex} total={items.length} />
      {isNewLocation && <LocationHeader location={currentItem.location} />}
      <div className="card-area">
        <SwipeCard
          key={currentIndex}
          ref={cardRef}
          item={currentItem}
          onSwipe={handleSwipe}
        />
      </div>
      <ActionButtons onSwipe={swipe} />
    </div>
  )
}
