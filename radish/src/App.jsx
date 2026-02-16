import { useState, useRef, useCallback } from 'react'
import { useItems } from './hooks/useItems'
import SwipeCard from './components/SwipeCard'
import ActionButtons from './components/ActionButtons'
import ProgressBar from './components/ProgressBar'
import LocationHeader from './components/LocationHeader'
import ShoppingList from './components/ShoppingList'

const LOCATION_COLORS = {
  'Produce Area': '#27ae60',
  'Shelves and Counter': '#e67e22',
  'Big Cabinet': '#8e44ad',
  'Pull-out Cabinet': '#a35db5',
  'Big Fridge': '#2980b9',
  'Leftovers Fridge': '#2471a3',
  'Big Freezer': '#5b6abf',
  'Leftovers Freezer': '#7986cb',
}

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
  const [swipeHistory, setSwipeHistory] = useState([])
  const cardRef = useRef(null)

  const handleSwipe = useCallback((direction) => {
    if (currentIndex >= items.length) return
    const item = items[currentIndex]
    const wasOutOfStock = direction === 'left'
    setSwipeHistory((prev) => [...prev, { direction, wasOutOfStock }])
    if (wasOutOfStock) {
      setOutOfStock((prev) => [...prev, item])
    }
    setCurrentIndex((prev) => prev + 1)
  }, [currentIndex, items])

  const swipe = useCallback((direction) => {
    if (cardRef.current) {
      cardRef.current.swipe(direction)
    }
  }, [])

  const handleUndo = useCallback(() => {
    if (currentIndex === 0) return
    const lastAction = swipeHistory[swipeHistory.length - 1]
    setSwipeHistory((prev) => prev.slice(0, -1))
    if (lastAction?.wasOutOfStock) {
      setOutOfStock((prev) => prev.slice(0, -1))
    }
    setCurrentIndex((prev) => prev - 1)
  }, [currentIndex, swipeHistory])

  const handleStartOver = () => {
    setMode(null)
    setCurrentIndex(0)
    setOutOfStock([])
    setSwipeHistory([])
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
  const locationColor = LOCATION_COLORS[currentItem?.location] || '#888'

  return (
    <div className="app">
      <div className="top-bar">
        <button
          className="undo-btn"
          onClick={handleUndo}
          disabled={currentIndex === 0}
          aria-label="Undo"
        >
          ↩ Back
        </button>
      </div>
      <ProgressBar current={currentIndex} total={items.length} color={locationColor} />
      {isNewLocation && <LocationHeader location={currentItem.location} color={locationColor} />}
      <div className="card-area">
        <SwipeCard
          key={currentIndex}
          ref={cardRef}
          item={currentItem}
          locationColor={locationColor}
          onSwipe={handleSwipe}
        />
      </div>
      <ActionButtons onSwipe={swipe} onUndo={handleUndo} canUndo={currentIndex > 0} />
    </div>
  )
}
