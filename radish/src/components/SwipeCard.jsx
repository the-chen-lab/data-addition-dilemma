import { forwardRef, useImperativeHandle, useRef } from 'react'
import TinderCard from 'react-tinder-card'
import './SwipeCard.css'

const SwipeCard = forwardRef(function SwipeCard({ item, locationColor, onSwipe }, ref) {
  const tinderCardRef = useRef(null)

  useImperativeHandle(ref, () => ({
    swipe: (dir) => tinderCardRef.current?.swipe(dir),
  }))

  return (
    <TinderCard
      ref={tinderCardRef}
      className="swipe-card-wrapper"
      onSwipe={onSwipe}
      preventSwipe={['up', 'down']}
    >
      <div className="swipe-card">
        <div className="card-location" style={{ color: locationColor }}>{item.location}</div>
        <div className="card-name">{item.name}</div>
        <div className="card-source">{item.order_source}</div>
      </div>
    </TinderCard>
  )
})

export default SwipeCard
