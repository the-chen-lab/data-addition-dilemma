export default function LocationHeader({ location, color }) {
  return (
    <div className="location-header" style={{ color }}>
      Now checking: <strong>{location}</strong>
    </div>
  )
}
