export default function LocationHeader({ location }) {
  return (
    <div className="location-header">
      Now checking: <strong>{location}</strong>
    </div>
  )
}
