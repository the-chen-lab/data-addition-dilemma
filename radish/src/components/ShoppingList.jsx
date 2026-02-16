import './ShoppingList.css'

export default function ShoppingList({ items, onStartOver }) {
  // Group items by order_source
  const grouped = {}
  for (const item of items) {
    const source = item.order_source
    if (!grouped[source]) grouped[source] = []
    grouped[source].push(item)
  }

  // Sort: specific stores alphabetically first, "Order from anywhere" last
  const sources = Object.keys(grouped).sort((a, b) => {
    const aGeneric = a === 'Order from anywhere'
    const bGeneric = b === 'Order from anywhere'
    if (aGeneric && !bGeneric) return 1
    if (!aGeneric && bGeneric) return -1
    return a.localeCompare(b)
  })

  return (
    <div className="shopping-list">
      <h1 className="shopping-title">Shopping List</h1>

      {items.length === 0 ? (
        <p className="shopping-empty">Everything is in stock!</p>
      ) : (
        sources.map((source) => (
          <div key={source} className="shopping-group">
            <h2 className="shopping-source">{source}</h2>
            <ul className="shopping-items">
              {grouped[source].map((item) => (
                <li key={item.id} className="shopping-item">
                  {item.name}
                </li>
              ))}
            </ul>
          </div>
        ))
      )}

      <button className="start-over-btn" onClick={onStartOver}>
        Start Over
      </button>
    </div>
  )
}
