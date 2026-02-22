import { useState, useEffect } from 'react'
import { supabase } from './supabaseClient'
import './EditApp.css'

export default function EditApp() {
  const [items, setItems] = useState([])
  const [locations, setLocations] = useState([])
  const [loading, setLoading] = useState(true)
  const [editingId, setEditingId] = useState(null)
  const [editValues, setEditValues] = useState({})
  const [newItem, setNewItem] = useState({ name: '', location: '', order_source: 'Order from anywhere', short_list: false })
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState(null)

  useEffect(() => { fetchData() }, [])

  async function fetchData() {
    setLoading(true)
    const [itemsRes, locsRes] = await Promise.all([
      supabase.from('items').select('*, locations(location_order)'),
      supabase.from('locations').select('*').order('location_order'),
    ])
    if (locsRes.data) {
      setLocations(locsRes.data)
      setNewItem(prev => ({ ...prev, location: locsRes.data[0]?.location_name || '' }))
    }
    if (itemsRes.data) {
      setItems(sortItems(itemsRes.data))
    }
    setLoading(false)
  }

  function sortItems(data) {
    return [...data].sort((a, b) => {
      const orderA = a.locations?.location_order ?? 999
      const orderB = b.locations?.location_order ?? 999
      if (orderA !== orderB) return orderA - orderB
      return a.id - b.id
    })
  }

  async function handleDelete(id) {
    if (!window.confirm('Delete this item?')) return
    const { error } = await supabase.from('items').delete().eq('id', id)
    if (error) showMessage('Error: ' + error.message, 'error')
    else {
      setItems(prev => prev.filter(i => i.id !== id))
      showMessage('Deleted.')
    }
  }

  function startEdit(item) {
    setEditingId(item.id)
    setEditValues({ name: item.name, location: item.location, order_source: item.order_source, short_list: item.short_list })
  }

  function cancelEdit() {
    setEditingId(null)
    setEditValues({})
  }

  async function saveEdit(id) {
    if (!editValues.name.trim()) return
    setSaving(true)
    const { error } = await supabase.from('items').update({
      name: editValues.name,
      location: editValues.location,
      order_source: editValues.order_source,
      short_list: editValues.short_list,
    }).eq('id', id)
    if (error) {
      showMessage('Error: ' + error.message, 'error')
    } else {
      await fetchData()
      setEditingId(null)
      showMessage('Saved.')
    }
    setSaving(false)
  }

  async function handleAdd() {
    if (!newItem.name.trim() || !newItem.location) return
    setSaving(true)
    const { error } = await supabase.from('items').insert({
      name: newItem.name.trim(),
      location: newItem.location,
      order_source: newItem.order_source,
      short_list: newItem.short_list,
    })
    if (error) {
      showMessage('Error: ' + error.message, 'error')
    } else {
      await fetchData()
      setNewItem(prev => ({ ...prev, name: '', order_source: 'Order from anywhere', short_list: false }))
      showMessage('Added.')
    }
    setSaving(false)
  }

  function showMessage(text, type = 'ok') {
    setMessage({ text, type })
    setTimeout(() => setMessage(null), 3000)
  }

  // Group items by location in sorted order
  const locationNames = locations.map(l => l.location_name)
  const grouped = {}
  for (const loc of locationNames) grouped[loc] = []
  for (const item of items) {
    if (!grouped[item.location]) grouped[item.location] = []
    grouped[item.location].push(item)
  }

  return (
    <div className="edit-app">
      <header className="edit-header">
        <h1>Radish — Edit Items</h1>
        {message && <span className={`edit-message ${message.type}`}>{message.text}</span>}
      </header>

      {loading ? (
        <p className="edit-loading">Loading...</p>
      ) : (
        <>
          {locationNames.map(loc => (
            <section key={loc} className="location-section">
              <h2 className="location-name">{loc}</h2>
              <table className="items-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Source</th>
                    <th>Short list</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {grouped[loc].map(item => (
                    <tr key={item.id} className={editingId === item.id ? 'editing' : ''}>
                      {editingId === item.id ? (
                        <>
                          <td>
                            <input
                              className="edit-input"
                              value={editValues.name}
                              onChange={e => setEditValues(v => ({ ...v, name: e.target.value }))}
                              onKeyDown={e => { if (e.key === 'Enter') saveEdit(item.id); if (e.key === 'Escape') cancelEdit() }}
                              autoFocus
                            />
                          </td>
                          <td>
                            <input
                              className="edit-input"
                              value={editValues.order_source}
                              onChange={e => setEditValues(v => ({ ...v, order_source: e.target.value }))}
                              onKeyDown={e => { if (e.key === 'Enter') saveEdit(item.id); if (e.key === 'Escape') cancelEdit() }}
                            />
                          </td>
                          <td className="center">
                            <input
                              type="checkbox"
                              checked={editValues.short_list}
                              onChange={e => setEditValues(v => ({ ...v, short_list: e.target.checked }))}
                            />
                          </td>
                          <td className="actions">
                            <button className="btn save-btn" onClick={() => saveEdit(item.id)} disabled={saving}>Save</button>
                            <button className="btn cancel-btn" onClick={cancelEdit}>Cancel</button>
                          </td>
                        </>
                      ) : (
                        <>
                          <td>{item.name}</td>
                          <td className="source">{item.order_source}</td>
                          <td className="center">{item.short_list ? '✓' : ''}</td>
                          <td className="actions">
                            <button className="btn edit-btn" onClick={() => startEdit(item)}>Edit</button>
                            <button className="btn delete-btn" onClick={() => handleDelete(item.id)}>Delete</button>
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          ))}

          <section className="add-section">
            <h2>Add item</h2>
            <div className="add-form">
              <input
                className="edit-input"
                placeholder="Name"
                value={newItem.name}
                onChange={e => setNewItem(v => ({ ...v, name: e.target.value }))}
                onKeyDown={e => { if (e.key === 'Enter') handleAdd() }}
              />
              <select
                className="edit-select"
                value={newItem.location}
                onChange={e => setNewItem(v => ({ ...v, location: e.target.value }))}
              >
                {locations.map(l => (
                  <option key={l.location_name} value={l.location_name}>{l.location_name}</option>
                ))}
              </select>
              <input
                className="edit-input"
                placeholder="Source (e.g. H Mart)"
                value={newItem.order_source}
                onChange={e => setNewItem(v => ({ ...v, order_source: e.target.value }))}
                onKeyDown={e => { if (e.key === 'Enter') handleAdd() }}
              />
              <label className="short-list-label">
                <input
                  type="checkbox"
                  checked={newItem.short_list}
                  onChange={e => setNewItem(v => ({ ...v, short_list: e.target.checked }))}
                />
                Quick Check
              </label>
              <button className="btn add-btn" onClick={handleAdd} disabled={saving || !newItem.name.trim()}>
                Add
              </button>
            </div>
          </section>
        </>
      )}
    </div>
  )
}
