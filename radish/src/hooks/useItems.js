import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'

export function useItems(mode) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function fetchItems() {
      let query = supabase
        .from('items')
        .select('*, locations(location_order)')

      if (mode === 'short') {
        query = query.eq('short_list', true)
      }

      const { data, error } = await query

      if (error) {
        setError(error.message)
      } else {
        data.sort((a, b) => {
          const orderA = a.locations?.location_order ?? 999
          const orderB = b.locations?.location_order ?? 999
          if (orderA !== orderB) return orderA - orderB
          return a.id - b.id
        })
        setItems(data)
      }
      setLoading(false)
    }

    fetchItems()
  }, [mode])

  return { items, loading, error }
}
