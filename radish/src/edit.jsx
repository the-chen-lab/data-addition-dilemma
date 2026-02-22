import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import EditApp from './EditApp'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <EditApp />
  </StrictMode>
)
