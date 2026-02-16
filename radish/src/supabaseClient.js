import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://wujpugalfrlyjcbzhcxa.supabase.co'
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind1anB1Z2FsZnJseWpjYnpoY3hhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIzNzk5MjAsImV4cCI6MjA2Nzk1NTkyMH0.wr2YiMhCPr3q-TPA7HCOgq8cyreaup0FmpGf1dWyTNE'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
