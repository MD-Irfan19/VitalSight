
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://ynfbhzkfsnejklogheos.supabase.co'
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InluZmJoemtmc25lamtsb2doZW9zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA1MTExMTIsImV4cCI6MjA4NjA4NzExMn0.i8jpbjxDGZgyCRwFlUzENPWplgdxdKtWR-b-aZXkwes'

export const supabase = createClient(supabaseUrl, supabaseKey)
