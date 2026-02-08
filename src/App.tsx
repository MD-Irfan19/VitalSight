
import { useEffect, useState } from 'react'
import { supabase } from './lib/supabaseClient'
import AuthPage from './auth/AuthPage'
import type { Session } from '@supabase/supabase-js'

function App() {
  const [session, setSession] = useState<Session | null>(null)

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session)
    })

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session)
      if (session) console.log("Supabase Session Verified:", session)
    })

    return () => subscription.unsubscribe()
  }, [])

  if (!session) {
    return <AuthPage />
  }

  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold">Welcome to VitalSight</h1>
        <p className="mt-4 text-lg">Logged in as {session.user.email}</p>
        <button
          className="mt-4 rounded bg-red-500 px-4 py-2 text-white hover:bg-red-600"
          onClick={() => supabase.auth.signOut()}
        >
          Sign Out
        </button>
      </div>
    </div>
  )
}

export default App
