
import { useState } from 'react'
import { supabase } from '../lib/supabaseClient'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../components/ui/card'
import { Activity, User, Stethoscope } from 'lucide-react'

export default function AuthPage() {
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [fullName, setFullName] = useState('')
    const [isSignUp, setIsSignUp] = useState(false)
    const [role, setRole] = useState<'patient' | 'doctor'>('patient')
    const [loading, setLoading] = useState(false)
    const [message, setMessage] = useState('')

    const handleAuth = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setMessage('')

        try {
            if (isSignUp) {
                const { error } = await supabase.auth.signUp({
                    email,
                    password,
                    options: {
                        data: {
                            full_name: fullName,
                            role: role,
                        },
                    },
                })
                if (error) throw error
                setMessage('Check your email for the confirmation link!')
            } else {
                const { error } = await supabase.auth.signInWithPassword({
                    email,
                    password,
                })
                if (error) throw error
            }
        } catch (error: any) {
            setMessage(error.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="flex min-h-screen items-center justify-center bg-gray-50 p-4">
            <Card className="w-full max-w-md shadow-lg">
                <CardHeader className="space-y-1">
                    <div className="flex justify-center mb-4">
                        <div className="rounded-full bg-primary/10 p-3">
                            <Activity className="h-8 w-8 text-primary" />
                        </div>
                    </div>
                    <CardTitle className="text-2xl font-bold text-center">VitalSight</CardTitle>
                    <CardDescription className="text-center">
                        {isSignUp ? 'Create an account to get started' : 'Sign in to your account'}
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleAuth} className="space-y-4">
                        {isSignUp && (
                            <div className="space-y-2">
                                <Label htmlFor="fullName">Full Name</Label>
                                <Input
                                    id="fullName"
                                    placeholder="John Doe"
                                    value={fullName}
                                    onChange={(e) => setFullName(e.target.value)}
                                    required
                                />
                            </div>
                        )}
                        <div className="space-y-2">
                            <Label htmlFor="email">Email</Label>
                            <Input
                                id="email"
                                type="email"
                                placeholder="m@example.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="password">Password</Label>
                            <Input
                                id="password"
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                        </div>

                        {isSignUp && (
                            <div className="space-y-2">
                                <Label>I am a...</Label>
                                <div className="grid grid-cols-2 gap-4">
                                    <div
                                        className={`cursor-pointer rounded-lg border p-4 hover:bg-gray-50 ${role === 'patient' ? 'border-primary bg-primary/5 ring-1 ring-primary' : ''
                                            }`}
                                        onClick={() => setRole('patient')}
                                    >
                                        <div className="flex flex-col items-center gap-2">
                                            <User className="h-6 w-6" />
                                            <span className="text-sm font-medium">Patient</span>
                                        </div>
                                    </div>
                                    <div
                                        className={`cursor-pointer rounded-lg border p-4 hover:bg-gray-50 ${role === 'doctor' ? 'border-primary bg-primary/5 ring-1 ring-primary' : ''
                                            }`}
                                        onClick={() => setRole('doctor')}
                                    >
                                        <div className="flex flex-col items-center gap-2">
                                            <Stethoscope className="h-6 w-6" />
                                            <span className="text-sm font-medium">Doctor</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {message && (
                            <p className="text-sm text-center text-red-500 bg-red-50 p-2 rounded">{message}</p>
                        )}

                        <Button className="w-full" type="submit" disabled={loading}>
                            {loading ? 'Processing...' : isSignUp ? 'Create Account' : 'Sign In'}
                        </Button>
                    </form>
                </CardContent>
                <CardFooter>
                    <Button
                        variant="link"
                        className="w-full"
                        onClick={() => setIsSignUp(!isSignUp)}
                    >
                        {isSignUp ? 'Already have an account? Sign in' : "Don't have an account? Sign up"}
                    </Button>
                </CardFooter>
            </Card>
        </div>
    )
}
