"use client"

import type React from "react"

import { useRouter } from "next/navigation"
import { useState } from "react"
import Link from "next/link"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { InputField } from "@/components/input-field"
import { useUser } from "@/lib/user-context"

export default function SignupPage() {
  const router = useRouter()
  const { signup } = useUser()
  const [username, setUsername] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirm, setConfirm] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (password !== confirm) {
      setError("Passwords do not match")
      return
    }
    
    setIsLoading(true)
    setError("")
    
    try {
      await signup(username, email, password)
      router.push("/dashboard")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Signup failed")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-lg px-4 py-10">
        <h1 className="text-3xl font-semibold mb-6">Sign Up</h1>
        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-100 text-red-700 border border-red-200">
            {error}
          </div>
        )}
        <form onSubmit={onSubmit} className="grid gap-4">
          <InputField 
            label="Username" 
            name="username" 
            value={username} 
            onChange={setUsername} 
            required 
          />
          <InputField 
            label="Email" 
            name="email" 
            type="email" 
            value={email} 
            onChange={setEmail} 
            required 
          />
          <InputField
            label="Password"
            name="password"
            type="password"
            value={password}
            onChange={setPassword}
            required
          />
          <InputField
            label="Confirm Password"
            name="confirm"
            type="password"
            value={confirm}
            onChange={setConfirm}
            required
          />
          <button
            type="submit"
            disabled={isLoading}
            className="rounded-lg bg-primary px-6 py-4 text-primary-foreground font-semibold disabled:opacity-50"
            aria-label="Create account"
          >
            {isLoading ? "Creating Account..." : "Create Account"}
          </button>
        </form>
        <p className="mt-4">
          Already have an account?{" "}
          <Link href="/login" className="text-primary underline">
            Log in
          </Link>
        </p>
      </main>
      <Footer />
    </div>
  )
}
