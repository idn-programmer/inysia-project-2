"use client"

import type React from "react"

import { useRouter } from "next/navigation"
import { useState } from "react"
import Link from "next/link"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { InputField } from "@/components/input-field"
import { useUser } from "@/lib/user-context"

export default function LoginPage() {
  const router = useRouter()
  const { login } = useUser()
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setIsLoading(true)
    setError("")
    
    try {
      await login(username, password)
      router.push("/dashboard")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Gagal masuk")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-lg px-4 py-10">
        <h1 className="text-3xl font-semibold mb-6">Masuk</h1>
        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-100 text-red-700 border border-red-200">
            {error}
          </div>
        )}
        <form onSubmit={onSubmit} className="grid gap-4">
          <InputField 
            label="Nama Pengguna" 
            name="username" 
            type="text" 
            value={username} 
            onChange={setUsername} 
            required 
          />
          <InputField
            label="Kata Sandi"
            name="password"
            type="password"
            value={password}
            onChange={setPassword}
            required
          />
          <button
            type="submit"
            disabled={isLoading}
            className="rounded-lg bg-primary px-6 py-4 text-primary-foreground font-semibold disabled:opacity-50"
            aria-label="Log in"
          >
            {isLoading ? "Masuk..." : "Masuk"}
          </button>
        </form>
        <p className="mt-4">
          Belum memiliki akun?{" "}
          <Link href="/signup" className="text-primary underline">
            Daftar
          </Link>
        </p>
      </main>
      <Footer />
    </div>
  )
}
