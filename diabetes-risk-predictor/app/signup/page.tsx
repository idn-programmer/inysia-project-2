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
      setError("Kata sandi tidak cocok")
      return
    }
    
    setIsLoading(true)
    setError("")
    
    try {
      await signup(username, email, password)
      router.push("/dashboard")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Gagal mendaftar")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-lg px-4 py-10">
        <h1 className="text-3xl font-semibold mb-6">Daftar</h1>
        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-100 text-red-700 border border-red-200">
            {error}
          </div>
        )}
        <form onSubmit={onSubmit} className="grid gap-4">
          <InputField 
            label="Nama Pengguna" 
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
            label="Kata Sandi"
            name="password"
            type="password"
            value={password}
            onChange={setPassword}
            required
          />
          <InputField
            label="Konfirmasi Kata Sandi"
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
            {isLoading ? "Membuat Akun..." : "Buat Akun"}
          </button>
        </form>
        <p className="mt-4">
          Sudah memiliki akun?{" "}
          <Link href="/login" className="text-primary underline">
            Masuk
          </Link>
        </p>
      </main>
      <Footer />
    </div>
  )
}
