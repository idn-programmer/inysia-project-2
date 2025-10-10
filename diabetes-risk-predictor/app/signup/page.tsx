"use client"

import type React from "react"

import { useRouter } from "next/navigation"
import { useState } from "react"
import Link from "next/link"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { InputField } from "@/components/input-field"

export default function SignupPage() {
  const router = useRouter()
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirm, setConfirm] = useState("")

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (password !== confirm) {
      alert("Passwords do not match")
      return
    }
    const res = await fetch("/api/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: email, email, password }),
    })
    if (res.ok) {
      localStorage.setItem("userName", name || "User")
      localStorage.setItem("userEmail", email)
      router.push("/dashboard")
    } else {
      const data = await res.json().catch(() => ({}))
      alert(data?.detail || "Signup failed")
    }
  }

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-lg px-4 py-10">
        <h1 className="text-3xl font-semibold mb-6">Sign Up</h1>
        <form onSubmit={onSubmit} className="grid gap-4">
          <InputField label="Name" name="name" value={name} onChange={setName} required />
          <InputField label="Email" name="email" type="email" value={email} onChange={setEmail} required />
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
            className="rounded-lg bg-primary px-6 py-4 text-primary-foreground font-semibold"
            aria-label="Create account"
          >
            Create Account
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
