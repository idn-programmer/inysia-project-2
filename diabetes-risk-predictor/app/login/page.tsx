"use client"

import type React from "react"

import { useRouter } from "next/navigation"
import { useState } from "react"
import Link from "next/link"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { InputField } from "@/components/input-field"

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: email, password }),
    })
    if (res.ok) {
      const name = (email.split("@")[0] || "User").replace(/\W+/g, " ")
      localStorage.setItem("userName", name)
      localStorage.setItem("userEmail", email)
      router.push("/dashboard")
    } else {
      const data = await res.json().catch(() => ({}))
      alert(data?.detail || "Login failed")
    }
  }

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-lg px-4 py-10">
        <h1 className="text-3xl font-semibold mb-6">Log In</h1>
        <form onSubmit={onSubmit} className="grid gap-4">
          <InputField label="Email" name="email" type="email" value={email} onChange={setEmail} required />
          <InputField
            label="Password"
            name="password"
            type="password"
            value={password}
            onChange={setPassword}
            required
          />
          <button
            type="submit"
            className="rounded-lg bg-primary px-6 py-4 text-primary-foreground font-semibold"
            aria-label="Log in"
          >
            Log In
          </button>
        </form>
        <p className="mt-4">
          Don&apos;t have an account?{" "}
          <Link href="/signup" className="text-primary underline">
            Sign up
          </Link>
        </p>
      </main>
      <Footer />
    </div>
  )
}
