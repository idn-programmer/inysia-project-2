"use client"

import Link from "next/link"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { useEffect, useState } from "react"

export default function DashboardPage() {
  const [name, setName] = useState<string>("")
  const [lastRisk, setLastRisk] = useState<number | null>(null)

  useEffect(() => {
    setName(localStorage.getItem("userName") || "User")
    const lr = localStorage.getItem("lastRisk")
    setLastRisk(lr ? Number(lr) : null)
  }, [])

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-5xl px-4 py-8">
        <h1 className="text-3xl font-semibold mb-2">Welcome, {name ? name : "User"}!</h1>
        <p className="text-muted-foreground mb-8">Choose an option to get started.</p>

        {lastRisk !== null && (
          <div className="mb-8 rounded-xl border border-border p-5 bg-card">
            <p className="font-medium">Last Predicted Risk</p>
            <p className="text-3xl font-bold mt-1">{lastRisk}%</p>
          </div>
        )}

        <div className="grid gap-4 sm:grid-cols-2">
          <Link
            href="/predict"
            className="rounded-xl bg-primary px-6 py-6 text-primary-foreground font-semibold text-center"
            aria-label="Go to Prediction"
          >
            Predict
          </Link>
          <Link
            href="/history"
            className="rounded-xl bg-accent px-6 py-6 text-accent-foreground font-semibold text-center"
            aria-label="Go to History"
          >
            History
          </Link>
          <Link
            href="/chat"
            className="rounded-xl border border-border px-6 py-6 font-semibold text-center hover:bg-muted"
            aria-label="Go to AI Assistant"
          >
            AI Assistant
          </Link>
          <Link
            href="/profile"
            className="rounded-xl border border-border px-6 py-6 font-semibold text-center hover:bg-muted"
            aria-label="Go to Profile"
          >
            Profile
          </Link>
        </div>
      </main>
      <Footer />
    </div>
  )
}
