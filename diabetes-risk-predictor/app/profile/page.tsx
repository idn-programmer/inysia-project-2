"use client"

import { useEffect, useState } from "react"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { useRouter } from "next/navigation"

export default function ProfilePage() {
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const router = useRouter()

  useEffect(() => {
    setName(localStorage.getItem("userName") || "")
    setEmail(localStorage.getItem("userEmail") || "")
  }, [])

  function logout() {
    localStorage.removeItem("userName")
    localStorage.removeItem("userEmail")
    localStorage.removeItem("lastRisk")
    // keep history to demo continuity, comment next line to preserve history on logout
    // localStorage.removeItem("predHistory")
    router.push("/")
  }

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-3xl px-4 py-8">
        <h1 className="text-3xl font-semibold mb-6">Profile</h1>
        <div className="grid gap-4 rounded-xl border border-border bg-card p-6">
          <div className="grid sm:grid-cols-3">
            <p className="font-medium">Name</p>
            <p className="sm:col-span-2">{name || "—"}</p>
          </div>
          <div className="grid sm:grid-cols-3">
            <p className="font-medium">Email</p>
            <p className="sm:col-span-2">{email || "—"}</p>
          </div>
        </div>
        <button
          className="mt-6 rounded-lg bg-danger px-6 py-4 text-white font-semibold"
          onClick={logout}
          aria-label="Log out"
        >
          Log Out
        </button>
      </main>
      <Footer />
    </div>
  )
}
