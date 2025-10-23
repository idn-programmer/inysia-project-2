"use client"

import { useState } from "react"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { useUser } from "@/lib/user-context"
import { useRouter } from "next/navigation"

export default function ProfilePage() {
  const { user, logout } = useUser()
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)

  const handleLogout = async () => {
    setIsLoading(true)
    logout()
    router.push("/")
  }

  if (!user) {
    return (
      <div className="min-h-dvh bg-background text-foreground">
        <Navbar />
        <main className="mx-auto max-w-lg px-4 py-10">
          <div className="text-center">
            <h1 className="text-2xl font-semibold mb-4">Silakan masuk untuk melihat profil Anda</h1>
            <button
              onClick={() => router.push("/login")}
              className="rounded-lg bg-primary px-6 py-3 text-primary-foreground font-semibold"
            >
              Ke Halaman Masuk
            </button>
          </div>
        </main>
        <Footer />
      </div>
    )
  }

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-lg px-4 py-10">
        <h1 className="text-3xl font-semibold mb-6">Profil</h1>
        
        <div className="rounded-xl border border-border p-6 bg-card">
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-muted-foreground">Nama Pengguna</label>
              <p className="text-lg">{user.username}</p>
            </div>
            
            {user.email && (
              <div>
                <label className="text-sm font-medium text-muted-foreground">Email</label>
                <p className="text-lg">{user.email}</p>
              </div>
            )}
            
            <div>
              <label className="text-sm font-medium text-muted-foreground">ID Pengguna</label>
              <p className="text-lg font-mono text-sm">{user.id}</p>
            </div>
          </div>
          
          <div className="mt-8 pt-6 border-t border-border">
            <button
              onClick={handleLogout}
              disabled={isLoading}
              className="w-full rounded-lg bg-red-600 px-6 py-3 text-white font-semibold hover:bg-red-700 disabled:opacity-50"
            >
              {isLoading ? "Keluar..." : "Keluar"}
            </button>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  )
}