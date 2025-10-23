"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { useUser } from "@/lib/user-context"
import { useEffect, useState } from "react"
import { apiClient } from "@/lib/api"
import { PredictionOut } from "@/lib/types"

export default function DashboardPage() {
  const router = useRouter()
  const { user, isAuthenticated, isLoading: authLoading } = useUser()
  const [lastPrediction, setLastPrediction] = useState<PredictionOut | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    // Redirect to login if not authenticated (only after auth check is complete)
    if (!authLoading && !isAuthenticated) {
      router.push('/login')
      return
    }

    if (isAuthenticated) {
      const loadLatestPrediction = async () => {
        try {
          const data = await apiClient.getHistory(1) // Get only the latest prediction
          if (data.length > 0) {
            setLastPrediction(data[0])
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : "Failed to load prediction data")
        } finally {
          setIsLoading(false)
        }
      }

      loadLatestPrediction()
    }
  }, [isAuthenticated, authLoading, router])

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-5xl px-4 py-8">
        <h1 className="text-3xl font-semibold mb-2">Selamat datang, {user?.username || "Pengguna"}!</h1>
        <p className="text-muted-foreground mb-8">Pilih opsi untuk memulai.</p>

        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-100 text-red-700 border border-red-200">
            {error}
          </div>
        )}

        {authLoading ? (
          <div className="mb-8 rounded-xl border border-border p-5 bg-card">
            <div className="animate-pulse">
              <div className="h-4 bg-muted rounded w-32 mb-2"></div>
              <div className="h-8 bg-muted rounded w-16"></div>
            </div>
          </div>
        ) : !isAuthenticated ? (
          <div className="mb-8 rounded-xl border border-border p-5 bg-card">
            <p className="text-muted-foreground">Silakan masuk untuk melihat riwayat prediksi Anda.</p>
          </div>
        ) : isLoading ? (
          <div className="mb-8 rounded-xl border border-border p-5 bg-card">
            <div className="animate-pulse">
              <div className="h-4 bg-muted rounded w-32 mb-2"></div>
              <div className="h-8 bg-muted rounded w-16"></div>
            </div>
          </div>
        ) : lastPrediction ? (
          <div className="mb-8 rounded-xl border border-border p-5 bg-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Prediksi Terbaru</p>
                <p className="text-3xl font-bold mt-1">{lastPrediction.risk}%</p>
                <p className="text-sm text-muted-foreground mt-1">
                  {new Date(lastPrediction.created_at).toLocaleDateString()}
                </p>
              </div>
              <Link
                href="/history"
                className="text-sm text-primary hover:text-primary/80 underline"
              >
                Lihat Semua
              </Link>
            </div>
          </div>
        ) : (
          <div className="mb-8 rounded-xl border border-border p-5 bg-card">
            <p className="text-muted-foreground">Belum ada prediksi. Mulai dengan penilaian risiko pertama Anda!</p>
          </div>
        )}

        <div className="grid gap-4 sm:grid-cols-2">
          <Link
            href="/predict"
            className="rounded-xl bg-primary px-6 py-6 text-primary-foreground font-semibold text-center"
            aria-label="Go to Prediction"
          >
            Prediksi
          </Link>
          <Link
            href="/history"
            className="rounded-xl bg-accent px-6 py-6 text-accent-foreground font-semibold text-center"
            aria-label="Go to History"
          >
            Riwayat
          </Link>
          <Link
            href="/chat"
            className="rounded-xl border border-border px-6 py-6 font-semibold text-center hover:bg-muted"
            aria-label="Go to AI Assistant"
          >
            Asisten AI
          </Link>
          <Link
            href="/profile"
            className="rounded-xl border border-border px-6 py-6 font-semibold text-center hover:bg-muted"
            aria-label="Go to Profile"
          >
            Profil
          </Link>
        </div>
      </main>
      <Footer />
    </div>
  )
}
