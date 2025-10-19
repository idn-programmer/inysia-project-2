"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"
import { apiClient } from "@/lib/api"
import { PredictionOut } from "@/lib/types"
import { useUser } from "@/lib/user-context"

export default function HistoryPage() {
  const router = useRouter()
  const { user, isAuthenticated, isLoading: authLoading } = useUser()
  const [items, setItems] = useState<PredictionOut[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    // Redirect to login if not authenticated (only after auth check is complete)
    if (!authLoading && !isAuthenticated) {
      router.push('/login')
      return
    }

    if (isAuthenticated) {
      const loadHistory = async () => {
        try {
          const data = await apiClient.getHistory(50)
          setItems(data)
        } catch (err) {
          setError(err instanceof Error ? err.message : "Failed to load history")
        } finally {
          setIsLoading(false)
        }
      }

      loadHistory()
    }
  }, [isAuthenticated, authLoading, router])

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-5xl px-4 py-8">
        <h1 className="text-3xl font-semibold mb-6">Prediction History</h1>

        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-100 text-red-700 border border-red-200">
            {error}
          </div>
        )}

        {authLoading ? (
          <div className="text-center py-8">
            <p>Loading...</p>
          </div>
        ) : !isAuthenticated ? (
          <div className="text-center py-8">
            <p>Please log in to view your prediction history.</p>
          </div>
        ) : isLoading ? (
          <div className="text-center py-8">
            <p>Loading history...</p>
          </div>
        ) : (
          <>
            {items.length > 0 && (
              <div className="rounded-xl border border-border p-4 mb-6 bg-card">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={items.map((x, i) => ({ idx: items.length - i, risk: x.risk }))}>
                      <XAxis dataKey="idx" tickLine={false} axisLine={false} />
                      <YAxis domain={[0, 100]} tickLine={false} axisLine={false} />
                      <Tooltip />
                      <Line type="monotone" dataKey="risk" stroke="oklch(0.6 0.118 184.704)" strokeWidth={3} dot />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </>
        )}

        <div className="overflow-x-auto rounded-xl border border-border">
          <table className="min-w-full bg-card">
            <thead className="bg-muted">
              <tr>
                <th className="px-4 py-3 text-left">Date</th>
                <th className="px-4 py-3 text-left">Result (%)</th>
              </tr>
            </thead>
            <tbody>
              {items.length === 0 && (
                <tr>
                  <td colSpan={2} className="px-4 py-4 text-muted-foreground">
                    No history yet.
                  </td>
                </tr>
              )}
              {items.map((it, idx) => (
                <tr key={idx} className="border-t border-border">
                  <td className="px-4 py-3">{new Date(it.created_at).toLocaleString()}</td>
                  <td className="px-4 py-3">{it.risk}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
      <Footer />
    </div>
  )
}
