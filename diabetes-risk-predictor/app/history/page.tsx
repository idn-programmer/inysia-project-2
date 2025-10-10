"use client"

import { useEffect, useState } from "react"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"

type HistoryItem = {
  date: string
  risk: number
  inputs?: any
}

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([])

  useEffect(() => {
    const data: HistoryItem[] = JSON.parse(localStorage.getItem("predHistory") || "[]")
    setItems(data)
  }, [])

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-5xl px-4 py-8">
        <h1 className="text-3xl font-semibold mb-6">Prediction History</h1>

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

        <div className="overflow-x-auto rounded-xl border border-border">
          <table className="min-w-full bg-card">
            <thead className="bg-muted">
              <tr>
                <th className="px-4 py-3 text-left">Date</th>
                <th className="px-4 py-3 text-left">Result (%)</th>
                <th className="px-4 py-3 text-left">Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.length === 0 && (
                <tr>
                  <td colSpan={3} className="px-4 py-4 text-muted-foreground">
                    No history yet.
                  </td>
                </tr>
              )}
              {items.map((it, idx) => (
                <tr key={idx} className="border-t border-border">
                  <td className="px-4 py-3">{new Date(it.date).toLocaleString()}</td>
                  <td className="px-4 py-3">{it.risk}%</td>
                  <td className="px-4 py-3">
                    <button
                      className="rounded-lg border border-border px-3 py-2 hover:bg-muted"
                      onClick={() => alert(JSON.stringify(it.inputs || {}, null, 2))}
                      aria-label="View details"
                    >
                      View Details
                    </button>
                  </td>
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
