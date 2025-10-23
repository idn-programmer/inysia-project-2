"use client"

import { useEffect, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { ChatMessage } from "@/components/chat-message"
import { apiClient } from "@/lib/api"
import { useUser } from "@/lib/user-context"
import { ChatMessageIn, PredictionContext } from "@/lib/types"

type Msg = { role: "user" | "assistant"; content: string }

export default function ChatPage() {
  const router = useRouter()
  const { user, isAuthenticated, isLoading: authLoading } = useUser()
  const [messages, setMessages] = useState<Msg[]>([
    { role: "assistant", content: "Halo! Tanyakan apa saja tentang hasil Anda atau kebiasaan sehat." },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [predictionContext, setPredictionContext] = useState<PredictionContext | null>(null)
  const [contextSent, setContextSent] = useState(false)
  const [isClient, setIsClient] = useState(false)
  const listRef = useRef<HTMLDivElement>(null)

  // Fix hydration by ensuring client-side only rendering for conditional content
  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!authLoading && !isAuthenticated) {
      router.push('/login')
      return
    }
  }, [isAuthenticated, authLoading, router])

  useEffect(() => {
    // Only run on client side to avoid hydration mismatch
    if (!isClient) return
    
    // Check for prediction context in localStorage
    const storedContext = localStorage.getItem("predictionContext")
    if (storedContext) {
      try {
        const context = JSON.parse(storedContext)
        setPredictionContext(context)
        
        // Auto-send initial message about prediction results
        if (!contextSent) {
          setContextSent(true)
          const initialMessage = "Saya baru saja menerima penilaian risiko diabetes. Bisakah Anda menjelaskan hasil saya dan memberikan rekomendasi yang dipersonalisasi?"
          sendMessageWithContext(initialMessage, context)
        }
      } catch (e) {
        console.error("Failed to parse prediction context", e)
      }
    }
  }, [isClient])

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" })
  }, [messages])

  async function sendMessageWithContext(message: string, context: PredictionContext | null) {
    const userMsg: Msg = { role: "user", content: message }
    setMessages((m) => [...m, userMsg])
    setIsLoading(true)
    
    try {
      const chatMessages: ChatMessageIn[] = [...messages, userMsg].map(msg => ({
        role: msg.role,
        content: msg.content
      }))
      
      console.log("🚀 Sending chat request:", {
        messages: chatMessages,
        userId: user?.id,
        prediction_context: context || undefined
      })
      
      const data = await apiClient.chat({
        messages: chatMessages,
        userId: user?.id,
        prediction_context: context || undefined
      })
      
      console.log("✅ Chat response received:", data)
      
      setMessages((m) => [...m, { role: "assistant", content: data.reply }])
    } catch (error) {
      console.error("❌ Chat request failed:", error)
      setMessages((m) => [...m, { 
        role: "assistant", 
        content: "Maaf, saya mengalami kesalahan. Silakan coba lagi." 
      }])
    } finally {
      setIsLoading(false)
    }
  }

  async function send() {
    if (!input.trim() || isLoading) return
    
    const message = input
    setInput("")
    
    // Send with context only if it exists and hasn't been sent yet
    await sendMessageWithContext(message, null)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const riskColor = predictionContext 
    ? predictionContext.risk_score < 33 
      ? "text-green-600" 
      : predictionContext.risk_score < 66 
        ? "text-yellow-600" 
        : "text-red-600"
    : ""

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-3xl px-4 py-8">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-3xl font-semibold">Asisten AI</h1>
          <button
            onClick={() => router.push('/predict')}
            className="rounded-lg bg-secondary hover:bg-secondary/80 px-4 py-2 text-sm font-medium transition-colors"
          >
            📊 Dapatkan Penilaian Risiko Baru
          </button>
        </div>

        {!isClient ? (
          <div className="text-center py-8">
            <p>Memuat...</p>
          </div>
        ) : authLoading ? (
          <div className="text-center py-8">
            <p>Memuat...</p>
          </div>
        ) : !isAuthenticated ? (
          <div className="text-center py-8">
            <p>Silakan masuk untuk menggunakan Asisten AI.</p>
          </div>
        ) : (
          <>
            {predictionContext && (
          <div className="mb-4 p-4 rounded-lg border border-border bg-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Skor Risiko Anda</p>
                <p className="text-2xl font-semibold">
                  <span className={riskColor}>{predictionContext.risk_score}%</span>
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-muted-foreground">Faktor Risiko Utama</p>
                <p className="text-sm font-medium">
                  {Object.entries(predictionContext.shap_values)
                    .filter(([key]) => key !== 'age' && key !== 'gender')
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .slice(0, 3)
                    .map(([key]) => key)
                    .join(", ")}
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="grid gap-4">
          <div ref={listRef} className="rounded-xl border border-border bg-card p-4 h-[60vh] overflow-y-auto">
            <div className="grid gap-3">
              {messages.map((m, i) => (
                <ChatMessage key={i} role={m.role} content={m.content} />
              ))}
              {isLoading && (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <div className="animate-pulse">●</div>
                  <div className="animate-pulse animation-delay-200">●</div>
                  <div className="animate-pulse animation-delay-400">●</div>
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="chat-input" className="sr-only">
              Message
            </label>
            <input
              id="chat-input"
              className="flex-1 rounded-lg border border-input bg-background px-4 py-3"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ketik pesan Anda..."
              aria-label="Type your message"
            />
            <button
              className="rounded-lg bg-primary px-5 py-3 text-primary-foreground font-semibold disabled:opacity-50"
              onClick={send}
              disabled={isLoading}
              aria-label="Send message"
            >
              {isLoading ? "Mengirim..." : "Kirim"}
            </button>
          </div>
        </div>
          </>
        )}
      </main>
      <Footer />
    </div>
  )
}
