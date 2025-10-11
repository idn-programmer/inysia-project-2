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
  const { user } = useUser()
  const [messages, setMessages] = useState<Msg[]>([
    { role: "assistant", content: "Hello! Ask me anything about your results or healthy habits." },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [predictionContext, setPredictionContext] = useState<PredictionContext | null>(null)
  const [contextSent, setContextSent] = useState(false)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Check for prediction context in localStorage
    const storedContext = localStorage.getItem("predictionContext")
    if (storedContext) {
      try {
        const context = JSON.parse(storedContext)
        setPredictionContext(context)
        
        // Auto-send initial message about prediction results
        if (!contextSent) {
          setContextSent(true)
          const initialMessage = "I just received my diabetes risk assessment. Can you explain my results and provide personalized recommendations?"
          sendMessageWithContext(initialMessage, context)
        }
      } catch (e) {
        console.error("Failed to parse prediction context", e)
      }
    }
  }, [])

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
      
      const data = await apiClient.chat({
        messages: chatMessages,
        userId: user?.id,
        prediction_context: context || undefined
      })
      
      setMessages((m) => [...m, { role: "assistant", content: data.reply }])
    } catch (error) {
      setMessages((m) => [...m, { 
        role: "assistant", 
        content: "Sorry, I encountered an error. Please try again." 
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
          <h1 className="text-3xl font-semibold">AI Assistant</h1>
          <button
            onClick={() => router.push('/predict')}
            className="rounded-lg bg-secondary hover:bg-secondary/80 px-4 py-2 text-sm font-medium transition-colors"
          >
            📊 Get New Risk Assessment
          </button>
        </div>

        {predictionContext && (
          <div className="mb-4 p-4 rounded-lg border border-border bg-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Your Risk Score</p>
                <p className="text-2xl font-semibold">
                  <span className={riskColor}>{predictionContext.risk_score}%</span>
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-muted-foreground">Top Risk Factors</p>
                <p className="text-sm font-medium">
                  {Object.entries(predictionContext.shap_values)
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
              placeholder="Type your message..."
              aria-label="Type your message"
            />
            <button
              className="rounded-lg bg-primary px-5 py-3 text-primary-foreground font-semibold disabled:opacity-50"
              onClick={send}
              disabled={isLoading}
              aria-label="Send message"
            >
              {isLoading ? "Sending..." : "Send"}
            </button>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  )
}
