"use client"

import { useEffect, useRef, useState } from "react"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { ChatMessage } from "@/components/chat-message"

type Msg = { role: "user" | "assistant"; content: string }

export default function ChatPage() {
  const [messages, setMessages] = useState<Msg[]>([
    { role: "assistant", content: "Hello! Ask me anything about your results or healthy habits." },
  ])
  const [input, setInput] = useState("")
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" })
  }, [messages])

  async function send() {
    if (!input.trim()) return
    const userMsg: Msg = { role: "user", content: input }
    setMessages((m) => [...m, userMsg])
    setInput("")
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [...messages, userMsg] }),
    })
    const data = await res.json()
    setMessages((m) => [...m, { role: "assistant", content: data.reply }])
  }

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-3xl px-4 py-8">
        <h1 className="text-3xl font-semibold mb-4">AI Assistant</h1>
        <div className="grid gap-4">
          <div ref={listRef} className="rounded-xl border border-border bg-card p-4 h-[60vh] overflow-y-auto">
            <div className="grid gap-3">
              {messages.map((m, i) => (
                <ChatMessage key={i} role={m.role} content={m.content} />
              ))}
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
              placeholder="Type your message..."
              aria-label="Type your message"
            />
            <button
              className="rounded-lg bg-primary px-5 py-3 text-primary-foreground font-semibold"
              onClick={send}
              aria-label="Send message"
            >
              Send
            </button>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  )
}
