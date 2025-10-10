type ChatMessageProps = {
  role: "user" | "assistant"
  content: string
}

export function ChatMessage({ role, content }: ChatMessageProps) {
  const isUser = role === "user"
  return (
    <div
      className={
        isUser
          ? "ml-auto max-w-[85%] rounded-2xl bg-primary text-primary-foreground px-4 py-3"
          : "mr-auto max-w-[85%] rounded-2xl bg-muted text-foreground px-4 py-3"
      }
      aria-label={isUser ? "Your message" : "Assistant message"}
    >
      {content}
    </div>
  )
}
