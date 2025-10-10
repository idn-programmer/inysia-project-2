"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Home, Activity, History, MessageSquare, User } from "lucide-react"
import { cn } from "@/lib/utils"

const links = [
  { href: "/dashboard", label: "Predict", icon: Activity },
  { href: "/history", label: "History", icon: History },
  { href: "/chat", label: "AI Assistant", icon: MessageSquare },
  { href: "/profile", label: "Profile", icon: User },
]

export function Navbar() {
  const pathname = usePathname()
  return (
    <nav
      aria-label="Main navigation"
      className="sticky top-0 z-40 w-full border-b border-border bg-background/80 backdrop-blur"
    >
      <div className="mx-auto max-w-5xl px-4 py-3 flex items-center gap-4">
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 rounded-lg px-3 py-2 hover:bg-muted"
          aria-label="Go to dashboard"
        >
          <Home className="size-6 text-primary" aria-hidden="true" />
          <span className="font-semibold">Diabetes Risk Predictor</span>
        </Link>
        <div className="ml-auto flex items-center gap-1">
          {links.map((l) => {
            const Icon = l.icon
            const active = pathname === l.href
            return (
              <Link
                key={l.href}
                href={l.href}
                className={cn(
                  "inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm md:text-base",
                  active ? "bg-primary text-primary-foreground" : "hover:bg-muted",
                )}
                aria-current={active ? "page" : undefined}
                aria-label={l.label}
              >
                <Icon className="size-5" aria-hidden="true" />
                <span className="hidden sm:inline">{l.label}</span>
              </Link>
            )
          })}
        </div>
      </div>
    </nav>
  )
}
