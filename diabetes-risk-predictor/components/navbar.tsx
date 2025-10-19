"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { Home, Activity, History, MessageSquare, User, LogOut } from "lucide-react"
import { cn } from "@/lib/utils"
import { useUser } from "@/lib/user-context"
import { useEffect, useState } from "react"

const links = [
  { href: "/predict", label: "Predict", icon: Activity },
  { href: "/history", label: "History", icon: History },
  { href: "/chat", label: "AI Assistant", icon: MessageSquare },
  { href: "/profile", label: "Profile", icon: User },
]

export function Navbar() {
  const pathname = usePathname()
  const router = useRouter()
  const { user, logout, isAuthenticated, isLoading } = useUser()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const handleLogout = () => {
    logout()
    router.push("/")
  }

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
          {mounted && isAuthenticated && (
            <>
              <Link
                href="/dashboard"
                className="text-sm text-muted-foreground hover:text-foreground hidden sm:inline px-2 py-1 rounded hover:bg-muted transition-colors"
              >
                Dashboard
              </Link>
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
              <button
                onClick={handleLogout}
                className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm md:text-base hover:bg-muted"
                aria-label="Logout"
              >
                <LogOut className="size-5" aria-hidden="true" />
                <span className="hidden sm:inline">Logout</span>
              </button>
            </>
          )}
          {mounted && !isAuthenticated && (
            <div className="flex items-center gap-2">
              <Link
                href="/login"
                className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-muted"
              >
                Login
              </Link>
              <Link
                href="/signup"
                className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm bg-primary text-primary-foreground hover:bg-primary/90"
              >
                Sign Up
              </Link>
            </div>
          )}
          {!mounted && (
            <div className="flex items-center gap-2">
              <div className="h-8 w-16 bg-muted animate-pulse rounded"></div>
              <div className="h-8 w-20 bg-muted animate-pulse rounded"></div>
            </div>
          )}
        </div>
      </div>
    </nav>
  )
}
