import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import { Suspense } from "react"
import { UserProvider } from "@/lib/user-context"
import { AccessibilityProvider } from "@/lib/accessibility-context"
import { AccessibilityWrapper } from "@/components/accessibility-wrapper"
import "./globals.css"

export const metadata: Metadata = {
  title: "v0 App",
  description: "Created with v0",
  generator: "v0.app",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable} antialiased`}>
        <AccessibilityProvider>
          <UserProvider>
            <Suspense fallback={<div>Loading...</div>}>
              <main role="main">{children}</main>
            </Suspense>
            <AccessibilityWrapper />
          </UserProvider>
        </AccessibilityProvider>
        <Analytics />
      </body>
    </html>
  )
}
