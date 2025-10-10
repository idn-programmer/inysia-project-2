import { cookies } from "next/headers"
import { NextResponse } from "next/server"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"

export async function POST(req: Request) {
  const body = await req.json()
  const res = await fetch(`${API_BASE}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  const data = await res.json()
  if (res.ok && data?.access_token) {
    const cookieStore = await cookies()
    cookieStore.set("token", data.access_token, { httpOnly: true, sameSite: "lax", path: "/" })
  }
  return NextResponse.json(data, { status: res.status })
}


