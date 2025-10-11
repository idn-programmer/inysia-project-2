import { cookies } from "next/headers"
import { NextResponse } from "next/server"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const limit = searchParams.get("limit") || "50"
    
    const token = (await cookies()).get("token")?.value
    const res = await fetch(`${API_BASE}/history?limit=${limit}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    })
    
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (error) {
    return NextResponse.json(
      { detail: "Failed to fetch history" },
      { status: 500 }
    )
  }
}

