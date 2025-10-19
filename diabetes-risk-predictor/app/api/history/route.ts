import { cookies } from "next/headers"
import { NextResponse } from "next/server"

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const limit = searchParams.get("limit") || "50"
    
    // Get token from cookies or Authorization header
    const cookieToken = (await cookies()).get("token")?.value
    const authHeader = req.headers.get("authorization")
    const token = cookieToken || authHeader?.replace("Bearer ", "")
    
    if (!token) {
      return NextResponse.json(
        { detail: "Authentication required" },
        { status: 401 }
      )
    }
    
    const res = await fetch(`${API_BASE}/history?limit=${limit}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
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

