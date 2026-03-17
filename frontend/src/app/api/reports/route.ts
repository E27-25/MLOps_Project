import { NextResponse } from 'next/server'
import { db } from '@/db'
import { reports } from '@/db/schema'
import { desc } from 'drizzle-orm'

export async function GET() {
  try {
    const rows = await db.select().from(reports).orderBy(desc(reports.created_at)).limit(100)
    return NextResponse.json(rows)
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
