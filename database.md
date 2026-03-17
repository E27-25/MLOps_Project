# Database Integration — ZoonoticSense

## Overview

ระบบใช้ **Neon (serverless PostgreSQL)** เป็น database กลาง โดยมี 2 ฝั่งที่เชื่อมต่อ:

- **Backend (FastAPI + SQLAlchemy)** — เขียนข้อมูลลง DB หลังจากประมวลผลเสร็จ
- **Frontend (Next.js + Drizzle)** — อ่านข้อมูลจาก DB เพื่อแสดงผล dashboard

---

## Architecture

```
User Voice
    │
    ▼
[FastAPI Backend]
    ├── ASR (Triton Whisper)
    ├── NER (vLLM Qwen3)
    ├── Router (MLP)
    ├── RAG (FAISS)
    ├── LLM Stream (vLLM)
    └── TTS (Triton Kokoro)
         │
         ▼ (stream เสร็จ)
    [SQLAlchemy] ──────────► [Neon PostgreSQL]
                                     │
                             [Drizzle ORM]
                                     │
                                     ▼
                           [Next.js Dashboard]
                           http://localhost:3000/dashboard
```

---

## ไฟล์ที่เพิ่ม/แก้ไข

### Backend

#### `backend/db.py` — Database Engine

```python
engine = create_engine(_url, pool_pre_ping=True)
SessionLocal = sessionmaker(...)
Base = declarative_base()
```

**หน้าที่:**
- รับ `DATABASE_URL` จาก environment variable
- แปลง URL scheme จาก `postgresql://` → `postgresql+psycopg2://` (SQLAlchemy ต้องการ)
- ลบ parameter `channel_binding` ที่ psycopg2 ไม่รองรับ
- สร้าง `engine` (connection pool) และ `SessionLocal` (factory สำหรับสร้าง session)
- `pool_pre_ping=True` = ทดสอบ connection ก่อนใช้ ป้องกัน stale connection

---

#### `backend/models.py` — Table Definition

```python
class Report(Base):
    __tablename__ = "reports"
    id              = Column(String, primary_key=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    transcript      = Column(String)
    domain          = Column(String)
    risk_level      = Column(String)
    ...
```

**หน้าที่:**
- กำหนด schema ของตาราง `reports` ใน PostgreSQL
- SQLAlchemy จะสร้างตารางนี้อัตโนมัติตอน startup (`create_tables()`)
- ใช้ `server_default=func.now()` = ให้ database กำหนดเวลาเอง (ไม่ใช้ Python time)

**ข้อมูลที่เก็บ:**

| Column | ประเภท | มาจาก |
|--------|--------|-------|
| `id` | UUID string | สร้างใหม่ทุก report |
| `created_at` | timestamp | PostgreSQL auto |
| `transcript` | text | ASR output |
| `domain` | text | Router (avian_flu, rabies, ...) |
| `confidence` | float | Router score |
| `risk_level` | text | LLM parse (LOW/MEDIUM/HIGH/CRITICAL) |
| `report_flag` | text | ควรแจ้งเจ้าหน้าที่ไหม |
| `llm_response` | text | คำตอบเต็มของ expert |
| `species` | JSON array | NER (["chicken", "duck"]) |
| `symptoms` | JSON array | NER (["lethargy", "death"]) |
| `mortality_count` | int | NER |
| `affected_count` | int | NER |
| `location` | text | NER |
| `reporter_role` | text | NER (farmer/vet/ranger) |
| `asr_time_s` | float | เวลา ASR (วินาที) |
| `ner_time_s` | float | เวลา NER (วินาที) |

---

#### `backend/app.py` — การบันทึกข้อมูล

**Startup — สร้างตาราง:**
```python
from db import create_tables, get_session
from models import Report as ReportModel
create_tables()  # CREATE TABLE IF NOT EXISTS
```

**StreamRequest — เพิ่ม fields:**
```python
class StreamRequest(BaseModel):
    domain: str = "general"
    epi_fields: dict = {}
    rag_chunks: list = []
    transcript: str = ""   # เพิ่ม
    timing: dict = {}      # เพิ่ม
```

**generate() — บันทึกหลัง stream เสร็จ:**
```python
# ท้ายสุดของ generate() ก่อน yield done
db = get_session()
row = ReportModel(
    id=str(uuid.uuid4()),
    transcript=transcript,
    domain=domain,
    risk_level=risk_level,
    llm_response=full_text.strip(),
    species=epi_fields.get("species", []),
    ...
)
db.add(row)
db.commit()
db.close()
```

**ทำไมบันทึกใน `/stream` ไม่ใช่ `/analyze`:**
- `/analyze` มีแค่ NER + Router + RAG — ยังไม่รู้ `risk_level` และ `llm_response`
- `/stream` รู้ข้อมูลครบเมื่อ LLM generate เสร็จ
- บันทึกตอน `generate()` ซึ่งรันใน threadpool thread → ใช้ sync SQLAlchemy ได้โดยตรง

---

### Frontend

#### `frontend/src/db/schema.ts` — Drizzle Schema

```typescript
export const reports = pgTable('reports', {
  id:         text('id').primaryKey(),
  created_at: timestamp('created_at', { withTimezone: true }).defaultNow(),
  transcript: text('transcript'),
  domain:     text('domain'),
  risk_level: text('risk_level'),
  species:    json('species').$type<string[]>(),
  ...
})
```

**หน้าที่:**
- กำหนด schema เดียวกับ backend แต่เป็น TypeScript
- Drizzle ใช้ schema นี้สร้าง type-safe queries
- ต้องให้ตรงกับ `backend/models.py` ทุก column

---

#### `frontend/src/db/index.ts` — Neon Connection

```typescript
import { neon } from '@neondatabase/serverless'
import { drizzle } from 'drizzle-orm/neon-http'

const sql = neon(process.env.DATABASE_URL!)
export const db = drizzle(sql, { schema })
```

**หน้าที่:**
- `neon()` สร้าง HTTP connection ไป Neon (ใช้ fetch แทน TCP socket)
- เหมาะกับ serverless/edge environment เช่น Next.js API routes
- `drizzle()` wrap connection ด้วย ORM layer

---

#### `frontend/src/app/api/reports/route.ts` — API Route

```typescript
export async function GET() {
  const rows = await db.select().from(reports)
                       .orderBy(desc(reports.created_at))
                       .limit(100)
  return NextResponse.json(rows)
}
```

**หน้าที่:**
- Next.js API route ที่ frontend เรียกเพื่อดึงข้อมูล
- รัน server-side ดังนั้น `DATABASE_URL` ไม่ถูก expose ให้ browser
- เรียกผ่าน `/api/reports` จาก dashboard page

---

#### `frontend/src/app/dashboard/page.tsx` — Dashboard UI

```
http://localhost:3000/dashboard
```

**แสดง:**
- **Summary cards** — จำนวน report ทั้งหมด, high/critical risk, domain ที่ active
- **Domain breakdown** — tag แสดงจำนวน report ต่อ domain
- **Report table** — รายการ 100 report ล่าสุด พร้อม time, domain, risk, species, location, transcript

---

#### `frontend/src/components/AssessmentCard.tsx` — ส่ง transcript + timing

เพิ่ม props:
```typescript
interface Props {
  transcript?: string
  timing?: Record<string, number>
  ...
}
```

ส่งไปใน stream request:
```typescript
body: JSON.stringify({
  domain,
  epi_fields: epiFields,
  rag_chunks: ragChunks,
  transcript: transcript ?? '',   // เพิ่ม
  timing: timing ?? {},           // เพิ่ม
})
```

---

### Config Files

#### `.env` (root)
```
DATABASE_URL=postgresql://...
```
ใช้โดย `docker-compose.yml` ผ่าน `${DATABASE_URL:-}`

#### `frontend/.env.local`
```
DATABASE_URL=postgresql://...
```
ใช้ตอน `npm run dev` (local development ไม่ใช้ Docker)

#### `frontend/drizzle.config.ts`
```typescript
export default defineConfig({
  schema: './src/db/schema.ts',
  dialect: 'postgresql',
  dbCredentials: { url: process.env.DATABASE_URL! },
})
```
ใช้ตอนรัน `npx drizzle-kit push` เพื่อ sync schema กับ DB

---

## Data Flow

```
1. User พูด → MicButton
2. POST /upload → ASR → transcript (+ asr_time)
3. POST /analyze → NER + Router + RAG → epi_fields, domain, timing
4. AssessmentCard เรียก POST /stream พร้อมส่ง transcript + timing
5. Backend stream LLM response ทีละ sentence
6. เมื่อ stream เสร็จ → parse risk_level → บันทึกลง Neon
7. Dashboard GET /api/reports → Drizzle query → แสดงผล
```

---

## Environment Variables

| Variable | ใช้ใน | ค่าตัวอย่าง |
|----------|-------|------------|
| `DATABASE_URL` | backend + frontend | `postgresql://user:pass@host/db?sslmode=require` |

---

## การ Deploy บน Google Cloud

1. สร้าง VM instance (GPU) หรือใช้ Cloud Run สำหรับ frontend
2. ใส่ `DATABASE_URL` ใน environment ของแต่ละ service
3. Neon ทำงานได้ทุก cloud — ไม่ต้อง setup DB บน GCP
4. Frontend สามารถ deploy บน Cloud Run (serverless) แยกจาก backend GPU VM ได้
