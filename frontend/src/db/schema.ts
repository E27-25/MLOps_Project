import { pgTable, text, integer, real, timestamp, json } from 'drizzle-orm/pg-core'

export const reports = pgTable('reports', {
  id:             text('id').primaryKey(),
  created_at:     timestamp('created_at', { withTimezone: true }).defaultNow(),
  session_id:     text('session_id'),
  transcript:     text('transcript'),
  species:        json('species').$type<string[]>(),
  symptoms:       json('symptoms').$type<string[]>(),
  mortality_count: integer('mortality_count'),
  affected_count:  integer('affected_count'),
  location:       text('location'),
  reporter_role:  text('reporter_role'),
  domain:         text('domain'),
  confidence:     real('confidence'),
  risk_level:     text('risk_level'),
  report_flag:    text('report_flag'),
  llm_response:   text('llm_response'),
  asr_time_s:     real('asr_time_s'),
  ner_time_s:     real('ner_time_s'),
})

export type Report = typeof reports.$inferSelect
