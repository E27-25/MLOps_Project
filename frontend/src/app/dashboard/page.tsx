'use client'
import { useEffect, useState } from 'react'
import type { Report } from '@/db/schema'

const RISK_COLOR: Record<string, string> = {
  CRITICAL: '#ef4444',
  HIGH:     '#f97316',
  MEDIUM:   '#eab308',
  LOW:      '#22c55e',
  NONE:     '#6b7280',
  UNKNOWN:  '#6b7280',
}

const DOMAIN_EMOJI: Record<string, string> = {
  avian_flu:     '🐦',
  rabies:        '🦇',
  fmd:           '🐄',
  nipah_hendra:  '🐎',
  leptospirosis: '🐀',
  general:       '🔬',
}

export default function DashboardPage() {
  const [reports, setReports] = useState<Report[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('/api/reports')
      .then(r => r.json())
      .then(data => { setReports(data); setLoading(false) })
      .catch(e => { setError(String(e)); setLoading(false) })
  }, [])

  // Stats
  const total = reports.length
  const byDomain = reports.reduce((acc, r) => {
    if (r.domain) acc[r.domain] = (acc[r.domain] ?? 0) + 1
    return acc
  }, {} as Record<string, number>)
  const byRisk = reports.reduce((acc, r) => {
    const k = r.risk_level ?? 'UNKNOWN'
    acc[k] = (acc[k] ?? 0) + 1
    return acc
  }, {} as Record<string, number>)
  const highRisk = (byRisk['HIGH'] ?? 0) + (byRisk['CRITICAL'] ?? 0)

  if (loading) return <div style={styles.center}>Loading…</div>
  if (error)   return <div style={styles.center}>Error: {error}</div>

  return (
    <div style={styles.page}>
      <h1 style={styles.title}>ZoonoticSense Dashboard</h1>
      <p style={styles.subtitle}>Disease Surveillance Reports</p>

      {/* Summary Cards */}
      <div style={styles.statsRow}>
        <StatCard label="Total Reports" value={total} color="#6366f1" />
        <StatCard label="High / Critical" value={highRisk} color="#ef4444" />
        <StatCard label="Domains Active" value={Object.keys(byDomain).length} color="#0ea5e9" />
      </div>

      {/* Domain breakdown */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>Reports by Domain</h2>
        <div style={styles.tagRow}>
          {Object.entries(byDomain).sort((a, b) => b[1] - a[1]).map(([domain, count]) => (
            <span key={domain} style={styles.tag}>
              {DOMAIN_EMOJI[domain] ?? '🔬'} {domain.replace(/_/g, ' ')} — {count}
            </span>
          ))}
        </div>
      </div>

      {/* Report Table */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>Recent Reports</h2>
        <div style={styles.tableWrap}>
          <table style={styles.table}>
            <thead>
              <tr>
                {['Time', 'Domain', 'Risk', 'Species', 'Location', 'Transcript'].map(h => (
                  <th key={h} style={styles.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {reports.map(r => (
                <tr key={r.id} style={styles.tr}>
                  <td style={styles.td}>
                    {r.created_at ? new Date(r.created_at).toLocaleString() : '—'}
                  </td>
                  <td style={styles.td}>
                    {DOMAIN_EMOJI[r.domain ?? ''] ?? '🔬'} {r.domain?.replace(/_/g, ' ') ?? '—'}
                  </td>
                  <td style={styles.td}>
                    <span style={{ ...styles.badge, background: RISK_COLOR[r.risk_level ?? 'UNKNOWN'] }}>
                      {r.risk_level ?? '—'}
                    </span>
                  </td>
                  <td style={styles.td}>{(r.species as string[] | null)?.join(', ') || '—'}</td>
                  <td style={styles.td}>{r.location ?? '—'}</td>
                  <td style={{ ...styles.td, maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {r.transcript ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ ...styles.statCard, borderTop: `4px solid ${color}` }}>
      <div style={{ ...styles.statValue, color }}>{value}</div>
      <div style={styles.statLabel}>{label}</div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  page:         { maxWidth: 1100, margin: '0 auto', padding: '2rem 1rem', fontFamily: 'sans-serif' },
  center:       { display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh', fontSize: 18 },
  title:        { fontSize: 28, fontWeight: 700, margin: 0 },
  subtitle:     { color: '#6b7280', marginBottom: '1.5rem' },
  statsRow:     { display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' },
  statCard:     { flex: 1, minWidth: 140, background: '#f9fafb', borderRadius: 8, padding: '1rem', boxShadow: '0 1px 3px rgba(0,0,0,.1)' },
  statValue:    { fontSize: 36, fontWeight: 700 },
  statLabel:    { fontSize: 13, color: '#6b7280', marginTop: 4 },
  section:      { marginBottom: '2rem' },
  sectionTitle: { fontSize: 18, fontWeight: 600, marginBottom: '0.75rem' },
  tagRow:       { display: 'flex', flexWrap: 'wrap', gap: '0.5rem' },
  tag:          { background: '#e0f2fe', color: '#0369a1', padding: '4px 12px', borderRadius: 99, fontSize: 13 },
  tableWrap:    { overflowX: 'auto' },
  table:        { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th:           { textAlign: 'left', padding: '8px 12px', borderBottom: '2px solid #e5e7eb', color: '#374151', fontWeight: 600 },
  tr:           { borderBottom: '1px solid #f3f4f6' },
  td:           { padding: '8px 12px', color: '#111827', verticalAlign: 'top' },
  badge:        { display: 'inline-block', padding: '2px 8px', borderRadius: 99, color: '#fff', fontSize: 11, fontWeight: 600 },
}
