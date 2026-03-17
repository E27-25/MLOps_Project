import styles from './EpiFields.module.css'

interface Props {
  fields: Record<string, unknown>
}

export default function EpiFields({ fields }: Props) {
  const rows = [
    ['Species', (fields.species as string[] || []).join(', ')],
    ['Symptoms', (fields.symptoms as string[] || []).slice(0, 4).join(', ')],
    ['Mortality', String(fields.mortality_count ?? '')],
    ['Affected Count', String(fields.affected_count ?? '')],
    ['Location', String(fields.location ?? '')],
    ['Timeframe', String(fields.timeframe ?? '')],
    ['Reporter Role', String(fields.reporter_role ?? '')],
    ['Summary', String(fields.raw_summary ?? '')],
  ]

  return (
    <div className={styles.grid}>
      {rows.map(([label, value]) => {
        const empty = !value || value === 'null' || value === 'undefined'
        return (
          <div key={label} className={styles.field}>
            <div className={styles.key}>{label}</div>
            <div className={`${styles.val} ${empty ? styles.empty : ''}`}>
              {empty ? 'not mentioned' : value}
            </div>
          </div>
        )
      })}
    </div>
  )
}
