import styles from './RouterScores.module.css'

interface Props {
  domain: string
  scores: Record<string, number>
}

export default function RouterScores({ domain, scores }: Props) {
  const entries = Object.entries(scores)
  const max = Math.max(...entries.map(([, v]) => v))
  const selected = entries.filter(([d]) => d === domain)
  const rest = entries.filter(([d]) => d !== domain).sort((a, b) => b[1] - a[1])
  const sorted = [...selected, ...rest]

  return (
    <div className={styles.bars}>
      {sorted.map(([d, score]) => {
        const isSelected = d === domain
        const pct = (score * 100).toFixed(1)
        const barW = (score / max * 100).toFixed(1)
        return (
          <div key={d} className={styles.row}>
            <div className={`${styles.label} ${isSelected ? styles.selected : ''}`}>
              {d.replace(/_/g, ' ')}
            </div>
            <div className={styles.track}>
              <div
                className={`${styles.fill} ${isSelected ? styles.selectedFill : ''}`}
                style={{ width: `${barW}%` }}
              />
            </div>
            <div className={`${styles.score} ${isSelected ? styles.selected : ''}`}>{pct}%</div>
          </div>
        )
      })}
    </div>
  )
}
