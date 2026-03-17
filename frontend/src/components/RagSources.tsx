import styles from './RagSources.module.css'

interface Props {
  chunks: string[]
}

export default function RagSources({ chunks }: Props) {
  if (!chunks.length) {
    return <p className={styles.empty}>No chunks retrieved.</p>
  }
  return (
    <div className={styles.list}>
      {chunks.map((chunk, i) => (
        <div key={i} className={styles.chunk}>
          <div className={styles.num}>SOURCE {i + 1}</div>
          <div className={styles.text}>
            {chunk.slice(0, 280)}{chunk.length > 280 ? '…' : ''}
          </div>
        </div>
      ))}
    </div>
  )
}
