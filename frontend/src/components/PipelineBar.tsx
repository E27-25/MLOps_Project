import styles from './PipelineBar.module.css'

export type StepStatus = 'idle' | 'active' | 'done'

export interface PipelineStep {
  id: string
  label: string
  detail: string
  time: string
  status: StepStatus
}

interface Props {
  steps: PipelineStep[]
  running: boolean
}

export default function PipelineBar({ steps, running }: Props) {
  return (
    <div className={`${styles.pipeline} ${running ? styles.running : ''}`}>
      {steps.map((step, i) => (
        <div key={step.id} className={styles.stepWrap}>
          <div className={`${styles.step} ${styles[step.status]}`}>
            <div className={styles.dot} />
            <div className={styles.info}>
              <span className={styles.name}>{step.label}</span>
              {step.detail && <span className={styles.detail}>{step.detail}</span>}
              {step.time && <span className={styles.time}>{step.time}</span>}
            </div>
          </div>
          {i < steps.length - 1 && <div className={styles.connector} />}
        </div>
      ))}
    </div>
  )
}
