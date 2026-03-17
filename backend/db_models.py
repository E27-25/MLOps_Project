import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, func
from db import Base


class Report(Base):
    __tablename__ = "reports"

    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    session_id    = Column(String, nullable=True)

    # ASR
    transcript    = Column(String)

    # NER
    species       = Column(JSON)
    symptoms      = Column(JSON)
    mortality_count = Column(Integer, nullable=True)
    affected_count  = Column(Integer, nullable=True)
    location      = Column(String, nullable=True)
    reporter_role = Column(String, nullable=True)

    # Router
    domain        = Column(String)
    confidence    = Column(Float, nullable=True)

    # LLM output
    risk_level    = Column(String, nullable=True)
    report_flag   = Column(String, nullable=True)   # 'true' / 'false'
    llm_response  = Column(String, nullable=True)

    # Timing (ms)
    asr_time_s    = Column(Float, nullable=True)
    ner_time_s    = Column(Float, nullable=True)
