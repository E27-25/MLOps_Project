"""
utils/rag.py
============
Domain-partitioned RAG system.

Each disease domain has its own FAISS index + document store.
Router selects domain → retrieve from that domain's index only.

Directory layout:
  knowledge_base/
    avian_flu/       ← FAISS index + docs.jsonl
    rabies/
    fmd/
    nipah_hendra/
    leptospirosis/
    general/

Building indexes:
  python utils/rag.py --build          # builds all from knowledge_base/*/raw/*.txt|pdf
  python utils/rag.py --query "text"   # test retrieval
"""

import os, json, logging, pickle
import numpy as np
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ─── In-memory chunk store (no FAISS dependency required for small corpora) ──

class SimpleVectorStore:
    """Lightweight cosine vector store — no external vector DB needed."""

    def __init__(self):
        self.embeddings = []   # List[np.ndarray]
        self.documents  = []   # List[str]
        self.metadata   = []   # List[dict]

    def add(self, texts: list[str], embeddings: np.ndarray, meta: list[dict] = None):
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            self.embeddings.append(emb / (np.linalg.norm(emb) + 1e-9))
            self.documents.append(text)
            self.metadata.append(meta[i] if meta else {})

    def search(self, query_emb: np.ndarray, top_k: int = 3) -> list[tuple[float, str, dict]]:
        if not self.embeddings:
            return []
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        matrix = np.stack(self.embeddings)
        scores = matrix @ q
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.documents[i], self.metadata[i]) for i in top_idx]

    def __len__(self):
        return len(self.documents)

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> 'SimpleVectorStore':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ─── Domain RAG manager ────────────────────────────────────────────────────

class DomainRAG:
    """
    Manages one vector store per disease domain.
    If no index found, falls back to built-in seed knowledge.
    """

    DOMAINS = ["avian_flu", "rabies", "fmd", "nipah_hendra", "leptospirosis", "general"]

    def __init__(self, kb_dir: Path, embedder):
        self.kb_dir   = Path(kb_dir)
        self.embedder = embedder
        self.stores   = {}

        for domain in self.DOMAINS:
            store = self._load_or_seed(domain)
            self.stores[domain] = store
            log.info(f"RAG [{domain}]: {len(store)} chunks")

    @property
    def n_indexes(self):
        return sum(1 for s in self.stores.values() if len(s) > 0)

    def retrieve(self, query: str, domain: str, top_k: int = 3) -> list[str]:
        """Retrieve top_k chunks from domain's index."""
        store = self.stores.get(domain) or self.stores["general"]
        if not store:
            return []
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        results = store.search(q_emb, top_k=top_k)
        return [text for _, text, _ in results]

    def _load_or_seed(self, domain: str) -> SimpleVectorStore:
        """Load persisted index or fall back to seed knowledge."""
        idx_path = self.kb_dir / domain / "index.pkl"
        if idx_path.exists():
            try:
                store = SimpleVectorStore.load(idx_path)
                log.info(f"RAG [{domain}]: loaded index with {len(store)} chunks")
                return store
            except Exception as e:
                log.warning(f"RAG [{domain}]: failed to load index ({e}), using seed")

        # Build from seed knowledge
        seed_chunks = SEED_KNOWLEDGE.get(domain, [])
        if seed_chunks:
            store = SimpleVectorStore()
            embs  = self.embedder.encode(seed_chunks, show_progress_bar=False,
                                          normalize_embeddings=True)
            store.add(seed_chunks, embs, [{"source": "seed", "domain": domain}] * len(seed_chunks))
            return store
        return SimpleVectorStore()

    def build_from_files(self, domain: str):
        """
        Build FAISS index from text files in knowledge_base/<domain>/raw/.
        Supports: .txt, .md, .json (as array of strings)
        """
        raw_dir = self.kb_dir / domain / "raw"
        if not raw_dir.exists():
            log.warning(f"RAG build: {raw_dir} does not exist")
            return

        chunks = []
        for fp in raw_dir.glob("*"):
            if fp.suffix == '.txt' or fp.suffix == '.md':
                text = fp.read_text(encoding='utf-8', errors='ignore')
                chunks.extend(self._chunk_text(text, source=fp.name))
            elif fp.suffix == '.json':
                data = json.loads(fp.read_text())
                if isinstance(data, list):
                    chunks.extend(data)
            log.info(f"RAG build [{domain}]: {fp.name} → {len(chunks)} total chunks")

        if not chunks:
            return

        store = SimpleVectorStore()
        embs  = self.embedder.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
        store.add(chunks, embs, [{"source": "raw_file", "domain": domain}] * len(chunks))

        idx_dir = self.kb_dir / domain
        idx_dir.mkdir(parents=True, exist_ok=True)
        store.save(idx_dir / "index.pkl")
        self.stores[domain] = store
        log.info(f"RAG build [{domain}]: saved {len(store)} chunks")

    @staticmethod
    def _chunk_text(text: str, source: str = "", chunk_size: int = 400, overlap: int = 60) -> list[str]:
        """Sliding window chunker."""
        words  = text.split()
        chunks = []
        i      = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk) > 80:
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks


# ─── Seed knowledge base (built-in, no file needed) ───────────────────────
# These chunks are used when no external corpus has been indexed.
# They represent condensed veterinary guidance from public sources.

SEED_KNOWLEDGE = {

    "avian_flu": [
        "Highly Pathogenic Avian Influenza (HPAI) H5N1 causes sudden death in poultry with up to 100% mortality. "
        "Clinical signs include depression, loss of appetite, cessation of egg production, soft-shelled eggs, "
        "respiratory distress, cyanosis of unfeathered skin, oedema of head and neck, and nervous signs. "
        "Source: WOAH Terrestrial Manual, Chapter 3.3.4.",

        "Avian influenza transmission to humans occurs mainly through direct contact with infected birds or "
        "contaminated environments. Human H5N1 infections are associated with exposure to sick or dead poultry. "
        "Person-to-person transmission is rare and has not been sustained. "
        "Source: WHO Fact Sheet on Avian Influenza.",

        "Key differential diagnoses for HPAI include Newcastle disease (paramyxovirus), infectious bronchitis, "
        "and fowl cholera. Newcastle disease also causes high mortality and respiratory/nervous signs. "
        "Laboratory confirmation (RT-PCR, virus isolation) is required for definitive diagnosis. "
        "Source: FAO Animal Health Manual No. 15.",

        "HPAI outbreak response: immediate quarantine of affected premises, culling of all birds, "
        "decontamination of equipment and facilities, restricted movement of poultry and equipment within "
        "a 10 km radius. Notify OIE/WOAH within 24 hours. "
        "Source: FAO EMPRES-i Outbreak Response Guidelines.",

        "Wild bird surveillance is critical for early HPAI detection. Migratory waterfowl (ducks, geese, "
        "shorebirds) are natural reservoir hosts. Dead wild birds near poultry farms should be reported "
        "to veterinary authorities immediately. "
        "Source: WOAH/FAO Wild Bird Surveillance Guidelines.",

        "Zoonotic risk from H5N1: case fatality rate in confirmed human cases approximately 60%. "
        "Antiviral oseltamivir (Tamiflu) effective if given within 48 hours. Healthcare workers treating "
        "suspected human cases should use full PPE including N95 respirators. "
        "Source: WHO Interim Risk Assessment H5N1.",
    ],

    "rabies": [
        "Rabies is caused by the Rabies lyssavirus and is almost universally fatal once clinical signs appear. "
        "The virus is transmitted through the saliva of infected animals, primarily via bite wounds. "
        "Incubation period: 2 weeks to 3 months in humans; can be up to 1 year. "
        "Source: WHO Rabies Fact Sheet, Updated 2023.",

        "Clinical signs in dogs: two forms. Furious rabies — excitability, aggression, altered phonation, "
        "biting inanimate objects, excessive salivation. Dumb (paralytic) rabies — progressive paralysis, "
        "dropping jaw, inability to swallow, coma. Both progress to death within 10 days of signs. "
        "Source: Merck Veterinary Manual, Rabies Chapter.",

        "Post-exposure prophylaxis (PEP) protocol: thoroughly wash wound with soap and water for 15 minutes, "
        "apply iodine or 70% ethanol. Administer rabies vaccine series (4 doses for unvaccinated: days 0, 3, 7, 14). "
        "Add rabies immunoglobulin (RIG) for category III exposures. Do not suture the wound. "
        "Source: WHO Expert Consultation on Rabies, Technical Report Series 1012.",

        "Animals at risk of being reservoir hosts: dogs (responsible for 99% of human cases), foxes, "
        "raccoons, skunks, bats, jackals, wolves, mongooses. Any warm-blooded mammal can be infected. "
        "Bats are a major source in the Americas. "
        "Source: WHO/OIE Rabies Blueprint.",

        "Oral rabies vaccination (ORV) baiting programs in wildlife have successfully reduced fox rabies "
        "in Western Europe and raccoon rabies in the US. Mass dog vaccination covering >70% of dogs in "
        "endemic areas can eliminate dog-mediated rabies. "
        "Source: FAO/WOAH/WHO Tripartite Rabies Guidance.",

        "Ten-day observation rule: a dog, cat, or ferret that bites a human should be observed for 10 days. "
        "If the animal is alive and healthy on day 10, it was not infectious at time of bite. "
        "This rule applies only to dogs, cats, and ferrets. "
        "Source: CDC Rabies Prevention Guidelines.",
    ],

    "fmd": [
        "Foot-and-mouth disease (FMD) is caused by Aphthovirus and affects cloven-hoofed animals including "
        "cattle, pigs, sheep, goats, and wild ruminants. It is one of the most contagious animal diseases. "
        "7 serotypes: O, A, C, SAT1, SAT2, SAT3, Asia1. Immunity to one does not protect against others. "
        "Source: WOAH Terrestrial Code, Section 8.8.",

        "FMD clinical signs: fever followed by vesicle formation on tongue, lips, gums, feet (coronary band, "
        "interdigital space), and teats. Animals salivate excessively, are lame, may have ruptured blisters. "
        "Mortality is low in adults but high in young animals. Milk production drops sharply. "
        "Source: FAO FMD Technical Manual.",

        "FMD spreads via: aerosol (virus can travel >60 km in wind), contact with infected animals or "
        "contaminated fomites (vehicles, equipment, clothing), and ingestion of contaminated animal products. "
        "Pigs are the amplifying host — shed large quantities of airborne virus. "
        "Source: WOAH FMD Disease Card.",

        "FMD is NOT a human health concern — humans can be infected very rarely and disease is mild. "
        "However, FMD causes devastating economic impacts. Affected countries lose export market access "
        "immediately. Cost of 2001 UK outbreak: estimated £8 billion. "
        "Source: World Bank FMD Economic Impact Analysis.",

        "FMD outbreak control: stamping out (slaughter of all susceptible animals on affected premises), "
        "emergency vaccination, movement restrictions, cleaning and disinfection. "
        "Virus is inactivated by pH < 6.0 or > 9.0, heat (60°C for 30 min), and many disinfectants. "
        "Source: FAO/WOAH FMD Progressive Control Pathway.",

        "FMD vaccination: emergency vaccination with matching serotype vaccine can reduce spread but does "
        "not prevent all infections. Vaccinated animals may become carriers. "
        "Countries may conduct 'vaccinate-to-die' or 'vaccinate-to-live' strategies. "
        "Source: WOAH/FAO FMD Reference Lab Guidelines.",
    ],

    "nipah_hendra": [
        "Nipah virus (NiV) is a paramyxovirus transmitted from fruit bats (Pteropus spp.) to pigs and humans. "
        "First identified in Malaysia 1998-99 — 265 human cases, 105 deaths (40% CFR). "
        "Pigs are an amplifying host. Bats shed virus in urine, saliva, and partially eaten fruit. "
        "Source: WHO Nipah Virus Fact Sheet.",

        "Nipah clinical signs in pigs: acute febrile illness, respiratory distress (labored breathing), "
        "neurological signs (trembling, twitching, rear leg weakness), barking cough in some pigs. "
        "High mortality in piglets, moderate in adults. Sudden onset, rapid spread through herd. "
        "Source: WHO/FAO Nipah Outbreak Investigation Report 1999.",

        "Hendra virus (HeV) is closely related to Nipah. Reservoir: large fruit bats (flying foxes) in Australia. "
        "Horses are the primary intermediate host. Human cases all involved close contact with infected horses. "
        "Case fatality rate in horses ~75%, humans ~57%. "
        "Source: Australian Department of Agriculture Hendra Factsheet.",

        "Nipah human clinical features: encephalitis, fever, headache, dizziness, altered consciousness. "
        "Bangladesh strain transmits human-to-human through close contact. "
        "No approved vaccine or specific treatment. Ribavirin has shown some benefit in vitro. "
        "Source: CDC Nipah Virus, Special Pathogens Branch.",

        "Risk factors for Nipah transmission: pig farming near bat roosting trees (especially Pteropus spp.), "
        "consumption of raw date palm sap (Bangladesh), bat-contaminated fruit, exposure to sick pig secretions. "
        "Deforestation forcing bats into agricultural areas increases risk. "
        "Source: EcoHealth Alliance Nipah Spillover Risk Model.",

        "Response to suspected Nipah outbreak: immediate isolation of affected pigs, PPE for all handlers "
        "(N95 + gloves + gown + eye protection), notify national authorities, laboratory samples to BSL-4 facility. "
        "Do not attempt field necropsy without full biosafety equipment. "
        "Source: WHO Nipah Preparedness and Response Plan.",
    ],

    "leptospirosis": [
        "Leptospirosis is caused by pathogenic Leptospira serovars and is one of the world's most widespread "
        "zoonoses. Rodents (especially rats) are the primary reservoir, shedding bacteria in urine. "
        "More than 1 million severe human cases estimated annually, >58,000 deaths. "
        "Source: WHO Leptospirosis Burden Epidemiology Reference Group (LERG) 2015.",

        "Transmission routes: contact with contaminated water, soil, or mud through cuts or mucous membranes. "
        "Occupational risk: farmers, sewage workers, slaughterhouse workers, veterinarians, military personnel. "
        "Recreational risk: swimming, wading, or white-water rafting in contaminated water. "
        "Source: CDC Leptospirosis Factsheet.",

        "Leptospirosis clinical signs in cattle: abortion storms, drop in milk production, jaundice (icterus), "
        "hemoglobinuria (bloody/dark urine), fever, anemia, eye inflammation (uveitis). "
        "Chronic carrier state — animals shed bacteria in urine for months to years. "
        "Source: Merck Veterinary Manual, Leptospirosis Chapter.",

        "Leptospirosis in dogs: most cases caused by serovars Icterohaemorrhagiae, Canicola, Pomona, Grippotyphosa. "
        "Clinical signs: acute renal failure, hepatitis, uveitis, pulmonary hemorrhage. "
        "Available vaccines protect against 4-6 serovars; do not cross-protect all serovars. "
        "Source: WSAVA Vaccination Guidelines 2022.",

        "Human leptospirosis treatment: mild cases — doxycycline 100mg twice daily for 7 days. "
        "Severe cases — IV penicillin G or ceftriaxone. "
        "Dialysis for acute renal failure. Early treatment critical — mortality high if delayed. "
        "Source: WHO Leptospirosis Diagnosis and Treatment Guidelines.",

        "Flood-associated outbreaks: standing floodwater concentrates leptospires from soil and rodent urine. "
        "Post-flood leptospirosis outbreaks documented in Thailand, Philippines, India, Brazil. "
        "Doxycycline 200mg once weekly as prophylaxis for high-risk individuals during floods. "
        "Source: CDC Leptospirosis After Floods Emergency Response.",
    ],

    "general": [
        "Zoonoses are diseases that can be transmitted between animals and humans. "
        "Over 60% of emerging infectious diseases are zoonotic, and 75% of new human pathogens "
        "originate in animals. Common zoonoses include rabies, influenza, salmonellosis, Q fever, "
        "brucellosis, and leptospirosis. Source: WHO Zoonoses Overview.",

        "One Health approach recognizes that human, animal, and environmental health are interconnected. "
        "Effective zoonotic disease control requires collaboration between public health, veterinary, "
        "and environmental sectors. Contact your local DLD (Department of Livestock Development) or "
        "public health office if you suspect a zoonotic outbreak. "
        "Source: FAO/OIE/WHO One Health Tripartite Guidance.",

        "Basic protective measures when working with animals: wear gloves and protective clothing, "
        "wash hands thoroughly after animal contact, avoid contact with sick animals without PPE, "
        "cook animal products thoroughly, report unusual illness or death patterns in animals promptly. "
        "Source: CDC General Zoonotic Disease Prevention Guidelines.",

        "When to report to authorities: sudden unexplained death of multiple animals, cluster of "
        "neurological signs in multiple animals, animals biting without provocation, unusual illness "
        "spreading rapidly through a herd or flock, concurrent illness in animals and people nearby. "
        "Contact the nearest District Livestock Development Officer (DLD) or call the Animal Disease "
        "Surveillance Center. Source: Thai Department of Livestock Development Emergency Protocol.",

        "Thailand DLD emergency contacts: Animal Disease Surveillance Center (ADSC): 02-653-4444. "
        "Regional Livestock Office handles outbreak investigation, sample collection, and laboratory testing. "
        "FAOSAP (FAO Southeast Asia) provides technical support for major outbreaks. "
        "Source: Thai DLD Emergency Response Handbook.",
    ],
}


# ─── CLI for building indexes ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description="ZoonoticSense RAG Index Builder")
    parser.add_argument("--build",  action="store_true", help="Build all domain indexes from raw files")
    parser.add_argument("--query",  type=str, help="Test retrieval with this query")
    parser.add_argument("--domain", type=str, default="avian_flu", help="Domain for test query")
    parser.add_argument("--kb-dir", type=str, default="knowledge_base")
    args = parser.parse_args()

    print("Loading embedder...")
    emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    rag = DomainRAG(kb_dir=Path(args.kb_dir), embedder=emb)

    if args.build:
        for domain in DomainRAG.DOMAINS:
            print(f"\nBuilding index for: {domain}")
            rag.build_from_files(domain)
        print("\nAll indexes built.")

    if args.query:
        print(f"\nQuery: {args.query}")
        print(f"Domain: {args.domain}\n")
        chunks = rag.retrieve(args.query, domain=args.domain, top_k=3)
        for i, chunk in enumerate(chunks, 1):
            print(f"[{i}] {chunk[:300]}...\n")
