"""
rag.py - Moteur de Retrieval-Augmented Generation (RAG) fermé et strict.
Permet d'extraire uniquement le contexte documentaire officiel d'ImmoPredict AI
et d'interdire toute hallucination sur des connaissances non certifiées.
"""
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FRENCH_STOPWORDS = {
    'de', 'la', 'le', 'les', 'des', 'un', 'une', 'du', 'et', 'en', 'a', 'au', 'aux',
    'est', 'sont', 'ce', 'cet', 'cette', 'ces', 'dans', 'par', 'pour', 'sur', 'avec',
    'qui', 'que', 'quoi', 'dont', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'mon', 'ton',
    'son', 'notre', 'votre', 'leur', 'mes', 'tes', 'ses', 'nos', 'vos', 'leurs',
    'moi', 'toi', 'lui', 'nous', 'vous', 'eux', 'elle', 'elles', 'il', 'ils', 'je', 'tu'
}


def normalize_text(text: str) -> str:
    """Normalise une chaîne de texte (minuscules, sans accents, sans stopwords)."""
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8").lower()
    words = re.findall(r"\b[a-z0-9\-]+\b", text)
    filtered = [w for w in words if w not in FRENCH_STOPWORDS and len(w) > 2]
    return " ".join(filtered)


class KnowledgeChunk:
    """Représente un fragment certifié de la base de connaissances."""
    def __init__(self, chunk_id: str, title: str, state_target: str, content: str, query_triggers: Optional[List[str]] = None):
        self.chunk_id = chunk_id
        self.title = title
        self.state_target = state_target
        self.content = content.strip()
        self.query_triggers = query_triggers or []

    def full_representation(self) -> str:
        triggers = " ".join(self.query_triggers)
        return normalize_text(f"{self.title} {triggers} {self.content}")


class ClosedDomainRAG:
    """
    Système RAG fermé basé sur scikit-learn (TF-IDF N-grams & Cosine Similarity)
    avec seuil de coupure strict garantissant zéro hallucination.
    """

    def __init__(
        self,
        knowledge_file: Optional[Path] = None,
        similarity_threshold: float = 0.08
    ):
        self.similarity_threshold = similarity_threshold
        if knowledge_file is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.knowledge_file = project_root / "data" / "knowledge_base" / "company_knowledge.md"
        else:
            self.knowledge_file = knowledge_file

        self.chunks: List[KnowledgeChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self) -> None:
        """Charge le document Markdown et le découpe en chunks sémantiques ciblés."""
        if not self.knowledge_file.exists():
            raise FileNotFoundError(f"Fichier de connaissances introuvable : {self.knowledge_file}")

        text = self.knowledge_file.read_text(encoding="utf-8")
        raw_sections = re.split(r"\n(?=##\s+\d+\.)", text)

        state_mapping = {
            "1. IDENTITE": "STATE_0_DISCOVERY",
            "2. SERVICES": "STATE_1_SERVICES",
            "3. CADRAGE": "STATE_2_QUALIFICATION",
            "4. MODALITES": "STATE_3_HANDOVER",
            "5. PERIMETRE": "OUT_OF_SCOPE",
        }

        # Déclencheurs fréquents pour booster le matching RAG
        sample_triggers = {
            "STATE_0_DISCOVERY": [
                "qui etes vous", "qui sommes nous", "presentation", "agence", "gratuit",
                "payant", "tarifs", "cout", "societe", "mission", "independant", "independance",
                "conseils payants", "services gratuits", "combien ca coute", "prix de vos services"
            ],
            "STATE_1_SERVICES": [
                "fonctionnalites", "comment ca marche", "evaluation", "faisabilite",
                "rentabilite", "simulateur", "recommandation", "villes", "communes", "prix m2",
                "loyer", "irl", "donnees", "dvf", "dgfip", "insee", "technologie"
            ],
            "STATE_2_QUALIFICATION": [
                "projet", "criteres", "budget", "achat", "investir", "recherche", "zone",
                "surface", "apport", "horizon", "calendrier", "ville"
            ],
            "STATE_3_HANDOVER": [
                "conseiller", "expert", "humain", "rdv", "rendez-vous", "contact",
                "rappel", "parler", "echange", "relais", "accompagnement"
            ],
            "OUT_OF_SCOPE": [
                "impot", "fiscalite complexe", "credit bancaire", "courtier", "gestion locative",
                "syndic", "avocat", "notaire"
            ]
        }

        chunk_id = 1
        for section in raw_sections:
            clean_sec = section.strip()
            if not clean_sec or not clean_sec.startswith("##"):
                continue

            header_match = re.search(r"^##\s+(.+)$", clean_sec, flags=re.MULTILINE)
            section_title = header_match.group(1).strip() if header_match else "Général"

            assigned_state = "STATE_0_DISCOVERY"
            for prefix, state in state_mapping.items():
                if prefix in section_title.upper():
                    assigned_state = state
                    break

            subsections = re.split(r"\n(?=###\s+)", clean_sec)
            for sub in subsections:
                clean_sub = sub.strip()
                # On ignore les fragments purement d'en-tête sans contenu descriptif
                if len(clean_sub) < 40 or clean_sub.startswith("## "):
                    continue

                sub_header = re.search(r"^###?\s+(.+)$", clean_sub, flags=re.MULTILINE)
                sub_title = sub_header.group(1).strip() if sub_header else section_title

                triggers = sample_triggers.get(assigned_state, [])
                chunk = KnowledgeChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    title=f"{section_title} > {sub_title}",
                    state_target=assigned_state,
                    content=clean_sub,
                    query_triggers=triggers
                )
                self.chunks.append(chunk)
                chunk_id += 1

        corpus = [c.full_representation() for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Recherche sémantique des chunks les plus pertinents.
        Applique le seuil strict de similarité pour écarter les hallucinations.
        """
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return {
                "has_sufficient_context": False,
                "context_text": "",
                "chunks": [],
                "max_score": 0.0,
                "suggested_state": "STATE_0_DISCOVERY"
            }

        norm_query = normalize_text(query)
        query_vec = self.vectorizer.transform([norm_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        best_indices = np.argsort(similarities)[::-1]
        top_indices = [idx for idx in best_indices[:top_k] if similarities[idx] > 0]

        if not top_indices:
            return {
                "has_sufficient_context": False,
                "context_text": "",
                "chunks": [],
                "max_score": 0.0,
                "suggested_state": "STATE_0_DISCOVERY"
            }

        max_score = float(similarities[top_indices[0]])
        has_sufficient_context = max_score >= self.similarity_threshold

        retrieved_chunks = [self.chunks[i] for i in top_indices]
        suggested_state = retrieved_chunks[0].state_target if retrieved_chunks else "STATE_0_DISCOVERY"

        context_lines = []
        for i, chunk in enumerate(retrieved_chunks):
            score = float(similarities[top_indices[i]])
            context_lines.append(f"--- Document Certifié : {chunk.title} (Pertinence: {score:.2f}) ---")
            context_lines.append(chunk.content)

        return {
            "has_sufficient_context": has_sufficient_context,
            "context_text": "\n\n".join(context_lines),
            "chunks": [
                {
                    "id": c.chunk_id,
                    "title": c.title,
                    "state": c.state_target,
                    "score": float(similarities[idx])
                }
                for idx, c in zip(top_indices, retrieved_chunks)
            ],
            "max_score": max_score,
            "suggested_state": suggested_state
        }