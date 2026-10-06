import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

AZ_CHAR_MAPPING = str.maketrans({
    "ə": "e", "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "Ə": "e", "Ç": "c", "Ğ": "g", "I": "i", "Ö": "o", "Ş": "s", "Ü": "u",
})


def normalize_azerbaijani_text(text: str) -> str:
    """Normalizes Azerbaijani characters to standard Latin equivalents to reduce embedding variance."""
    return text.lower().translate(AZ_CHAR_MAPPING).strip()


class BilingualRAG:
    """Enterprise RAG engine supporting PDF ingestion via Docling and dense vector search on Vertex AI."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        embedding_model_name: str = "gemini-embedding-001",
        generation_model_name: str = "gemini-2.5-flash",
    ):
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID", "local-dev-project")
        self.location = location or os.getenv("GCP_LOCATION", "us-central1")

        # Lazy imports for Vertex AI
        from google.cloud import aiplatform
        import vertexai
        from vertexai.generative_models import GenerativeModel
        from vertexai.language_models import TextEmbeddingModel

        aiplatform.init(project=self.project_id, location=self.location)
        self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
        self.generation_model = GenerativeModel(generation_model_name)

        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Batch computes dense embeddings via Vertex AI."""
        if not texts:
            return np.empty((0, 768))
        embeddings_result = self.embedding_model.get_embeddings(texts)
        return np.array([e.values for e in embeddings_result])

    def ingest_pdf(self, pdf_path: Path, chunk_size: int = 500) -> int:
        """Converts PDF to Markdown using IBM Docling and indexes text chunks."""
        from docling.document_converter import DocumentConverter

        logger.info("Converting document %s via Docling...", pdf_path.name)
        converter = DocumentConverter()
        conversion_result = converter.convert(str(pdf_path))
        markdown_text = conversion_result.document.export_to_markdown()

        chunks = []
        start = 0
        while start < len(markdown_text):
            chunk = markdown_text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size

        self.chunks = chunks
        normalized = [normalize_azerbaijani_text(c) for c in chunks]
        logger.info("Computing embeddings for %s chunks...", len(chunks))
        self.embeddings = self.get_embeddings(normalized)
        return len(chunks)

    def ingest_faq_json(self, json_path: Path) -> int:
        """Loads structured question-answer pairs from JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = [item.get("question", "") for item in data]
        self.chunks = [item.get("answer", "") for item in data]

        normalized_questions = [normalize_azerbaijani_text(q) for q in questions]
        logger.info("Computing embeddings for %s FAQ questions...", len(questions))
        self.embeddings = self.get_embeddings(normalized_questions)
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> Tuple[List[str], np.ndarray]:
        """Calculates cosine similarity and returns top-k chunk matches."""
        if self.embeddings is None or len(self.chunks) == 0:
            return [], np.array([])

        normalized_query = normalize_azerbaijani_text(query)
        query_vector = self.get_embeddings([normalized_query])[0]

        # Cosine similarity via dot product of unit-normalized vectors
        norm_base = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_query = np.linalg.norm(query_vector)

        if norm_query == 0 or np.any(norm_base == 0):
            scores = np.dot(self.embeddings, query_vector)
        else:
            scores = np.dot(self.embeddings / norm_base, query_vector / norm_query)

        ranked_indices = np.argsort(scores)[-top_k:][::-1]
        top_chunks = [self.chunks[i] for i in ranked_indices]
        return top_chunks, scores[ranked_indices]

    def answer_query(self, user_query: str, score_threshold: float = 0.65, language_target: str = "ru") -> str:
        """Retrieves grounded context and queries Gemini model."""
        matched_chunks, scores = self.search(user_query, top_k=5)

        if len(scores) == 0 or scores[0] < score_threshold:
            if language_target == "az":
                return "Bağışlayın, daxili bazamızda bu suala uyğun məlumat tapılmadı."
            return "Извините, во внутренней базе нет информации по вашему вопросу."

        context = "\n\n---\n\n".join(matched_chunks)
        lang_prompt = "Ответь на азербайджанском языке." if language_target == "az" else "Ответь на русском языке."

        prompt = f"""
Ты корпоративный ассистент компании. 
Используй ТОЛЬКО предоставленный ниже контекст, чтобы ответить на вопрос пользователя.
Если в контексте нет ответа, прямо скажи, что не обладаешь данной информацией.

Контекст:
{context}

Вопрос пользователя:
{user_query}

Язык ответа:
{lang_prompt}
"""
        response = self.generation_model.generate_content(prompt)
        return response.text.strip()


def run_cli():
    base_dir = Path(__file__).resolve().parent
    doc_path = base_dir / "Document" / "Daxili_Reglament_FAQ.pdf"

    print("\n--- Корпоративный RAG-ассистент (Vertex AI + IBM Docling) ---")
    print("1. Поиск по корпоративному регламенту (Daxili_Reglament_FAQ.pdf)")
    print("2. FAQ: Вопросы на азербайджанском языке (questaz.json)")
    print("3. FAQ: Вопросы на русском языке (quest.json)")

    choice = input("\nВыберите режим работы [1/2/3]: ").strip()
    bot = BilingualRAG()

    if choice == "1":
        if not doc_path.exists():
            print(f"Ошибка: файл {doc_path} не найден.")
            return
        bot.ingest_pdf(doc_path)
        lang = input("Язык ответов [az/ru] (по умолчанию az): ").strip().lower() or "az"
    elif choice == "2":
        bot.ingest_faq_json(base_dir / "questaz.json")
        lang = "az"
    elif choice == "3":
        bot.ingest_faq_json(base_dir / "quest.json")
        lang = "ru"
    else:
        print("Неверный выбор.")
        return

    print("\nБаза проиндексирована. Задавайте вопросы (для выхода введите 'выход' или 'exit'):")
    while True:
        query = input("\nВы: ").strip()
        if query.lower() in ["выход", "exit", "q"]:
            break
        if not query:
            continue
        answer = bot.answer_query(query, language_target=lang)
        print(f"\nБот: {answer}")


if __name__ == "__main__":
    run_cli()
