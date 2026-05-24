from crewai.tools import BaseTool
from typing import Type, ClassVar, Optional
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


class PDFSegmentInput(BaseModel):
    """Input schema for PDF segmentation."""
    pdf_path: str = Field(..., description="Path to the PDF file")
    k: Optional[int] = Field(None, description="Number of clusters (default: elbow heuristic)")


class PDFSegmentTool(BaseTool):
    name: str = "pdf_segment_tool"
    description: str = "Extract paragraphs from a lab PDF, embed with MiniLM, cluster with K-Means, return labeled segments."
    args_schema: Type[BaseModel] = PDFSegmentInput

    # Singleton model — load once
    _model: ClassVar[Optional[SentenceTransformer]] = None

    @classmethod
    def _get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model

    def _run(self, pdf_path: str, k: Optional[int] = None) -> list:
        import fitz, re
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Step 1: Extract + clean (from notebook Cell 1)
        doc = fitz.open(pdf_path)
        chunks = []
        for page in doc:
            raw = page.get_text("text")
            for para in raw.split('\n\n'):
                clean = re.sub(r'\s+', ' ', para).strip()
                if len(clean) > 200:
                    chunks.append(clean)

        # Step 2: Embed (from notebook Cell 2)
        model = self._get_model()
        embs = model.encode(chunks, show_progress_bar=True)

        # Step 3: Determine K (elbow or default 3, from notebook)
        if k is None:
            k = min(3, len(chunks))

        # Step 4: Cluster + label (from notebook cluster loop)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embs)
        segments = []
        for i in range(k):
            mask = labels == i
            centroid = km.cluster_centers_[i].reshape(1, -1)
            sims = cosine_similarity(embs[mask], centroid).flatten()
            best = np.where(mask)[0][np.argmax(sims)]
            title = chunks[best][:60]
            members = [chunks[j] for j, lbl in enumerate(labels) if lbl == i]
            segments.append({"title": title, "paragraphs": members})
        return segments