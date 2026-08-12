"""
Embedding provider abstraction.

Why this file exists:
Every embedding provider (Voyage, OpenAI, a local model) turns text into a
list of numbers ("a vector"), but each has a slightly different API.
This file hides that difference behind ONE simple interface: `embed(texts)`.

The rest of our pipeline never needs to know or care which provider is
actually being used underneath. Switching providers = changing one line
in your .env file, nothing else.
"""

import os
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Every embedding provider must implement this one method."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Takes a list of text strings, returns a list of vectors (one per text)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """How many numbers are in each vector (needed to set up the vector store)."""
        raise NotImplementedError


class VoyageEmbedder(EmbeddingProvider):
    """
    Production-grade embedding provider using Voyage AI.
    This is Anthropic's recommended embedding partner and what you should
    use once you're running this outside this sandbox with a real API key.
    """

    def __init__(self, model: str = "voyage-3.5", api_key: str | None = None):
        import voyageai

        self.client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))
        self.model = model
        self._dimension = 1024  # voyage-3.5 default output size

    def embed(self, texts: list[str]) -> list[list[float]]:
        # input_type="document" tells Voyage these are documents being stored
        # (it has a separate mode for embedding a search query, which we use in retrieval.py)
        result = self.client.embed(texts, model=self.model, input_type="document")
        return result.embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedder(EmbeddingProvider):
    """
    Free, local, no-API-key embedding provider using fastembed (runs entirely
    on this machine, ONNX-based, no torch needed). Good for development,
    testing, and learning without spending API credits.

    Quality is a bit lower than Voyage/OpenAI's cloud models, but the
    mechanics -- chunking, storage, retrieval, grounding -- are IDENTICAL.
    That's the whole point of the abstraction.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model)
        self._dimension = 384  # bge-small output size

    def embed(self, texts: list[str]) -> list[list[float]]:
        # fastembed returns a generator of numpy arrays; convert to plain lists
        return [vec.tolist() for vec in self.model.embed(texts)]

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedder() -> EmbeddingProvider:
    """
    Factory function: reads EMBEDDING_PROVIDER from environment and returns
    the right embedder. This is the ONLY place that decides which provider
    to use -- everything else just calls get_embedder().embed(...).
    """
    provider = os.environ.get("EMBEDDING_PROVIDER", "local").lower()

    if provider == "voyage":
        return VoyageEmbedder()
    elif provider == "local":
        return LocalEmbedder()
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")
