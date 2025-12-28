"""
Vector store management for document embeddings and retrieval.
Provides abstraction over different vector store implementations (FAISS, Chroma, etc.).
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .logging_config import get_logger


class EmbeddingProvider:
    """Abstract base class for embedding providers."""

    def __init__(self, **kwargs):
        self.config = kwargs

    def get_embeddings(self):
        """Get the embeddings instance."""
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def get_embeddings(self):
        api_key = self.config.get('api_key')
        if not api_key:
            raise ValueError("OpenAI API key is required for embeddings")

        model = self.config.get('model', 'text-embedding-3-small')
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key
        )


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """OpenRouter embedding provider using OpenAI-compatible API."""

    def get_embeddings(self):
        api_key = self.config.get('api_key')
        if not api_key:
            raise ValueError("OpenRouter API key is required for embeddings")

        model = self.config.get('model', 'text-embedding-3-small')
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )

logger = get_logger(__name__)


class VectorStoreProvider(ABC):
    """Abstract base class for vector store providers."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 5, **kwargs) -> List[tuple]:
        """Search for similar documents with scores."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        """Get all documents in the store."""
        pass


class FAISSProvider(VectorStoreProvider):
    """FAISS vector store provider."""

    def __init__(self, embeddings, **kwargs):
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None

    def add_documents(self, documents: List[Document]) -> None:
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_score(self, query: str, k: int = 5, **kwargs) -> List[tuple]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search_with_score(query, k=k, **kwargs)

    def save(self, path: str) -> None:
        if self.vector_store:
            self.vector_store.save_local(path)

    def load(self, path: str) -> None:
        try:
            self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception:
            # Create empty store if loading fails
            self.vector_store = FAISS.from_texts([""], self.embeddings)

    def delete(self, ids: List[str]) -> None:
        # FAISS doesn't support deletion easily
        logger.warning("FAISS does not support document deletion. Consider using Chroma for this feature.")

    def get_all_documents(self) -> List[Document]:
        if self.vector_store is None:
            return []
        # This is approximate - FAISS doesn't provide easy access to all documents
        return self.vector_store.similarity_search("", k=10000)


class ChromaProvider(VectorStoreProvider):
    """Chroma vector store provider."""

    def __init__(self, embeddings, collection_name: str = "documents", **kwargs):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.persist_directory = kwargs.get('persist_directory', './chroma_db')
        self.vector_store: Optional[Chroma] = None

    def add_documents(self, documents: List[Document]) -> None:
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents,
                self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        else:
            self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_score(self, query: str, k: int = 5, **kwargs) -> List[tuple]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search_with_score(query, k=k, **kwargs)

    def save(self, path: str) -> None:
        if self.vector_store:
            self.vector_store.persist()

    def load(self, path: str) -> None:
        self.vector_store = Chroma(
            persist_directory=path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

    def delete(self, ids: List[str]) -> None:
        if self.vector_store:
            self.vector_store.delete(ids)

    def get_all_documents(self) -> List[Document]:
        if self.vector_store is None:
            return []

        # Get data from Chroma
        data = self.vector_store.get(include=['documents', 'metadatas'])

        documents = []
        docs_content = data.get('documents', [])
        docs_metadata = data.get('metadatas', [])

        # Reconstruct Document objects
        for content, metadata in zip(docs_content, docs_metadata):
            doc = Document(page_content=content, metadata=metadata or {})
            documents.append(doc)

        return documents


class VectorManager:
    """
    Manages vector store operations with support for multiple providers.
    Handles document ingestion, retrieval, and metadata management.
    """

    PROVIDERS = {
        'faiss': FAISSProvider,
        'chroma': ChromaProvider,
    }

    EMBEDDING_PROVIDERS = {
        'openai': OpenAIEmbeddingProvider,
        'openrouter': OpenRouterEmbeddingProvider,
    }

    def __init__(self,
                 provider: str = 'faiss',
                 store_path: str = './data/vector_store',
                 embedding_provider: str = 'openai',
                 embedding_model: str = 'text-embedding-3-small',
                 api_key: Optional[str] = None,
                 **provider_kwargs):
        """
        Initialize the vector manager.

        Args:
            provider: Vector store provider ('faiss' or 'chroma')
            store_path: Path to store vector database
            embedding_provider: Embedding provider ('openai' or 'openrouter')
            embedding_model: Embedding model to use
            api_key: API key for the embedding provider
            **provider_kwargs: Additional provider-specific arguments
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.embedding_provider_name = embedding_provider
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.provider_kwargs = provider_kwargs
        self._embeddings = None  # Lazy initialization

        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Available: {list(self.PROVIDERS.keys())}")

        if embedding_provider not in self.EMBEDDING_PROVIDERS:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}. Available: {list(self.EMBEDDING_PROVIDERS.keys())}")

        self.provider_name = provider
        self.provider = None  # Lazy initialization

        logger.info(f"VectorManager initialized with {provider} provider and {embedding_provider} embeddings at {store_path}")

    @property
    def embeddings(self):
        """Get embeddings instance, creating it if necessary."""
        if self._embeddings is None:
            try:
                embedding_provider_class = self.EMBEDDING_PROVIDERS[self.embedding_provider_name]
                embedding_provider = embedding_provider_class(
                    api_key=self.api_key,
                    model=self.embedding_model
                )
                self._embeddings = embedding_provider.get_embeddings()
                logger.info(f"Created embeddings using {self.embedding_provider_name} provider")
            except Exception as e:
                logger.error(f"Failed to create embeddings: {e}")
                raise ValueError(f"API key not configured for {self.embedding_provider_name}. Please configure API key in settings.")
        return self._embeddings

    @property
    def provider(self):
        """Get provider instance, creating it if necessary."""
        if self._provider is None:
            self._provider = self.PROVIDERS[self.provider_name](self.embeddings, **self.provider_kwargs)
            # Load existing vector store when provider is first accessed
            self._load_store()
        return self._provider

    @provider.setter
    def provider(self, value):
        """Set provider instance."""
        self._provider = value

    def _load_store(self):
        """Load the vector store from disk."""
        try:
            store_dir = str(self.store_path / self.provider_name)
            self._provider.load(store_dir)
            logger.info(f"Loaded existing vector store from {store_dir}")
        except Exception as e:
            logger.info(f"No existing vector store found or failed to load: {e}. Creating new store.")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add
        """
        if not documents:
            return

        try:
            self.provider.add_documents(documents)
            self._save_store()
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def search_similar(self,
                      query: str,
                      k: int = 5,
                      score_threshold: float = 0.0,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            filters: Optional metadata filters

        Returns:
            List of search results with content, metadata, and scores
        """
        try:
            # Get results with scores
            docs_with_scores = self.provider.similarity_search_with_score(query, k=k * 2)  # Get more for filtering

            results = []
            for doc, score in docs_with_scores:
                # Apply score threshold
                if score < score_threshold:
                    continue

                # Apply metadata filters
                if filters:
                    if not self._matches_filters(doc.metadata, filters):
                        continue

                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'file_name': doc.metadata.get('file_name', 'Unknown'),
                    'file_path': doc.metadata.get('file_path', 'Unknown')
                }
                results.append(result)

                # Stop when we have enough results
                if len(results) >= k:
                    break

            logger.info(f"Vector search returned {len(results)} results for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches the given filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def hybrid_search(self,
                     query: str,
                     k: int = 5,
                     semantic_weight: float = 0.7,
                     keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword-based retrieval.

        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic similarity (0.0 to 1.0)
            keyword_weight: Weight for keyword matching (0.0 to 1.0)

        Returns:
            List of search results with combined scores
        """
        try:
            # Get semantic results
            semantic_results = self.search_similar(query, k=k*2)

            # Get keyword-based results (simple text matching)
            keyword_results = self._keyword_search(query, k=k*2)

            # Combine results
            combined_scores = {}

            # Add semantic scores
            for result in semantic_results:
                doc_id = result['metadata'].get('file_hash', '') + str(result['metadata'].get('chunk_id', 0))
                combined_scores[doc_id] = {
                    'result': result,
                    'semantic_score': result['score'],
                    'keyword_score': 0.0
                }

            # Add keyword scores
            for result in keyword_results:
                doc_id = result['metadata'].get('file_hash', '') + str(result['metadata'].get('chunk_id', 0))
                if doc_id in combined_scores:
                    combined_scores[doc_id]['keyword_score'] = result['score']
                else:
                    combined_scores[doc_id] = {
                        'result': result,
                        'semantic_score': 0.0,
                        'keyword_score': result['score']
                    }

            # Calculate combined scores and sort
            final_results = []
            for doc_id, scores in combined_scores.items():
                combined_score = (
                    semantic_weight * scores['semantic_score'] +
                    keyword_weight * scores['keyword_score']
                )
                result = scores['result'].copy()
                result['score'] = combined_score
                result['search_type'] = 'hybrid'
                final_results.append(result)

            # Sort by combined score and return top k
            final_results.sort(key=lambda x: x['score'], reverse=True)
            return final_results[:k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _keyword_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform simple keyword-based search."""
        try:
            # Get all documents (this is inefficient for large stores)
            all_docs = self.provider.get_all_documents()

            query_words = set(query.lower().split())
            results = []

            for doc in all_docs:
                content = doc.page_content.lower()
                # Simple word matching
                matching_words = sum(1 for word in query_words if word in content)
                if matching_words > 0:
                    score = matching_words / len(query_words)  # Normalized score
                    result = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': score,
                        'file_name': doc.metadata.get('file_name', 'Unknown'),
                        'file_path': doc.metadata.get('file_path', 'Unknown')
                    }
                    results.append(result)

            # Sort by score and return top k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        try:
            all_docs = self.provider.get_all_documents()
            return len(all_docs)
        except Exception:
            return 0

    def get_files_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all ingested files.

        Returns:
            List of file metadata
        """
        try:
            all_docs = self.provider.get_all_documents()

            files = {}
            for doc in all_docs:
                file_hash = doc.metadata.get('file_hash')
                if file_hash and file_hash not in files:
                    files[file_hash] = {
                        'file_path': doc.metadata.get('file_path'),
                        'file_name': doc.metadata.get('file_name'),
                        'file_size': doc.metadata.get('file_size'),
                        'file_modified': doc.metadata.get('file_modified'),
                        'mime_type': doc.metadata.get('mime_type'),
                        'ingestion_time': doc.metadata.get('ingestion_time'),
                        'chunks_count': 0
                    }
                if file_hash in files:
                    files[file_hash]['chunks_count'] += 1

            return list(files.values())

        except Exception as e:
            logger.error(f"Failed to get files info: {e}")
            return []

    def delete_documents(self, filters: Dict[str, Any]) -> int:
        """
        Delete documents matching the given filters.

        Args:
            filters: Metadata filters to match documents for deletion

        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.provider.get_all_documents()

            # Find documents to delete
            to_delete = []
            for doc in all_docs:
                if self._matches_filters(doc.metadata, filters):
                    # Most providers need document IDs for deletion
                    # This is a simplified approach
                    to_delete.append(doc)

            if to_delete:
                # For providers that support it, delete by IDs
                if hasattr(self.provider, 'delete'):
                    # This would need to be implemented based on provider capabilities
                    logger.warning("Document deletion not fully implemented for current provider")

            logger.info(f"Would delete {len(to_delete)} documents (not implemented)")
            return 0

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return 0

    def _save_store(self):
        """Save the vector store to disk."""
        try:
            store_dir = str(self.store_path / self.provider_name)
            self.provider.save(store_dir)
            logger.debug("Vector store saved to disk")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise

    def rebuild_index(self):
        """Rebuild the vector store index."""
        try:
            self._save_store()
            logger.info("Vector store index rebuilt")
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            raise
