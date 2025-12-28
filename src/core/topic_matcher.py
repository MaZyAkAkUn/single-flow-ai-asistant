"""
Topic matching using semantic similarity and embeddings.
Provides intelligent topic deduplication and similarity detection.
"""
import uuid
from typing import List, Tuple, Optional, Dict, Any
from difflib import SequenceMatcher

from ..data.schemas import TopicInfo
from .vector_manager import VectorManager
from .logging_config import get_logger

logger = get_logger(__name__)


class TopicMatcher:
    """
    Intelligent topic matching using embeddings and semantic similarity.
    Supports centroid-based topic clustering and merging.
    """

    # Thresholds for topic merging
    MERGE_THRESHOLD = 0.78  # Automatic merge if similarity >= 0.78
    MERGE_THRESHOLD_HIGH = 0.72  # Candidate merge if 0.72 <= similarity < 0.78
    NEW_TOPIC_MIN_TOKENS = 4  # Don't create topics for very short messages

    def __init__(self, vector_manager: VectorManager, llm_adapter=None):
        """
        Initialize topic matcher.

        Args:
            vector_manager: VectorManager instance for embeddings
            llm_adapter: Optional LLM adapter for advanced comparison
        """
        self.vector_manager = vector_manager
        self.llm_adapter = llm_adapter
        self.embedding_cache: Dict[str, List[float]] = {}  # text -> embedding
        self.centroid_cache: Dict[str, List[float]] = {}  # centroid_id -> embedding

        logger.info("TopicMatcher initialized with centroid-based matching support")

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text, with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache first
        cache_key = text.lower().strip()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            # Get embedding using vector manager
            embedding = self.vector_manager.embeddings.embed_query(text)

            # Cache the result
            self.embedding_cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Failed to get embedding for text: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536  # Typical OpenAI embedding dimension

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)

            if not emb1 or not emb2:
                return 0.0

            return self._cosine_similarity(emb1, emb2)

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            # Fallback to string similarity
            return self._string_similarity(text1, text2)

    def find_similar_topics(self, new_topic: str, existing_topics: List[str],
                           threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find existing topics similar to the new topic.

        Args:
            new_topic: New topic name to match
            existing_topics: List of existing topic names
            threshold: Minimum similarity threshold

        Returns:
            List of (topic_name, similarity_score) tuples, sorted by similarity
        """
        similarities = []

        for existing_topic in existing_topics:
            similarity = self.calculate_similarity(new_topic, existing_topic)
            if similarity >= threshold:
                similarities.append((existing_topic, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(similarities)} similar topics for '{new_topic}' above threshold {threshold}")
        return similarities

    def should_merge_topics(self, new_topic: TopicInfo, existing_topic: TopicInfo) -> bool:
        """
        Decide if new topic should be merged with existing topic.

        Args:
            new_topic: New topic information
            existing_topic: Existing topic information

        Returns:
            True if topics should be merged
        """
        # Quick domain check
        if new_topic.domain and existing_topic.domain:
            if new_topic.domain.lower() != existing_topic.domain.lower():
                return False

        # String similarity check
        string_sim = self._string_similarity(new_topic.topic_name, existing_topic.topic_name)
        if string_sim < 0.3:  # Too different for basic string matching
            return False

        # Semantic similarity
        semantic_sim = self.calculate_similarity(new_topic.topic_name, existing_topic.topic_name)

        # Decision logic
        if semantic_sim >= 0.9:  # Very similar
            return True
        elif semantic_sim >= 0.8:  # Moderately similar, check with LLM if available
            if self.llm_adapter:
                return self._llm_topic_comparison(new_topic.topic_name, existing_topic.topic_name)
            else:
                return True  # Conservative approach without LLM
        else:
            return False

    def generate_topic_id(self) -> str:
        """
        Generate a unique topic ID.

        Returns:
            Unique topic identifier
        """
        return f"topic_{uuid.uuid4().hex[:16]}"

    def extract_topic_keywords(self, topic_name: str) -> List[str]:
        """
        Extract key terms from topic name for better matching.

        Args:
            topic_name: Topic name

        Returns:
            List of keywords
        """
        # Simple keyword extraction (can be enhanced with NLP)
        words = topic_name.lower().split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        import numpy as np

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity using multiple methods.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not s1 or not s2:
            return 0.0

        # Jaccard similarity on words
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        jaccard = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0

        # Levenshtein similarity
        levenshtein = SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

        # Weighted combination
        return (jaccard * 0.6) + (levenshtein * 0.4)

    def _llm_topic_comparison(self, topic1: str, topic2: str) -> bool:
        """
        Use LLM to compare topics when similarity is borderline.

        Args:
            topic1: First topic name
            topic2: Second topic name

        Returns:
            True if topics are the same/related
        """
        if not self.llm_adapter:
            return False

        prompt = f"""Compare these two conversation topics and determine if they refer to the same subject:

TOPIC 1: "{topic1}"
TOPIC 2: "{topic2}"

Are these topics essentially the same conversation subject, just worded differently?
Answer only 'yes' or 'no'."""

        try:
            response = self.llm_adapter.generate_response(prompt)
            response_lower = response.lower().strip()

            is_same = 'yes' in response_lower[:10]  # Check first few characters

            logger.info(f"LLM topic comparison: '{topic1}' vs '{topic2}' -> {is_same}")
            return is_same

        except Exception as e:
            logger.error(f"LLM topic comparison failed: {e}")
            return False  # Conservative fallback

    def clear_cache(self):
        """Clear embedding cache to free memory."""
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"Cleared embedding cache ({cache_size} entries)")

    def find_similar_topics_by_centroid(self, message_embedding: List[float],
                                       existing_topics: List[TopicInfo],
                                       top_k: int = 5) -> List[Tuple[TopicInfo, float]]:
        """
        Find existing topics similar to a message embedding using centroids.

        Args:
            message_embedding: Embedding of the new message
            existing_topics: List of existing TopicInfo objects
            top_k: Number of top similar topics to return

        Returns:
            List of (TopicInfo, similarity_score) tuples, sorted by similarity
        """
        similarities = []

        for topic in existing_topics:
            if not topic.centroid_embedding:
                # Fallback to topic name similarity if no centroid
                similarity = self.calculate_similarity("", topic.topic_name)
            else:
                similarity = self._cosine_similarity(message_embedding, topic.centroid_embedding)

            similarities.append((topic, similarity))

        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Found top {min(top_k, len(similarities))} similar topics by centroid")
        return similarities[:top_k]

    def should_merge_with_topic(self, message_embedding: List[float], topic: TopicInfo) -> Tuple[bool, str]:
        """
        Determine if a message should be merged with an existing topic.

        Args:
            message_embedding: Embedding of the new message
            topic: Existing topic to check

        Returns:
            Tuple of (should_merge, merge_type) where merge_type is 'automatic', 'candidate', or 'no_merge'
        """
        if not topic.centroid_embedding:
            return False, 'no_merge'

        similarity = self._cosine_similarity(message_embedding, topic.centroid_embedding)

        if similarity >= self.MERGE_THRESHOLD:
            return True, 'automatic'
        elif similarity >= self.MERGE_THRESHOLD_HIGH:
            return False, 'candidate'  # Could be merged after review
        else:
            return False, 'no_merge'

    def update_topic_centroid(self, topic: TopicInfo, new_embedding: List[float],
                             message_text: str, message_timestamp: str) -> TopicInfo:
        """
        Update topic centroid with new message embedding using incremental averaging.

        Args:
            topic: TopicInfo to update
            new_embedding: New message embedding
            message_text: Text of the new message
            message_timestamp: Timestamp of the message

        Returns:
            Updated TopicInfo
        """
        import numpy as np
        from datetime import datetime

        # Update message count and last_seen
        topic.message_count += 1
        topic.last_seen = datetime.now()

        # Update centroid incrementally
        if not topic.centroid_embedding:
            # First embedding becomes the centroid
            topic.centroid_embedding = new_embedding
        else:
            # Incremental centroid update: centroid = (centroid * n + new) / (n + 1)
            n = topic.message_count - 1  # Previous count
            old_centroid = np.array(topic.centroid_embedding)
            new_emb = np.array(new_embedding)
            updated_centroid = (old_centroid * n + new_emb) / (n + 1)
            topic.centroid_embedding = updated_centroid.tolist()

        # Add to recent embeddings (keep last 50)
        recent_emb = {
            "timestamp": message_timestamp,
            "embedding": new_embedding
        }
        topic.recent_embeddings.append(recent_emb)
        if len(topic.recent_embeddings) > 50:
            topic.recent_embeddings = topic.recent_embeddings[-50:]

        # Add to representative messages (keep top 5 by recency)
        rep_msg = {
            "id": f"msg_{topic.message_count}",
            "text": message_text[:500],  # Truncate long messages
            "timestamp": message_timestamp
        }
        topic.representative_messages.append(rep_msg)
        if len(topic.representative_messages) > 5:
            topic.representative_messages = topic.representative_messages[-5:]

        # Update topic confidence based on message count and consistency
        topic.confidence = min(0.95, 0.5 + (topic.message_count * 0.05))

        topic.updated_at = datetime.now()

        logger.debug(f"Updated centroid for topic '{topic.topic_name}' with {topic.message_count} messages")
        return topic

    def should_create_new_topic(self, message_text: str) -> bool:
        """
        Determine if a message should create a new topic based on length and content.

        Args:
            message_text: The message text

        Returns:
            True if should create new topic
        """
        # Tokenize roughly (split by spaces and punctuation)
        import re
        tokens = re.findall(r'\b\w+\b', message_text.lower())
        token_count = len(tokens)

        # Don't create topics for very short messages
        if token_count < self.NEW_TOPIC_MIN_TOKENS:
            return False

        # Don't create topics for common conversational phrases
        conversational_phrases = [
            "yes", "no", "okay", "thanks", "thank you", "please", "sorry",
            "hello", "hi", "bye", "goodbye", "good morning", "good night"
        ]

        if message_text.lower().strip() in conversational_phrases:
            return False

        return True

    def detect_language(self, text: str) -> str:
        """
        Simple language detection for topic metadata.

        Args:
            text: Text to analyze

        Returns:
            Language code ('en', 'uk', 'es', etc.)
        """
        # Simple heuristics - can be enhanced with langdetect library
        text_lower = text.lower()

        # Ukrainian indicators
        if any(word in text_lower for word in ['як', 'що', 'де', 'коли', 'хто', 'він', 'вона', 'вони']):
            return 'uk'

        # Spanish indicators
        if any(word in text_lower for word in ['el', 'la', 'los', 'las', 'es', 'son', 'está', 'están']):
            return 'es'

        # Default to English
        return 'en'

    def extract_topic_keywords(self, text: str, language: str = 'en') -> List[str]:
        """
        Extract keywords from text for topic enrichment.

        Args:
            text: Text to analyze
            language: Language code

        Returns:
            List of extracted keywords
        """
        import re

        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())

        # Language-specific stop words
        stop_words = {
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those'},
            'uk': {'і', 'або', 'але', 'в', 'на', 'у', 'до', 'для', 'з', 'від', 'це', 'той', 'та', 'чи'},
            'es': {'el', 'la', 'los', 'las', 'y', 'o', 'pero', 'en', 'sobre', 'a', 'para', 'de', 'con', 'por', 'es', 'son', 'está', 'están'}
        }

        language_stops = stop_words.get(language, stop_words['en'])

        # Extract meaningful keywords
        keywords = []
        for word in words:
            if len(word) > 2 and word not in language_stops:
                keywords.append(word)

        # Return top keywords by frequency (simple approach)
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, _ in keyword_counts.most_common(10)]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "embedding_cache_size": len(self.embedding_cache),
            "centroid_cache_size": len(self.centroid_cache),
            "cache_memory_mb": (len(self.embedding_cache) * 1536 + len(self.centroid_cache) * 1536) * 4 / (1024 * 1024)
        }
