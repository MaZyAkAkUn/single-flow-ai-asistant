"""
Unit tests for Topic Tracker functionality.
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from src.core.topic_storage import TopicStorage
from src.core.topic_matcher import TopicMatcher
from src.core.topic_tracker import TopicTracker
from src.data.schemas import TopicInfo


class TestTopicStorage:
    """Test TopicStorage functionality."""

    def test_topic_storage_init(self):
        """Test topic storage initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = TopicStorage(storage_path=temp_dir)
            assert storage.storage_path.exists()
            assert isinstance(storage.topic_db, dict)
            assert isinstance(storage.topic_index, dict)

    def test_save_and_load_topic(self):
        """Test saving and loading topics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = TopicStorage(storage_path=temp_dir)

            # Create test topic
            topic = TopicInfo(
                topic_id="test_topic_123",
                topic_name="Test Topic",
                domain="technology",
                message_count=5,
                summary="Test summary",
                start_message_index=0
            )

            # Save topic
            result = storage.save_topic(topic)
            assert result is True

            # Load topic
            loaded_topic = storage.load_topic("test_topic_123")
            assert loaded_topic is not None
            assert loaded_topic.topic_name == "Test Topic"
            assert loaded_topic.domain == "technology"

    def test_find_topics_by_name(self):
        """Test finding topics by name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = TopicStorage(storage_path=temp_dir)

            # Create test topics
            topics = [
                TopicInfo(topic_id="topic1", topic_name="Python Programming", domain="programming"),
                TopicInfo(topic_id="topic2", topic_name="Java Programming", domain="programming"),
                TopicInfo(topic_id="topic3", topic_name="Cooking Recipes", domain="cooking")
            ]

            for topic in topics:
                storage.save_topic(topic)

            # Search for programming topics
            results = storage.find_topics_by_name("Programming")
            assert len(results) == 2
            assert all("Programming" in topic.topic_name for topic in results)


class TestTopicMatcher:
    """Test TopicMatcher functionality."""

    @patch('src.core.vector_manager.VectorManager')
    def test_topic_similarity(self, mock_vector_manager):
        """Test topic similarity calculation."""
        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_query.side_effect = [
            [1.0, 0.0, 0.0],  # "Python coding"
            [0.9, 0.1, 0.0],  # "Python programming"
            [0.0, 1.0, 0.0],  # "Cooking recipes"
        ]
        mock_vector_manager.embeddings = mock_embeddings

        matcher = TopicMatcher(mock_vector_manager)

        # Test similar topics
        similarity1 = matcher.calculate_similarity("Python coding", "Python programming")
        assert similarity1 > 0.8  # Should be very similar

        # Test different topics
        similarity2 = matcher.calculate_similarity("Python coding", "Cooking recipes")
        assert similarity2 < 0.5  # Should be different

    @patch('src.core.vector_manager.VectorManager')
    def test_find_similar_topics(self, mock_vector_manager):
        """Test finding similar topics from a list."""
        mock_embeddings = Mock()
        mock_embeddings.embed_query.side_effect = [
            [1.0, 0.0],  # New topic
            [0.9, 0.1],  # Similar existing topic
            [0.0, 1.0],  # Different topic
        ]
        mock_vector_manager.embeddings = mock_embeddings

        matcher = TopicMatcher(mock_vector_manager)
        existing_topics = ["Python development", "Cooking recipes"]

        similar = matcher.find_similar_topics("Python programming", existing_topics, threshold=0.8)
        assert len(similar) >= 1
        assert similar[0][0] == "Python development"


class TestTopicTracker:
    """Test TopicTracker functionality."""

    @patch('src.core.vector_manager.VectorManager')
    def test_topic_analysis(self, mock_vector_manager):
        """Test topic analysis for user messages."""
        # Setup mocks
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.5, 0.5]
        mock_vector_manager.embeddings = mock_embeddings

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = TopicStorage(storage_path=temp_dir)
            matcher = TopicMatcher(mock_vector_manager)
            tracker = TopicTracker(storage, matcher)

            # Test topic analysis
            topic_info = tracker.analyze_message_topic(
                user_message="How does Python programming work?",
                message_index=0,
                conversation_context=[]
            )

            assert topic_info is not None
            assert topic_info.topic_name is not None
            assert topic_info.domain is not None
            assert topic_info.message_count == 1

    @patch('src.core.vector_manager.VectorManager')
    def test_topic_continuation(self, mock_vector_manager):
        """Test continuing existing topics."""
        # Setup mocks for high similarity
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [1.0, 0.0]  # Perfect match
        mock_vector_manager.embeddings = mock_embeddings

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = TopicStorage(storage_path=temp_dir)
            matcher = TopicMatcher(mock_vector_manager)
            tracker = TopicTracker(storage, matcher)

            # Create initial topic
            initial_topic = tracker.analyze_message_topic(
                "Python programming basics",
                message_index=0
            )

            # Continue the topic
            continued_topic = tracker.analyze_message_topic(
                "How to use Python functions?",
                message_index=1,
                conversation_context=[{"content": "Python programming basics", "role": "user"}]
            )

            # Should be the same topic with increased count
            assert continued_topic.topic_id == initial_topic.topic_id
            assert continued_topic.message_count == 2


class TestTopicInfo:
    """Test TopicInfo dataclass."""

    def test_topic_info_creation(self):
        """Test TopicInfo object creation."""
        topic = TopicInfo(
            topic_id="test_123",
            topic_name="Test Topic",
            domain="technology"
        )

        assert topic.topic_id == "test_123"
        assert topic.topic_name == "Test Topic"
        assert topic.domain == "technology"
        assert topic.message_count == 0  # Default value

    def test_topic_info_serialization(self):
        """Test TopicInfo JSON serialization."""
        topic = TopicInfo(
            topic_id="test_123",
            topic_name="Test Topic",
            domain="technology",
            message_count=5
        )

        # Test to_dict
        data = topic.to_dict()
        assert data["topic_id"] == "test_123"
        assert data["topic_name"] == "Test Topic"
        assert data["domain"] == "technology"
        assert data["message_count"] == 5

        # Test from_dict
        recreated_topic = TopicInfo.from_dict(data)
        assert recreated_topic.topic_id == topic.topic_id
        assert recreated_topic.topic_name == topic.topic_name
        assert recreated_topic.domain == topic.domain


if __name__ == "__main__":
    pytest.main([__file__])
