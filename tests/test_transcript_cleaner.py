import pytest
import os
from transcript_cleaner import ZoomTranscriptCleaner

# Test data
SAMPLE_TRANSCRIPT = """
John Smith 10:30:15: Hello everyone, welcome to today's meeting.
Jane Doe 10:30:22: Thanks for having us, John.
John Smith 10:30:28: So, let's start with the agenda items.
Mike Johnson 10:30:35: Yeah, I wanted to discuss the quarterly results.
Jane Doe 10:30:42: That sounds good. I have the numbers right here.
"""

def test_chunking():
    """Test transcript chunking functionality"""
    cleaner = ZoomTranscriptCleaner(use_langchain=False)
    chunks = cleaner._chunk_transcript(SAMPLE_TRANSCRIPT)
    assert len(chunks) > 0
    assert all('index' in chunk for chunk in chunks)

def test_context_management():
    """Test context management"""
    cleaner = ZoomTranscriptCleaner(use_langchain=False)
    context_points = ["discussing quarterly results", "reviewing numbers"]
    cleaner._update_context(0, context_points)
    context = cleaner._get_context_for_chunk(1)
    assert len(context) > 0

def test_assembly():
    """Test transcript assembly"""
    from transcript_cleaner.utils import merge_segments_with_overlap
    segments = ["Hello world", "world today", "today is great"]
    result = merge_segments_with_overlap(segments)
    assert "Hello world" in result
    assert "today is great" in result

if __name__ == "__main__":
    pytest.main([__file__])