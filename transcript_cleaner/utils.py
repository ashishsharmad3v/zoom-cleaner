import json
import logging
import re
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

def create_langchain_documents(transcript: str) -> List[Document]:
    """Create LangChain documents from transcript text"""
    # Use LangChain's intelligent chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    
    doc = Document(
        page_content=transcript,
        metadata={
            "source": "zoom_transcript",
            "total_length": len(transcript)
        }
    )
    
    docs = text_splitter.split_documents([doc])
    
    # Add chunk metadata
    for i, doc in enumerate(docs):
        doc.metadata["chunk_index"] = i
        doc.metadata["chunk_total"] = len(docs)
    
    return docs

def find_overlap(text1: str, text2: str) -> str:
    """Find overlapping text between two strings"""
    min_len = min(len(text1), len(text2))
    for i in range(min(500, min_len), 0, -1):
        if text1.endswith(text2[:i]):
            return text2[:i]
    return ""

def merge_segments_with_overlap(segments: List[str]) -> str:
    """Merge segments while handling overlaps"""
    if not segments:
        return ""
    
    merged_text = segments[0]
    
    for i in range(1, len(segments)):
        current_segment = segments[i]
        overlap = find_overlap(merged_text, current_segment)
        if overlap:
            current_segment = current_segment[len(overlap):].strip()
        merged_text += "\n\n" + current_segment
    
    return merged_text

def safe_json_parse(text: str) -> Dict:
    """Safely parse JSON with error handling"""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        return {}

def extract_speakers_from_text(text: str) -> List[str]:
    """Extract speaker names from transcript text"""
    speaker_pattern = r'^([A-Za-z\s\.]+)(?:\s+\d{1,2}:\d{2}:\d{2})?:'
    speakers = set()
    
    lines = text.split('\n')
    for line in lines:
        match = re.match(speaker_pattern, line)
        if match:
            speaker = match.group(1).strip()
            speakers.add(speaker)
    
    return list(speakers)