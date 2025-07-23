import json
import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

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