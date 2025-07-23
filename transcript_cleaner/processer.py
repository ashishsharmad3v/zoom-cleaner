from typing import Dict, List, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import logging
from .custom_llm import CustomOpenAILLM
from .utils import create_langchain_documents
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class TranscriptProcessor:
    def __init__(self, llm: CustomOpenAILLM):
        self.llm = llm
        self.context_memory = {}
        self.setup_chains()
    
    def setup_chains(self):
        """Setup processing chains with custom LLM"""
        
        # Speaker Identification Prompt
        speaker_prompt = PromptTemplate.from_template("""
        Analyze this Zoom transcript segment and identify all speakers. 
        For each speaker, provide:
        1. Their original name/timestamp format
        2. A standardized speaker ID (Speaker 1, Speaker 2, etc.)
        3. The actual speaker name if identifiable
        
        Transcript segment:
        {transcript_segment}
        
        Return ONLY a JSON object in this exact format:
        {{
            "speakers": [
                {{
                    "original_format": "string",
                    "speaker_id": "Speaker 1",
                    "actual_name": "John Smith or Unknown"
                }}
            ],
            "speaker_utterances": [
                {{
                    "speaker_id": "Speaker 1",
                    "text": "their complete utterance"
                }}
            ]
        }}
        """)
        
        self.speaker_chain = LLMChain(llm=self.llm, prompt=speaker_prompt)
        
        # Grammar Correction Prompt
        grammar_prompt = PromptTemplate.from_template("""
        You are a professional transcript cleaner. Process this Zoom transcript segment with these requirements:
        
        1. CORRECT GRAMMAR: Fix all grammatical errors, punctuation, and sentence structure
        2. MAINTAIN SPEAKER ATTRIBUTION: Keep all speaker information intact
        3. PRESERVE CONTEXT: Maintain conversation flow and meaning
        4. CLEAN FORMAT: Remove filler words, but keep important content
        5. STANDARDIZE SPEAKERS: Use consistent speaker naming
        
        Previous context for continuity:
        {previous_context}
        
        Current transcript segment to process:
        {transcript_segment}
        
        Return ONLY a JSON object in this exact format:
        {{
            "processed_text": "cleaned transcript with proper formatting",
            "speakers_identified": ["Speaker 1", "Speaker 2", ...],
            "key_context_points": ["important topics or decisions made"],
            "processing_notes": "any notes about challenges or decisions made"
        }}
        """)
        
        self.grammar_chain = LLMChain(llm=self.llm, prompt=grammar_prompt)
        
        # Quality Assurance Prompt
        qa_prompt = PromptTemplate.from_template("""
        Review this transcript processing result and validate quality:
        
        Original segment:
        {original_segment}
        
        Processed segment:
        {processed_segment}
        
        Check for:
        1. Content completeness (no information loss)
        2. Speaker attribution accuracy
        3. Grammar improvement
        4. Context preservation
        5. Formatting quality
        
        Return ONLY a JSON object:
        {{
            "quality_score": 0-100,
            "issues_found": ["list of any issues"],
            "content_loss_detected": true/false,
            "recommendations": ["improvement suggestions"]
        }}
        """)
        
        self.qa_chain = LLMChain(llm=self.llm, prompt=qa_prompt)
    
    def chunk_transcript(self, transcript_text: str) -> List[Dict]:
        """Chunk transcript using LangChain"""
        documents = create_langchain_documents(transcript_text)
        
        # Convert to our expected format
        chunks = []
        for doc in documents:
            chunks.append({
                'index': doc.metadata.get('chunk_index', 0),
                'text': doc.page_content,
                'metadata': doc.metadata
            })
        
        return chunks
    
    def process_segment(self, transcript_segment: str, previous_context: str = "") -> Dict:
        """Process a single transcript segment"""
        try:
            # Process speaker identification
            speaker_result = self.speaker_chain.invoke({
                "transcript_segment": transcript_segment
            })
            
            # Process grammar correction
            grammar_result = self.grammar_chain.invoke({
                "transcript_segment": transcript_segment,
                "previous_context": previous_context[:1000]
            })
            
            # Parse results
            speaker_data = self._safe_json_parse(speaker_result["text"])
            grammar_data = self._safe_json_parse(grammar_result["text"])
            
            return {
                "success": True,
                "speaker_analysis": speaker_data,
                "grammar_correction": grammar_data
            }
        except Exception as e:
            logger.error(f"Segment processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _safe_json_parse(self, text: str) -> Dict:
        """Safely parse JSON response"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {text[:100]}...")
            return {}
    
    def update_context(self, segment_index: int, context_points: List[str]):
        """Update context memory"""
        self.context_memory[segment_index] = context_points
        if len(self.context_memory) > 10:
            oldest_key = min(self.context_memory.keys())
            del self.context_memory[oldest_key]
    
    def get_context_for_segment(self, segment_index: int) -> str:
        """Get context for a specific segment"""
        context_points = []
        for i in range(max(0, segment_index - 3), segment_index):
            if i in self.context_memory:
                context_points.extend(self.context_memory[i])
        return "\n".join(context_points[-5:])
    
    def quality_check(self, original: str, processed: str) -> Dict:
        """Perform quality check"""
        try:
            result = self.qa_chain.invoke({
                "original_segment": original[:2000],
                "processed_segment": processed[:2000]
            })
            
            parsed_result = self._safe_json_parse(result["text"])
            return {
                "success": True,
                "quality_score": parsed_result.get("quality_score", 0),
                "issues": parsed_result.get("issues_found", []),
                "content_loss": parsed_result.get("content_loss_detected", False)
            }
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }