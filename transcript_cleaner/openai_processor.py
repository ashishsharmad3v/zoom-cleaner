import openai
import json
import logging
from typing import Dict, List
from config.settings import OPENAI_API_KEY, DEFAULT_MODEL, ADVANCED_MODEL

logger = logging.getLogger(__name__)

class OpenAIProcessor:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        self.default_model = DEFAULT_MODEL
        self.advanced_model = ADVANCED_MODEL
    
    def process_with_openai(self, prompt: str, model: str = None, max_tokens: int = 2000) -> Dict:
        """Process text with OpenAI API"""
        if model is None:
            model = self.default_model
            
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens
            )
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "usage": response.usage
            }
        except Exception as e:
            logger.error(f"OpenAI processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": ""
            }
    
    def identify_speakers(self, transcript_chunk: str) -> Dict:
        """Identify and standardize speakers in transcript"""
        prompt = f"""
        Analyze this Zoom transcript chunk and identify all speakers. 
        For each speaker, provide:
        1. Their original name/timestamp format
        2. A standardized speaker ID (Speaker 1, Speaker 2, etc.)
        3. The actual speaker name if identifiable
        
        Transcript chunk:
        {transcript_chunk}
        
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
        """
        
        result = self.process_with_openai(prompt, max_tokens=2000)
        if result["success"]:
            try:
                parsed_content = json.loads(result["content"])
                return {
                    "success": True,
                    "speakers": parsed_content.get("speakers", []),
                    "utterances": parsed_content.get("speaker_utterances", [])
                }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Failed to parse speaker identification response"
                }
        return result
    
    def correct_grammar(self, transcript_chunk: str, context: str = "") -> Dict:
        """Correct grammar and clean transcript"""
        prompt = f"""
        You are a professional transcript cleaner. Process this Zoom transcript chunk with these requirements:
        
        1. CORRECT GR(grammar errors, punctuation, and sentence structure
        2. MAINTAIN SPEAKER ATTRIBUTION: Keep all speaker information intact
        3. PRESERVE CONTEXT: Maintain conversation flow and meaning
        4. CLEAN FORMAT: Remove filler words, but keep important content
        5. STANDARDIZE SPEAKERS: Use consistent speaker naming
        
        Previous context for continuity:
        {context[:1000]}  # Limit context size
        
        Current transcript chunk to process:
        {transcript_chunk}
        
        Return ONLY a JSON object in this exact format:
        {{
            "processed_text": "cleaned transcript with proper formatting",
            "speakers_identified": ["Speaker 1", "Speaker 2", ...],
            "key_context_points": ["important topics or decisions made"],
            "processing_notes": "any notes about challenges or decisions made"
        }}
        """
        
        result = self.process_with_openai(prompt, max_tokens=3000)
        if result["success"]:
            try:
                parsed_content = json.loads(result["content"])
                return {
                    "success": True,
                    "processed_text": parsed_content.get("processed_text", ""),
                    "speakers": parsed_content.get("speakers_identified", []),
                    "context_points": parsed_content.get("key_context_points", [])
                }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Failed to parse grammar correction response"
                }
        return result
    
    def quality_assurance(self, original: str, processed: str) -> Dict:
        """Perform quality assurance on processed transcript"""
        prompt = f"""
        Review this transcript processing result and validate quality:
        
        Original segment:
        {original[:2000]}  # Limit size
        
        Processed segment:
        {processed[:2000]}  # Limit size
        
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
        """
        
        result = self.process_with_openai(prompt, model=self.default_model, max_tokens=500)
        if result["success"]:
            try:
                parsed_content = json.loads(result["content"])
                return {
                    "success": True,
                    "quality_score": parsed_content.get("quality_score", 0),
                    "issues": parsed_content.get("issues_found", []),
                    "content_loss": parsed_content.get("content_loss_detected", False)
                }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Failed to parse QA response"
                }
        return result