from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
import json
import logging
from typing import List, Dict
from config.settings import OPENAI_API_KEY

logger = logging.getLogger(__name__)

class LangChainTranscriptCleaner:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
        self.setup_chains()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
    
    def setup_chains(self):
        """Setup LangChain prompt templates and chains"""
        
        # Speaker Identification Chain
        speaker_identification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional transcript analyzer. Identify and standardize speakers."),
            ("human", """
            Analyze this Zoom transcript segment and identify all speakers.
            
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
        ])
        
        self.speaker_chain = LLMChain(
            llm=self.llm,
            prompt=speaker_identification_prompt,
            output_key="speaker_analysis"
        )
        
        # Grammar Correction and Cleaning Chain
        grammar_correction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional transcript cleaner. Correct grammar and improve readability while preserving all content and speaker attribution."),
            ("human", """
            Clean this transcript segment with these requirements:
            
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
        ])
        
        self.grammar_chain = LLMChain(
            llm=self.llm,
            prompt=grammar_correction_prompt,
            output_key="grammar_correction"
        )
        
        # Quality Assurance Chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a quality assurance expert. Review transcript processing quality."),
            ("human", """
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
        ])
        
        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=qa_prompt,
            output_key="quality_assessment"
        )
    
    def load_and_chunk_transcript(self, transcript_text: str) -> List[Document]:
        """Load transcript and chunk using LangChain"""
        doc = Document(
            page_content=transcript_text,
            metadata={
                "source": "zoom_transcript",
                "total_length": len(transcript_text)
            }
        )
        
        docs = self.text_splitter.split_documents([doc])
        
        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = i
            doc.metadata["chunk_total"] = len(docs)
        
        return docs
    
    def process_segment(self, transcript_segment: str, previous_context: str = "") -> Dict:
        """Process a single transcript segment using LangChain chains"""
        try:
            with get_openai_callback() as cb:
                speaker_result = self.speaker_chain.run({
                    "transcript_segment": transcript_segment
                })
                
                grammar_result = self.grammar_chain.run({
                    "transcript_segment": transcript_segment,
                    "previous_context": previous_context[:1000]
                })
                
                logger.info(f"Tokens used: {cb.total_tokens}, Cost: ${cb.total_cost}")
                
                return {
                    "success": True,
                    "speaker_analysis": json.loads(speaker_result),
                    "grammar_correction": json.loads(grammar_result),
                    "cost": cb.total_cost,
                    "tokens": cb.total_tokens
                }
        except Exception as e:
            logger.error(f"Chain processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def quality_check(self, original: str, processed: str) -> Dict:
        """Perform quality check using LangChain"""
        try:
            with get_openai_callback() as cb:
                result = self.qa_chain.run({
                    "original_segment": original[:2000],
                    "processed_segment": processed[:2000]
                })
                
                parsed_result = json.loads(result)
                return {
                    "success": True,
                    "quality_score": parsed_result.get("quality_score", 0),
                    "issues": parsed_result.get("issues_found", []),
                    "content_loss": parsed_result.get("content_loss_detected", False),
                    "cost": cb.total_cost
                }
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }