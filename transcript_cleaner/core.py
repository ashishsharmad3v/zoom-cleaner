import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import time
from .custom_llm import CustomOpenAILLM
from .processor import TranscriptProcessor
from .utils import merge_segments_with_overlap
from config.settings import MAX_WORKERS

logger = logging.getLogger(__name__)

class ZoomTranscriptCleaner:
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        self.llm = CustomOpenAILLM(
            model_name=model_name,
            temperature=temperature
        )
        self.processor = TranscriptProcessor(self.llm)
        logger.info(f"Initialized ZoomTranscriptCleaner with {model_name}")
    
    def clean_transcript(self, transcript_text: str) -> str:
        """Main method to clean entire transcript"""
        logger.info("Starting transcript cleaning process...")
        
        # Step 1: Chunk the transcript using LangChain
        logger.info("Chunking transcript with LangChain...")
        chunks = self.processor.chunk_transcript(transcript_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Process chunks in parallel
        logger.info("Processing chunks...")
        processed_chunks = self._process_chunks_parallel(chunks)
        
        # Step 3: Sort chunks by index
        processed_chunks.sort(key=lambda x: x['index'])
        
        # Step 4: Assemble final transcript
        logger.info("Assembling final transcript...")
        final_transcript = self._assemble_transcript(processed_chunks)
        
        # Step 5: Final quality check
        logger.info("Performing final quality check...")
        final_qa = self._perform_final_qa(transcript_text, final_transcript)
        
        logger.info("Transcript cleaning completed successfully!")
        return final_transcript
    
    def _process_chunks_parallel(self, chunks: List[Dict]) -> List[Dict]:
        """Process chunks in parallel"""
        processed_chunks = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for chunk in chunks:
                context = self.processor.get_context_for_segment(chunk['index'])
                future = executor.submit(self._process_single_chunk, chunk, context)
                futures.append((future, chunk))
            
            for future, original_chunk in futures:
                try:
                    result = future.result()
                    processed_chunks.append(result)
                except Exception as e:
                    logger.error(f"Failed to process chunk: {e}")
                    processed_chunks.append({
                        'index': original_chunk['index'],
                        'processed_text': original_chunk['text'],
                        'speakers': [],
                        'context_points': []
                    })
        
        return processed_chunks
    
    def _process_single_chunk(self, chunk: Dict, context: str) -> Dict:
        """Process a single chunk"""
        result = self.processor.process_segment(chunk['text'], context)
        
        if result["success"]:
            processed_text = result["grammar_correction"].get("processed_text", chunk['text'])
            speakers = result["grammar_correction"].get("speakers_identified", [])
            context_points = result["grammar_correction"].get("key_context_points", [])
        else:
            processed_text = chunk['text']
            speakers = []
            context_points = []
        
        # Update context memory
        self.processor.update_context(chunk['index'], context_points)
        
        return {
            'index': chunk['index'],
            'processed_text': processed_text,
            'speakers': speakers,
            'context_points': context_points
        }
    
    def _assemble_transcript(self, processed_chunks: List[Dict]) -> str:
        """Assemble processed chunks into final transcript"""
        segments = [chunk['processed_text'] for chunk in processed_chunks]
        return merge_segments_with_overlap(segments)
    
    def _perform_final_qa(self, original: str, processed: str) -> Dict:
        """Perform final quality assurance"""
        return self.processor.quality_check(original, processed)