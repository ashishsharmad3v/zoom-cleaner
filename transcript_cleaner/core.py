import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import time
from .openai_processor import OpenAIProcessor
from .langchain_integration import LangChainTranscriptCleaner
from .utils import merge_segments_with_overlap, find_overlap
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS

logger = logging.getLogger(__name__)

class ZoomTranscriptCleaner:
    def __init__(self, use_langchain: bool = True):
        self.use_langchain = use_langchain
        self.openai_processor = OpenAIProcessor()
        self.langchain_processor = LangChainTranscriptCleaner() if use_langchain else None
        self.context_memory = {}
    
    def clean_transcript(self, transcript_text: str) -> str:
        """Main method to clean entire transcript"""
        logger.info("Starting transcript cleaning process...")
        
        # Step 1: Chunk the transcript
        logger.info("Chunking transcript...")
        chunks = self._chunk_transcript(transcript_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Process chunks
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
    
    def _chunk_transcript(self, transcript: str) -> List[Dict]:
        """Split transcript into chunks"""
        lines = transcript.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_length = len(line)
            
            if current_length + line_length <= CHUNK_SIZE:
                current_chunk.append(line)
                current_length += line_length
                i += 1
            else:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'index': chunk_index,
                    'text': chunk_text,
                    'start_line': max(0, i - len(current_chunk)),
                    'end_line': i
                })
                
                # Calculate overlap for next chunk
                overlap_lines = []
                overlap_length = 0
                j = len(current_chunk) - 1
                while j >= 0 and overlap_length < CHUNK_OVERLAP:
                    overlap_lines.insert(0, current_chunk[j])
                    overlap_length += len(current_chunk[j])
                    j -= 1
                
                current_chunk = overlap_lines
                current_length = overlap_length
                chunk_index += 1
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'index': chunk_index,
                'text': chunk_text,
                'start_line': max(0, len(lines) - len(current_chunk)),
                'end_line': len(lines)
            })
        
        return chunks
    
    def _process_chunks_parallel(self, chunks: List[Dict]) -> List[Dict]:
        """Process chunks in parallel"""
        processed_chunks = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for chunk in chunks:
                context = self._get_context_for_chunk(chunk['index'])
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
        if self.use_langchain:
            result = self.langchain_processor.process_segment(chunk['text'], context)
            if result["success"]:
                processed_text = result["grammar_correction"].get("processed_text", chunk['text'])
                speakers = result["grammar_correction"].get("speakers_identified", [])
                context_points = result["grammar_correction"].get("key_context_points", [])
            else:
                # Fallback to basic processing
                processed_text = chunk['text']
                speakers = []
                context_points = []
        else:
            result = self.openai_processor.correct_grammar(chunk['text'], context)
            if result["success"]:
                processed_text = result["processed_text"]
                speakers = result["speakers"]
                context_points = result["context_points"]
            else:
                processed_text = chunk['text']
                speakers = []
                context_points = []
        
        # Update context memory
        self._update_context(chunk['index'], context_points)
        
        return {
            'index': chunk['index'],
            'processed_text': processed_text,
            'speakers': speakers,
            'context_points': context_points
        }
    
    def _update_context(self, chunk_index: int, context_points: List[str]):
        """Update context memory"""
        self.context_memory[chunk_index] = context_points
        if len(self.context_memory) > 10:
            oldest_key = min(self.context_memory.keys())
            del self.context_memory[oldest_key]
    
    def _get_context_for_chunk(self, chunk_index: int) -> str:
        """Get context for a specific chunk"""
        context_points = []
        for i in range(max(0, chunk_index - 3), chunk_index):
            if i in self.context_memory:
                context_points.extend(self.context_memory[i])
        return "\n".join(context_points[-5:])
    
    def _assemble_transcript(self, processed_chunks: List[Dict]) -> str:
        """Assemble processed chunks into final transcript"""
        segments = [chunk['processed_text'] for chunk in processed_chunks]
        return merge_segments_with_overlap(segments)
    
    def _perform_final_qa(self, original: str, processed: str) -> Dict:
        """Perform final quality assurance"""
        if self.use_langchain:
            return self.langchain_processor.quality_check(original, processed)
        else:
            return self.openai_processor.quality_assurance(original, processed)