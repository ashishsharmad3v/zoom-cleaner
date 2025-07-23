#!/usr/bin/env python3

import argparse
import logging
import sys
import os
from transcript_cleaner import ZoomTranscriptCleaner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Clean Zoom transcripts using AI')
    parser.add_argument('--input', '-i', required=True, help='Input transcript file')
    parser.add_argument('--output', '-o', required=True, help='Output cleaned transcript file')
    parser.add_argument('--use-langchain', action='store_true', help='Use LangChain integration')
    parser.add_argument('--api-key', help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} not found")
        sys.exit(1)
    
    # Check if API key is set
    if not os.environ.get('OPENAI_API_KEY'):
        logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    try:
        # Read input file
        logger.info(f"Reading transcript from {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        # Initialize cleaner
        logger.info("Initializing transcript cleaner")
        cleaner = ZoomTranscriptCleaner(use_langchain=args.use_langchain)
        
        # Clean transcript
        logger.info("Cleaning transcript...")
        cleaned_transcript = cleaner.clean_transcript(transcript_text)
        
        # Write output file
        logger.info(f"Writing cleaned transcript to {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(cleaned_transcript)
        
        logger.info("Transcript cleaning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing transcript: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()