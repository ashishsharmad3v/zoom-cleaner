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
    parser = argparse.ArgumentParser(description='Clean Zoom transcripts using custom LLM')
    parser.add_argument('--input', '-i', required=True, help='Input transcript file')
    parser.add_argument('--output', '-o', required=True, help='Output cleaned transcript file')
    parser.add_argument('--model', '-m', default='gpt-3.5-turbo', help='OpenAI model to use')
    parser.add_argument('--temperature', '-t', type=float, default=0.1, help='Model temperature')
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
        logger.info(f"Initializing transcript cleaner with model {args.model}")
        cleaner = ZoomTranscriptCleaner(
            model_name=args.model