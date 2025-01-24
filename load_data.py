import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import logging
import re

class DocumentChunkLoader:
    def __init__(self, file_path: str = "data/crimeandpunishment.txt"):
        self.file_path = Path(file_path)
        self.logger = logging.getLogger(__name__)
        
    def load_document(self) -> str:
        """Load the full document text"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Find the actual start of the narrative
        # Skip past the Project Gutenberg header
        start_marker = "CRIME AND PUNISHMENT"
        narrative_start = text.find(start_marker)
        # Skip past the initial chapter markers to get to actual content
        chapter_start = text.find("CHAPTER", narrative_start)
        actual_content_start = text.find("\n\n", chapter_start)
        
        # Get the main text, skipping headers
        main_text = text[actual_content_start:].strip()
        
        self.logger.info(f"Loaded {len(main_text)} characters of narrative content")
        return main_text
    
    def get_chunk(self, text: str, chunk_size: int = 1000, offset: int = 1000) -> str:
        """Get a chunk of text from the middle of the document"""
        # Start from a point well into the narrative
        start_pos = offset
        if start_pos + chunk_size > len(text):
            start_pos = len(text) - chunk_size
        
        # Get a clean chunk (try to start at a sentence boundary)
        chunk = text[start_pos:start_pos + chunk_size]
        # Adjust to start at a sentence boundary
        first_period = chunk.find(". ") + 2
        if first_period > 1:  # if we found a sentence boundary
            chunk = chunk[first_period:]
            
        return chunk

    def prepare_context_window(self, text: str, tokenizer, context_length: int, 
                             offset: int = 1000) -> torch.Tensor:
        """Prepare a context window from the narrative"""
        # Get appropriate chunk
        chunk = self.get_chunk(text, chunk_size=context_length * 4, offset=offset)  # Get larger chunk for tokenization
        
        # Tokenize
        tokens = tokenizer(
            chunk,
            truncation=True,
            max_length=context_length,
            return_tensors="pt"
        )
        
        actual_length = tokens.input_ids.shape[1]
        self.logger.info(f"Created context window of {actual_length} tokens from narrative text")
        
        return tokens

def test_document_loading():
    """Test the improved document loading"""
    loader = DocumentChunkLoader("data/crimeandpunishment.txt")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    
    # Load document
    text = loader.load_document()
    print(f"\nDocument stats:")
    print(f"Total characters: {len(text)}")
    
    # Test different context lengths
    for length, offset in [(512, 5000), (1024, 10000)]:
        print(f"\nContext length {length} (offset {offset}):")
        chunk = loader.get_chunk(text, chunk_size=length*4, offset=offset)
        print(f"Sample chunk:\n{chunk[:200]}...")
        
        tokens = loader.prepare_context_window(text, tokenizer, length, offset)
        print(f"Actual tokens: {tokens.input_ids.shape[1]}")
        decoded = tokenizer.decode(tokens.input_ids[0][:50])
        print(f"Sample decoded text:\n{decoded}...")

if __name__ == "__main__":
    test_document_loading()