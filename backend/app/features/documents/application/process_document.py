"""Use case for extracting content from a PDF and processing it into a structured
Chonkie MarkdownDocument."""

import tempfile
import os
from typing import Tuple, Dict, List
from pathlib import Path
from datetime import datetime

# Import the return type for type hinting
from chonkie import MarkdownDocument
from chonkie import MarkdownChef
import pymupdf4llm
from app.features.documents.domain.value_objects import FileUpload
from app.shared.exceptions import DocumentProcessingError
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class ProcessDocument:
    """
    Extracts markdown from a PDF and processes it using Chonkie's MarkdownChef,
    returning a structured MarkdownDocument object with page number metadata.
    """
    
    def __init__(self, tokenizer: str = "gpt2"):
        self.tokenizer = tokenizer
        self.page_map: Dict[str, int] = {}  # Maps content to page numbers
        self._ensure_output_dirs()
    
    def _ensure_output_dirs(self):
        """Ensure output directories exist."""
        Path(settings.markdown_output_dir).mkdir(parents=True, exist_ok=True)
    
    async def execute(self, file_upload: FileUpload) -> Tuple[MarkdownDocument, int, List[Tuple[int, str]]]:
        """
        Processes an uploaded PDF into a Chonkie MarkdownDocument with page tracking.
        
        Args:
            file_upload: The uploaded file value object.
            
        Returns:
            A tuple of (MarkdownDocument, total_pages, page_chunks)
            page_chunks: List of (page_number, content) for direct page-level chunking
        """
        logger.info(f"Starting full processing for {file_upload.filename}")
        
        # Extract markdown with page information
        markdown_content, total_pages, page_chunks = await self._extract_with_pages(file_upload)
        
        # Save pymupdf4llm output
        self._save_pymupdf_output(file_upload.filename, markdown_content)
        
        # Process with MarkdownChef
        chonkie_doc = await self._process_with_chef(markdown_content, file_upload.filename)
        
        # Store page mapping for later use
        self.page_map = self._build_page_map(page_chunks)
        
        logger.info(
            f"MarkdownChef processed: {len(chonkie_doc.chunks)} text chunks, "
            f"{len(chonkie_doc.code)} code blocks, {len(chonkie_doc.tables)} tables from {total_pages} pages."
        )
        
        return chonkie_doc, total_pages, page_chunks
    
    async def _extract_with_pages(self, file_upload: FileUpload) -> Tuple[str, int, List[Tuple[int, str]]]:
        """
        Extract markdown from PDF with page number tracking.
        
        Returns:
            Tuple of (full_markdown, total_pages, list of (page_num, content))
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_upload.content)
                tmp.flush()
                tmp_pdf_path = tmp.name
            
            # Extract with page chunks enabled - this returns a LIST of dicts
            page_data = pymupdf4llm.to_markdown(
                tmp_pdf_path,
                page_chunks=True,      # Returns list instead of string!
                page_separators=False  # We don't need separators if we get a list
            )
            
            os.unlink(tmp_pdf_path)
            
            # page_data is a list of dicts with structure: [{'metadata': {...}, 'text': '...'}, ...]
            page_chunks = []
            full_markdown_parts = []
            
            if isinstance(page_data, list):
                # It's already a list of pages
                for i, page_dict in enumerate(page_data):
                    # Extract text from the page dict
                    if isinstance(page_dict, dict):
                        # Get the actual page number from metadata
                        page_num = page_dict.get('metadata', {}).get('page', i + 1)
                        content = page_dict.get('text', '') or page_dict.get('content', '')
                    else:
                        page_num = i + 1
                        content = str(page_dict)
                    
                    if content.strip():
                        page_chunks.append((page_num, content))
                        full_markdown_parts.append(content)
                
                total_pages = len(page_chunks)
                full_markdown = "\n\n".join(full_markdown_parts)
            else:
                # Fallback: it's a string (shouldn't happen with page_chunks=True)
                full_markdown = str(page_data)
                page_chunks = [(1, full_markdown)]
                total_pages = 1
            
            logger.info(f"Extracted {total_pages} pages, total length: {len(full_markdown)} characters")
            return full_markdown, total_pages, page_chunks
            
        except Exception as e:
            logger.error(f"Failed to extract markdown from PDF: {e}")
            raise DocumentProcessingError(
                document_id="unknown",
                reason=f"PDF to markdown extraction failed: {str(e)}"
            )
    
    def _build_page_map(self, page_chunks: List[Tuple[int, str]]) -> Dict[str, int]:
        """
        Build a mapping of content snippets to page numbers.
        This helps associate chunks with their source pages.
        """
        page_map = {}
        for page_num, content in page_chunks:
            # Store first 100 chars of each page as key
            key = content[:100].strip()
            page_map[key] = page_num
        
        logger.info(f"Built page map with {len(page_map)} pages")
        if page_chunks:
            logger.debug(f"First page starts with: {page_chunks[0][1][:50]}...")
            logger.debug(f"Last page (#{page_chunks[-1][0]}) starts with: {page_chunks[-1][1][:50]}...")
        
        return page_map
    
    def get_page_number_for_content(self, content: str) -> int:
        """
        Try to determine the page number for a given content chunk.
        
        Returns:
            Page number (1-based) or 0 if unknown
        """
        if not content or not self.page_map:
            return 0
            
        content_start = content[:100].strip()
        
        # Try exact match first
        if content_start in self.page_map:
            logger.debug(f"Exact match found for chunk starting with: {content_start[:50]}...")
            return self.page_map[content_start]
        
        # Try fuzzy match (find best overlap)
        best_match_page = 0
        best_match_length = 0
        
        for key, page_num in self.page_map.items():
            if key in content or content_start in key:
                match_length = len(set(key.split()) & set(content_start.split()))
                if match_length > best_match_length:
                    best_match_length = match_length
                    best_match_page = page_num
        
        if best_match_page > 0:
            logger.debug(f"Fuzzy match found (score={best_match_length}) for chunk -> page {best_match_page}")
        else:
            logger.debug(f"No page match found for chunk starting with: {content_start[:50]}...")
        
        return best_match_page
    
    def _save_pymupdf_output(self, filename: str, markdown_content: str):
        """Save the pymupdf4llm markdown output to file."""
        try:
            # Create filename with timestamp
            base_name = Path(filename).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_name}_{timestamp}_pymupdf4llm.md"
            output_path = Path(settings.markdown_output_dir) / output_filename
            
            # Save the markdown content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# pymupdf4llm Output\n")
                f.write(f"Source: {filename}\n")
                f.write(f"Generated: {timestamp}\n\n")
                f.write("---\n\n")
                f.write(markdown_content)
            
            logger.info(f"Saved pymupdf4llm output to: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save pymupdf4llm output: {e}")
    
    async def _process_with_chef(self, markdown_content: str, filename: str = "unknown") -> MarkdownDocument:
        """Process markdown content with MarkdownChef."""
        chef = MarkdownChef(tokenizer=self.tokenizer)
        
        tmp_path = ""
        try:
            # Create a temporary file to hold the markdown content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
                tmp.write(markdown_content)
                tmp_path = tmp.name
            
            logger.info(f"Processing temporary markdown file: {tmp_path}")
            
            # Process the file with the chef
            chonkie_doc = chef.process(tmp_path)
            
            # Save Chonkie's processed output
            self._save_chonkie_output(filename, chonkie_doc)
            
            return chonkie_doc
            
        except Exception as e:
            logger.error(f"Failed to process document with MarkdownChef: {e}")
            raise DocumentProcessingError(
                document_id="unknown",
                reason=f"MarkdownChef processing failed: {str(e)}"
            )
        finally:
            # Clean up the temporary file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _save_chonkie_output(self, filename: str, chonkie_doc: MarkdownDocument):
        """Save the Chonkie MarkdownChef output to file."""
        try:
            # Create filename with timestamp
            base_name = Path(filename).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_name}_{timestamp}_chonkie.md"
            output_path = Path(settings.markdown_output_dir) / output_filename
            
            # Save the processed chunks
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Chonkie MarkdownChef Output\n")
                f.write(f"Source: {filename}\n")
                f.write(f"Generated: {timestamp}\n")
                f.write(f"Total Chunks: {len(chonkie_doc.chunks)}\n")
                f.write(f"Code Blocks: {len(chonkie_doc.code)}\n")
                f.write(f"Tables: {len(chonkie_doc.tables)}\n\n")
                f.write("---\n\n")
                
                # Write text chunks
                f.write("## Text Chunks\n\n")
                for i, chunk in enumerate(chonkie_doc.chunks, 1):
                    f.write(f"### Chunk {i}\n")
                    f.write(f"**Tokens:** {chunk.token_count}\n\n")
                    f.write(f"{chunk.text}\n\n")
                    f.write("---\n\n")
                
                # Write code blocks if any
                if chonkie_doc.code:
                    f.write("## Code Blocks\n\n")
                    for i, code_block in enumerate(chonkie_doc.code, 1):
                        f.write(f"### Code Block {i}\n")
                        f.write(f"```\n{code_block}\n```\n\n")
                
                # Write tables if any
                if chonkie_doc.tables:
                    f.write("## Tables\n\n")
                    for i, table in enumerate(chonkie_doc.tables, 1):
                        f.write(f"### Table {i}\n")
                        f.write(f"{table}\n\n")
            
            logger.info(f"Saved Chonkie output to: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save Chonkie output: {e}")