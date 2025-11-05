"""Use case for extracting content from a PDF and processing it into a structured
Chonkie MarkdownDocument."""

import tempfile
import os
from typing import Tuple, List

# Import the return type for type hinting
from chonkie import MarkdownDocument
from chonkie import MarkdownChef
import pymupdf4llm
from app.features.documents.domain.value_objects import FileUpload
from app.shared.exceptions import DocumentProcessingError
from app.core.logger import get_logger

logger = get_logger(__name__)


class ProcessDocument:
    """
    Extracts markdown from a PDF and processes it using Chonkie's MarkdownChef.
    Each page is processed separately to maintain accurate page number tracking.
    """
    
    def __init__(self, tokenizer: str = "gpt2"):
        self.tokenizer = tokenizer
    
    async def execute(self, file_upload: FileUpload) -> Tuple[List[Tuple[MarkdownDocument, int]], int, List[Tuple[int, str]]]:
        """
        Processes an uploaded PDF into Chonkie MarkdownDocuments with page tracking.
        Each page is processed separately to maintain accurate page numbers.
        
        Args:
            file_upload: The uploaded file value object.
            
        Returns:
            A tuple of (page_documents, total_pages, page_chunks)
            page_documents: List of (MarkdownDocument, page_number) tuples
            total_pages: Total number of pages
            page_chunks: List of (page_number, content) for reference
        """
        logger.info(f"Starting page-by-page processing for {file_upload.filename}")
        
        # Extract markdown with page information
        markdown_content, total_pages, page_chunks = await self._extract_with_pages(file_upload)
        
        # Process each page separately with MarkdownChef
        page_documents = []
        for page_num, page_content in page_chunks:
            if page_content.strip():
                chonkie_page_doc = await self._process_with_chef(page_content)
                page_documents.append((chonkie_page_doc, page_num))
                logger.debug(f"Processed page {page_num}: {len(chonkie_page_doc.chunks)} chunks")
        
        total_chunks = sum(len(doc.chunks) for doc, _ in page_documents)
        total_code = sum(len(doc.code) for doc, _ in page_documents)
        total_tables = sum(len(doc.tables) for doc, _ in page_documents)
        
        logger.info(
            f"MarkdownChef processed {len(page_documents)} pages: "
            f"{total_chunks} text chunks, {total_code} code blocks, {total_tables} tables"
        )
        
        return page_documents, total_pages, page_chunks
    
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
    
    async def _process_with_chef(self, markdown_content: str) -> MarkdownDocument:
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