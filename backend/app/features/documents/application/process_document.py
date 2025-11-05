"""Use case for extracting content from a PDF and processing it into a structured
Chonkie MarkdownDocument."""

import asyncio
import tempfile
import os
from typing import Tuple, List

# Import the return type for type hinting
from chonkie import MarkdownDocument
from chonkie import MarkdownChef
import pymupdf4llm
import fitz  # PyMuPDF
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
        self.chef = MarkdownChef(tokenizer=self.tokenizer)
    
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
        logger.info(f"Processing PDF: {file_upload.filename}")
        
        # Extract markdown with page information
        total_pages, page_chunks = await self._extract_with_pages(file_upload)
        
        # Process all pages in batch with MarkdownChef (optimized)
        page_documents = await self._process_pages_batch(page_chunks)
        
        # Summary logging
        total_chunks = sum(len(doc.chunks) for doc, _ in page_documents)
        total_code = sum(len(doc.code) for doc, _ in page_documents)
        total_tables = sum(len(doc.tables) for doc, _ in page_documents)
        
        logger.info(
            f"Processed {len(page_documents)} pages → "
            f"{total_chunks} chunks, {total_code} code blocks, {total_tables} tables"
        )
        
        return page_documents, total_pages, page_chunks
    
    async def _extract_with_pages(self, file_upload: FileUpload) -> Tuple[int, List[Tuple[int, str]]]:
        """
        Extract markdown from PDF with page number tracking.
        
        Returns:
            Tuple of (total_pages, list of (page_num, content))
        """
        tmp_pdf_path = None
        pdf_doc = None
        
        try:
            # Create temporary PDF file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_upload.content)
                tmp.flush()
                tmp_pdf_path = tmp.name
            
            # Open PDF with PyMuPDF for fallback extraction
            pdf_doc = fitz.open(tmp_pdf_path)
            
            # Extract with page chunks enabled - this returns a LIST of dicts
            page_data = pymupdf4llm.to_markdown(
                tmp_pdf_path,
                page_chunks=True,      # Returns list instead of string!
                page_separators=False  # We don't need separators if we get a list
            )
            
            # page_data is a list of dicts with structure: [{'metadata': {...}, 'text': '...'}, ...]
            page_chunks = []
            
            
            logger.info(f"PDF extraction returned {len(page_data)} page chunks")
            
            for i, page_dict in enumerate(page_data):
                page_num = page_dict.get('metadata', {}).get('page', i + 1)
                
                # Get markdown-formatted content from pymupdf4llm
                content = page_dict.get('text', '')
                
                # Fallback: If pymupdf4llm couldn't extract text, use direct PyMuPDF
                # Note: Direct extraction returns plain text without markdown formatting
                if not content.strip():
                    try:
                        page = pdf_doc[i]
                        direct_text = page.get_text("text")
                        
                        if direct_text.strip():
                            content = direct_text
                            logger.info(
                                f"Page {page_num}: Recovered {len(direct_text)} chars "
                                f"(plain text, no markdown formatting)"
                            )
                        else:
                            logger.warning(f"Page {page_num}: No extractable text found")
                    except Exception as e:
                        logger.error(f"Page {page_num}: Fallback extraction failed: {e}")
                
                # Always include the page, even if empty (preserve page numbering)
                page_chunks.append((page_num, content))
            
            total_pages = len(page_chunks)
            
            # Verify all pages were extracted
            extracted_page_nums = [page_num for page_num, _ in page_chunks]
            logger.info(f"Extracted {len(extracted_page_nums)} pages: {extracted_page_nums}")
            
            logger.info(f"Extracted {total_pages} pages")
            return total_pages, page_chunks
            
        except Exception as e:
            logger.error(f"Failed to extract markdown from PDF: {e}")
            raise DocumentProcessingError(
                document_id="unknown",
                reason=f"PDF to markdown extraction failed: {str(e)}"
            )
        finally:
            # Clean up resources
            if pdf_doc:
                pdf_doc.close()
            if tmp_pdf_path and os.path.exists(tmp_pdf_path):
                os.unlink(tmp_pdf_path)
    
    async def _process_pages_batch(
        self, page_chunks: List[Tuple[int, str]]
    ) -> List[Tuple[MarkdownDocument, int]]:
        """
        Process multiple pages in batch using MarkdownChef for better performance.
        
        Uses contextlib.ExitStack to manage temporary files and ensure proper cleanup.
        Process all pages in a single batch call to leverage internal optimizations.
        
        Args:
            page_chunks: List of (page_number, markdown_content) tuples
            
        Returns:
            List of (MarkdownDocument, page_number) tuples
        """
        from contextlib import ExitStack
        from pathlib import Path
        
        if not page_chunks:
            logger.warning("No pages to process")
            return []
        
        logger.info(f"Batch processing {len(page_chunks)} pages with MarkdownChef")
        
        # Create all temporary files and keep track of them
        with ExitStack() as stack:
            temp_files = []
            
            # Create temp file for each page
            for page_num, page_content in page_chunks:
                temp_file = stack.enter_context(tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".md",
                    delete=False,
                    encoding="utf-8"
                ))
                temp_file.write(page_content)
                temp_file.flush()
                temp_files.append((Path(temp_file.name), page_num))
                logger.debug(f"Created temp file for page {page_num}: {temp_file.name}")
            
            try:
                # Process all files in batch
                file_paths = [path for path, _ in temp_files]
                logger.info(f"Calling MarkdownChef.process_batch with {len(file_paths)} files")
                
                # Run in thread to avoid blocking
                processed_docs = await asyncio.to_thread(
                    self.chef.process_batch,
                    file_paths
                )
                
                logger.info(f"Batch processing completed: {len(processed_docs)} documents")
                
                # Map results back to page numbers
                if len(processed_docs) != len(temp_files):
                    logger.error(
                        f"CRITICAL: Batch processing returned {len(processed_docs)} documents "
                        f"but expected {len(temp_files)}!"
                    )
                
                # Pair each processed document with its page number
                page_documents = [
                    (doc, page_num) 
                    for doc, (_, page_num) in zip(processed_docs, temp_files)
                ]
                
                # Verify batch processing didn't lose pages
                if len(page_documents) != len(page_chunks):
                    logger.error(
                        f"Page count mismatch: input={len(page_chunks)}, "
                        f"output={len(page_documents)}"
                    )
                
                return page_documents
                
            finally:
                # Clean up all temporary files
                for path, page_num in temp_files:
                    try:
                        path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {path}: {e}")
    
    async def _process_with_chef(self, markdown_content: str) -> MarkdownDocument:
        """Process markdown content with MarkdownChef."""
        tmp_path = ""
        try:
            # Create a temporary file to hold the markdown content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
                tmp.write(markdown_content)
                tmp_path = tmp.name
            
            logger.info(f"Processing temporary markdown file: {tmp_path}")
            
            # Process the file with the chef
            chonkie_doc = self.chef.process(tmp_path)
            
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