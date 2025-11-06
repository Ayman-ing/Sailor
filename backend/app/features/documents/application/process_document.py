import asyncio
import tempfile
import os
from typing import Tuple, List
from contextlib import ExitStack
from pathlib import Path

from chonkie import MarkdownDocument, MarkdownChef
import pymupdf4llm
import fitz
from app.features.documents.domain.value_objects import FileUpload
from app.shared.exceptions import DocumentProcessingError
from app.core.logger import get_logger

logger = get_logger(__name__)


class ProcessDocument:
    
    def __init__(self, tokenizer: str = "gpt2"):
        self.tokenizer = tokenizer
        self.chef = MarkdownChef(tokenizer=self.tokenizer)
    


    
    async def execute(self, file_upload: FileUpload) -> Tuple[List[Tuple[MarkdownDocument, int]], int, List[Tuple[int, str]]]:
        total_pages, page_chunks = await self._extract_with_pages(file_upload)
        page_documents = await self._process_pages_batch(page_chunks)
        
        return page_documents, total_pages, page_chunks
    




    async def _extract_with_pages(self, file_upload: FileUpload) -> Tuple[int, List[Tuple[int, str]]]:
        tmp_pdf_path = None
        pdf_doc = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_upload.content)
                tmp.flush()
                tmp_pdf_path = tmp.name
            
            pdf_doc = fitz.open(tmp_pdf_path)
            
            page_data = pymupdf4llm.to_markdown(
                tmp_pdf_path,
                page_chunks=True,
                page_separators=False
            )
            
            page_chunks = []
            
            for i, page_dict in enumerate(page_data):
                page_num = page_dict.get('metadata', {}).get('page', i + 1)
                content = page_dict.get('text', '')
                
                if not content.strip():
                    try:
                        page = pdf_doc[i]
                        direct_text = page.get_text("text")
                        if direct_text.strip():
                            content = direct_text
                    except Exception as e:
                        logger.error(f"Page {page_num} fallback extraction failed: {e}")
                
                page_chunks.append((page_num, content))
            
            total_pages = len(page_chunks)
            return total_pages, page_chunks
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise DocumentProcessingError(
                document_id="unknown",
                reason=f"PDF extraction failed: {str(e)}"
            )
        finally:
            if pdf_doc:
                pdf_doc.close()
            if tmp_pdf_path and os.path.exists(tmp_pdf_path):
                os.unlink(tmp_pdf_path)
    
    async def _process_pages_batch(
        self, page_chunks: List[Tuple[int, str]]
    ) -> List[Tuple[MarkdownDocument, int]]:
        if not page_chunks:
            return []
        
        with ExitStack() as stack:
            temp_files = []
            
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
            
            try:
                file_paths = [path for path, _ in temp_files]
                
                processed_docs = await asyncio.to_thread(
                    self.chef.process_batch,
                    file_paths
                )
                
                page_documents = [
                    (doc, page_num) 
                    for doc, (_, page_num) in zip(processed_docs, temp_files)
                ]
                
                return page_documents
                
            finally:
                for path, page_num in temp_files:
                    try:
                        path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {path}: {e}")
