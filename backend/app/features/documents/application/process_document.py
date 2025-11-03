"""Use case for extracting content from a PDF and processing it into a structured
Chonkie MarkdownDocument."""

import tempfile
import os
import uuid

# Import the return type for type hinting
from chonkie import MarkdownDocument
from chonkie import MarkdownChef
import pymupdf4llm
import pymupdf
from app.features.documents.domain.value_objects import FileUpload
from app.shared.exceptions import DocumentProcessingError
from app.core.logger import get_logger

logger = get_logger(__name__)


class ProcessDocument:
    """
    Extracts markdown from a PDF and processes it using Chonkie's MarkdownChef,
    returning a structured MarkdownDocument object.
    """
    
    def __init__(self, tokenizer: str = "gpt2"):
        self.tokenizer = tokenizer
    
    async def execute(self, file_upload: FileUpload) -> MarkdownDocument:
        """
        Processes an uploaded PDF into a Chonkie MarkdownDocument.
        
        Args:
            file_upload: The uploaded file value object.
            
        Returns:
            A chonkie.documents.MarkdownDocument object containing structured data.
        """
        logger.info(f"Starting full processing for {file_upload.filename}")
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_upload.content)
                tmp.flush()
                markdown_content = pymupdf4llm.to_markdown(tmp.name)
            logger.info(f"Extracted markdown content from PDF, length: {len(markdown_content)} characters")
        except Exception as e:
            logger.error(f"Failed to extract markdown from PDF: {e}")
            raise DocumentProcessingError(
                document_id="unknown", # We don't have a DB ID at this stage
                reason=f"PDF to markdown extraction failed: {str(e)}"
            )
          # --- START: SAVE MARKDOWN TO FILE ---
        try:
            output_dir = "output_markdown"
            os.makedirs(output_dir, exist_ok=True)
            markdown_filepath = os.path.join(output_dir, f"{uuid.uuid4()}.md")
            with open(markdown_filepath, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.info(f"Saved extracted markdown for inspection to: {markdown_filepath}")
        except Exception as e:
            logger.warning(f"Could not save markdown file for inspection: {e}")
        # --- END: SAVE MARKDOWN TO FILE ---
        # 2. Process the markdown string with MarkdownChef
        chef = MarkdownChef(tokenizer=self.tokenizer)
        
        # MarkdownChef needs a file path, so we create a temporary file
        tmp_path = ""
        try:
            # Create a temporary file to hold the markdown content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
                tmp.write(markdown_content)
                tmp_path = tmp.name
            
            logger.info(f"Processing temporary markdown file: {tmp_path}")
            
            # Process the file with the chef
            chonkie_doc = chef.process(tmp_path)
            
            logger.info(
                f"MarkdownChef processed: {len(chonkie_doc.chunks)} text chunks, "
                f"{len(chonkie_doc.code)} code blocks, {len(chonkie_doc.tables)} tables."
            )
            
            return chonkie_doc
            
        except Exception as e:
            logger.error(f"Failed to process document with MarkdownChef: {e}")
            raise DocumentProcessingError(
                document_id="unknown", # We don't have a DB ID at this stage
                reason=f"MarkdownChef processing failed: {str(e)}"
            )
        finally:
            # Clean up the temporary file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)