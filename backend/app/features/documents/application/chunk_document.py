"""Use case for chunking different content types using Chonkie's specialized chunkers."""

from typing import List, Tuple

# Import the types from Chonkie for clear type hinting
from chonkie import RecursiveChunker, CodeChunker, TableChunker

from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.value_objects import ChunkingConfig
from app.features.documents.infrastructure.llm_groq_service import LLMGroqService
from app.shared.exceptions import ChunkingError
from app.core.logger import get_logger

logger = get_logger(__name__)


class ChunkDocument:
    """
    Chunks processed Chonkie MarkdownDocuments using specialized chunkers.
    Processes each page separately to maintain accurate page numbers.
    - Text is chunked using RecursiveChunker.
    - Code and tables are summarized by an LLM before being chunked to capture context.
    - All chunks are tagged with their source page number.
    """

    def __init__(self, llm_service: LLMGroqService, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.llm_service = llm_service

    async def execute(self, page_documents: List[Tuple], document_id: str) -> List[DocumentChunk]:
        """
        Chunks the components of MarkdownDocuments (one per page) using the correct chunker for each type.
        Processes all components in their original document order to preserve sequence.

        Args:
            page_documents: List of (MarkdownDocument, page_number) tuples - one per page.
            document_id: The ID of the parent document.

        Returns:
            A flat list of all DocumentChunk entities, ready for embedding.
        """
        try:
            logger.info(f"Starting specialized chunking for document {document_id}")

            all_chunks: List[DocumentChunk] = []
            
            # Process each page's document separately
            for chonkie_doc, page_number in page_documents:
                logger.debug(f"Chunking page {page_number}")
                
                # Collect components from this page with their positions for ordering
                positioned_chunks = []
                
                # Process text chunks
                for text_block in chonkie_doc.chunks:
                    chunks = await self._chunk_single_text(document_id, text_block)
                    for chunk in chunks:
                        positioned_chunks.append((self._get_position(text_block), chunk))
                
                # Process code blocks
                for i, code_block in enumerate(chonkie_doc.code):
                    chunks = await self._chunk_single_code(document_id, code_block, i)
                    for chunk in chunks:
                        positioned_chunks.append((self._get_position(code_block), chunk))
                
                # Process tables
                for i, table in enumerate(chonkie_doc.tables):
                    chunks = await self._chunk_single_table(document_id, table, i)
                    for chunk in chunks:
                        positioned_chunks.append((self._get_position(table), chunk))
                
                # Sort by position to maintain order within this page
                positioned_chunks.sort(key=lambda x: x[0])
                
                # Assign page number to all chunks from this page
                for position, chunk in positioned_chunks:
                    chunk.page_number = page_number
                    all_chunks.append(chunk)

            # Re-index all chunks sequentially across all pages
            for i, chunk in enumerate(all_chunks):
                chunk.chunk_index = i

            pages_assigned = sum(1 for chunk in all_chunks if chunk.page_number > 0)
            logger.info(f"Created {len(all_chunks)} total chunks from document {document_id}")
            logger.info(f"Page numbers assigned to {pages_assigned}/{len(all_chunks)} chunks (page-by-page processing)")
            return all_chunks

        except Exception as e:
            logger.error(f"Failed to chunk document {document_id}: {e}", exc_info=True)
            raise ChunkingError(f"Chunking failed: {str(e)}")
    
    def _get_position(self, block) -> int:
        """
        Get the position of a block in the original document.
        Chonkie objects have a 'start_index' attribute that tracks position.
        """
        if hasattr(block, 'start_index'):
            return block.start_index
        elif hasattr(block, 'index'):
            return block.index
        # Fallback: use hash as position (not ideal but maintains some order)
        return hash(str(block)[:100]) % 1000000

    async def _chunk_single_text(self, document_id: str, text_block) -> List[DocumentChunk]:
        """Chunks a single natural language text block using RecursiveChunker."""
        chunker = RecursiveChunker(
            chunk_size=self.config.chunk_size,
        )
        chunks = []
        content_to_chunk = text_block.text if hasattr(text_block, 'text') else str(text_block)
        if not content_to_chunk.strip():
            return chunks
            
        for chonkie_chunk in chunker.chunk(content_to_chunk):
            chunks.append(DocumentChunk(
                document_id=document_id,
                content=chonkie_chunk.text,
                token_count=chonkie_chunk.token_count,
                metadata={"type": "text", "chunker": "sentence"}
            ))
        return chunks

    async def _chunk_single_code(self, document_id: str, code_block, block_index: int) -> List[DocumentChunk]:
        """Summarizes and chunks a single code block."""
        chunker = CodeChunker(
            chunk_size=self.config.chunk_size,
        )
        chunks = []
        content_to_summarize = code_block.content if hasattr(code_block, 'content') else str(code_block)
        if not content_to_summarize.strip():
            return chunks

        # Get a natural language summary from the LLM with error handling
        try:
            summary = self.llm_service.summarize_code(content_to_summarize)
        except Exception as e:
            logger.warning(f"Failed to summarize code block {block_index} for document {document_id}: {e}")
            # Fallback: use truncated code as summary
            summary = f"Code block (summarization failed): {content_to_summarize[:100]}..."

        # Prepend the summary to the original code for context
        combined_content = f"""Summary of the following code:
{summary}

---

Original Code:
```
{content_to_summarize}
```"""

        for chonkie_chunk in chunker.chunk(combined_content):
            chunks.append(DocumentChunk(
                document_id=document_id,
                content=chonkie_chunk.text,
                token_count=chonkie_chunk.token_count,
                metadata={
                    "type": "code_with_summary",
                    "language": code_block.language if hasattr(code_block, 'language') else "unknown",
                    "chunker": "recursive_after_summary",
                    "source_block_index": block_index
                }
            ))
        return chunks

    async def _chunk_single_table(self, document_id: str, table, table_index: int) -> List[DocumentChunk]:
        """Summarizes and chunks a single table."""
        chunker = TableChunker(
            chunk_size=self.config.chunk_size,
        )
        chunks = []
        content_to_summarize = table.content if hasattr(table, 'content') else str(table)
        if not content_to_summarize.strip():
            return chunks

        # Get a natural language summary from the LLM with error handling
        try:
            summary = self.llm_service.summarize_table(content_to_summarize)
        except Exception as e:
            logger.warning(f"Failed to summarize table {table_index} for document {document_id}: {e}")
            # Fallback: use truncated table as summary
            summary = f"Table (summarization failed): {content_to_summarize[:100]}..."

        # Prepend the summary to the original table for context
        combined_content = f"""Summary of the following table:
{summary}

---

Original Table:
{content_to_summarize}"""

        for chonkie_chunk in chunker.chunk(combined_content):
            chunks.append(DocumentChunk(
                document_id=document_id,
                content=chonkie_chunk.text,
                token_count=chonkie_chunk.token_count,
                metadata={
                    "type": "table_with_summary",
                    "chunker": "recursive_after_summary",
                    "source_table_index": table_index
                }
            ))
        return chunks