"""Use case for chunking different content types using Chonkie's specialized chunkers."""

from typing import List, Optional

# Import the types from Chonkie for clear type hinting
from chonkie import MarkdownDocument
from chonkie import RecursiveChunker, CodeChunker,TableChunker

from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.value_objects import ChunkingConfig
from app.features.documents.infrastructure.llm_groq_service import LLMGroqService
from app.shared.exceptions import ChunkingError
from app.core.logger import get_logger

logger = get_logger(__name__)


class ChunkDocument:
    """
    Chunks a processed Chonkie MarkdownDocument using specialized chunkers.
    - Text is chunked using RecursiveChunker.
    - Code and tables are summarized by an LLM before being chunked to capture context.
    - Page numbers are tracked for each chunk.
    """

    def __init__(self, llm_service: LLMGroqService, config: ChunkingConfig = None, process_document=None):
        self.config = config or ChunkingConfig()
        self.llm_service = llm_service
        self.process_document = process_document  # Reference to ProcessDocument for page mapping

    async def execute(self, chonkie_doc: MarkdownDocument, document_id: str, page_chunks: List = None) -> List[DocumentChunk]:
        """
        Chunks the components of a MarkdownDocument using the correct chunker for each type.

        Args:
            chonkie_doc: The structured document object from MarkdownChef.
            document_id: The ID of the parent document.
            page_chunks: Optional list of (page_number, content) tuples for page tracking.

        Returns:
            A flat list of all DocumentChunk entities, ready for embedding.
        """
        try:
            logger.info(f"Starting specialized chunking for document {document_id}")

            all_chunks: List[DocumentChunk] = []
            
            # 1. Chunk the main text content
            all_chunks.extend(self._chunk_text(document_id, chonkie_doc.chunks))

            # 2. Chunk the code blocks (with summarization)
            all_chunks.extend(await self._chunk_code(document_id, chonkie_doc.code))

            # 3. Chunk the tables (with summarization)
            all_chunks.extend(await self._chunk_tables(document_id, chonkie_doc.tables))

            # Re-index all chunks sequentially and attempt to assign page numbers
            pages_assigned = 0
            for i, chunk in enumerate(all_chunks):
                chunk.chunk_index = i
                
                # Try to determine page number if not already set
                if chunk.page_number == 0 and self.process_document:
                    page_num = self.process_document.get_page_number_for_content(chunk.content)
                    if page_num > 0:
                        chunk.page_number = page_num
                        chunk.metadata["page_number"] = page_num
                        pages_assigned += 1

            logger.info(f"Created {len(all_chunks)} total chunks from document {document_id}")
            logger.info(f"Page numbers assigned to {pages_assigned}/{len(all_chunks)} chunks")
            return all_chunks

        except Exception as e:
            logger.error(f"Failed to chunk document {document_id}: {e}", exc_info=True)
            raise ChunkingError(f"Chunking failed: {str(e)}")

    def _chunk_text(self, document_id: str, text_blocks: List[str]) -> List[DocumentChunk]:
        """Chunks natural language text using RecursiveChunker with proper configuration."""
        # Configure chunker with our settings
        chunker = RecursiveChunker(
            chunk_size=self.config.chunk_size,
        )
        chunks = []
        for text_block in text_blocks:
            content_to_chunk = text_block.text if hasattr(text_block, 'text') else str(text_block)
            if not content_to_chunk.strip():
                continue
            for chonkie_chunk in chunker.chunk(content_to_chunk):
                chunks.append(DocumentChunk(
                    document_id=document_id,
                    content=chonkie_chunk.text,
                    token_count=chonkie_chunk.token_count,
                    metadata={"type": "text", "chunker": "sentence"}
                ))
        return chunks

    async def _chunk_code(self, document_id: str, code_blocks: List) -> List[DocumentChunk]:
        """Summarizes and chunks code blocks."""
        # Configure chunker with our settings
        chunker = CodeChunker(
            chunk_size=self.config.chunk_size,
        )
        chunks = []
        for i, code_block in enumerate(code_blocks):
            content_to_summarize = code_block.content if hasattr(code_block, 'content') else str(code_block)
            if not content_to_summarize.strip():
                continue

            # Get a natural language summary from the LLM with error handling
            try:
                summary = self.llm_service.summarize_code(content_to_summarize)
            except Exception as e:
                logger.warning(f"Failed to summarize code block {i} for document {document_id}: {e}")
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
                        "language": code_block.language or "unknown",
                        "chunker": "recursive_after_summary",
                        "source_block_index": i
                    }
                ))
        return chunks

    async def _chunk_tables(self, document_id: str, tables: List) -> List[DocumentChunk]:
        """Summarizes and chunks tables."""
        # Configure chunker with our settings
        chunker = TableChunker(
            chunk_size=self.config.chunk_size,
        )
        chunks = []
        for i, table in enumerate(tables):
            content_to_summarize = table.content if hasattr(table, 'content') else str(table)
            if not content_to_summarize.strip():
                continue

            # Get a natural language summary from the LLM with error handling
            try:
                summary = self.llm_service.summarize_table(content_to_summarize)
            except Exception as e:
                logger.warning(f"Failed to summarize table {i} for document {document_id}: {e}")
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
                        "source_table_index": i
                    }
                ))
        return chunks