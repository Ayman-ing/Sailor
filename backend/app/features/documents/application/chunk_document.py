from typing import List, Tuple

from chonkie import RecursiveChunker, CodeChunker, TableChunker

from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.value_objects import ChunkingConfig
from app.features.documents.infrastructure.llm_groq_service import LLMGroqService
from app.shared.exceptions import ChunkingError
from app.core.logger import get_logger

logger = get_logger(__name__)


class ChunkDocument:

    def __init__(self, llm_service: LLMGroqService, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.llm_service = llm_service

    async def execute(self, page_documents: List[Tuple], document_id: str) -> List[DocumentChunk]:
        try:
            all_chunks: List[DocumentChunk] = []
            
            for chonkie_doc, page_number in page_documents:
                positioned_chunks = []
                
                for text_block in chonkie_doc.chunks:
                    chunks = await self._chunk_text(document_id, text_block)
                    for chunk in chunks:
                        positioned_chunks.append((self._get_position(text_block), chunk))
                
                for code_block in chonkie_doc.code:
                    chunks = await self._chunk_code(document_id, code_block)
                    for chunk in chunks:
                        positioned_chunks.append((self._get_position(code_block), chunk))
                
                for table in chonkie_doc.tables:
                    chunks = await self._chunk_table(document_id, table)
                    for chunk in chunks:
                        positioned_chunks.append((self._get_position(table), chunk))
                
                positioned_chunks.sort(key=lambda x: x[0])
                
                for position, chunk in positioned_chunks:
                    chunk.page_number = page_number
                    all_chunks.append(chunk)

            for i, chunk in enumerate(all_chunks):
                chunk.chunk_index = i

            return all_chunks

        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise ChunkingError(f"Chunking failed: {str(e)}")
    
    def _get_position(self, block) -> int:
        if hasattr(block, 'start_index'):
            return block.start_index
        elif hasattr(block, 'index'):
            return block.index
        return hash(str(block)[:100]) % 1000000

    async def _chunk_text(self, document_id: str, text_block) -> List[DocumentChunk]:
        chunker = RecursiveChunker(chunk_size=self.config.chunk_size)
        content = text_block.text if hasattr(text_block, 'text') else str(text_block)
        
        if not content.strip():
            return []
            
        chunks = []
        for chonkie_chunk in chunker(content):
            chunks.append(DocumentChunk(
                document_id=document_id,
                content=chonkie_chunk.text,
                token_count=chonkie_chunk.token_count,
                metadata={"type": "text", "chunker": "recursive"}
            ))
        return chunks

    async def _chunk_code(self, document_id: str, code_block) -> List[DocumentChunk]:
        chunker = CodeChunker(chunk_size=self.config.chunk_size)
        content = code_block.content if hasattr(code_block, 'content') else str(code_block)
        
        if not content.strip():
            return []

        try:
            summary = self.llm_service.summarize_code(content)
        except Exception as e:
            logger.warning(f"Code summarization failed: {e}")
            summary = f"Code block: {content[:100]}..."

        combined = f"""Summary: {summary}

---

Code:
```
{content}
```"""

        chunks = []
        for chonkie_chunk in chunker(combined):
            chunks.append(DocumentChunk(
                document_id=document_id,
                content=chonkie_chunk.text,
                token_count=chonkie_chunk.token_count,
                metadata={
                    "type": "code_with_summary",
                    "language": code_block.language if hasattr(code_block, 'language') else "unknown",
                    "chunker": "code"
                }
            ))
        return chunks

    async def _chunk_table(self, document_id: str, table) -> List[DocumentChunk]:
        chunker = TableChunker(chunk_size=self.config.chunk_size)
        content = table.content if hasattr(table, 'content') else str(table)
        
        if not content.strip():
            return []

        try:
            summary = self.llm_service.summarize_table(content)
        except Exception as e:
            logger.warning(f"Table summarization failed: {e}")
            summary = f"Table: {content[:100]}..."

        combined = f"""Summary: {summary}

---

Table:
{content}"""

        chunks = []
        for chonkie_chunk in chunker(combined):
            chunks.append(DocumentChunk(
                document_id=document_id,
                content=chonkie_chunk.text,
                token_count=chonkie_chunk.token_count,
                metadata={"type": "table_with_summary", "chunker": "table"}
            ))
        return chunks