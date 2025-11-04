"""Use case for chatting with context using RAG (Retrieval-Augmented Generation)."""

from typing import List, Optional

from app.features.chat.domain.entities import QueryContext, RetrievedChunk, ChatResponse
from app.features.chat.application.query_documents import QueryDocuments
from app.features.documents.infrastructure.llm_groq_service import LLMGroqService
from app.core.config import settings
from app.core.logger import get_logger
from app.shared.exceptions import ExternalAPIError

logger = get_logger(__name__)


class ChatWithContext:
    """
    Implements the RAG (Retrieval-Augmented Generation) pipeline:
    1. Retrieve relevant document chunks
    2. Build context from retrieved chunks
    3. Generate answer using LLM with context
    """
    
    def __init__(self, query_use_case: QueryDocuments, llm_service: LLMGroqService):
        """
        Initialize with query use case and LLM service.
        
        Args:
            query_use_case: Use case for retrieving relevant chunks
            llm_service: Service for LLM generation
        """
        self.query_use_case = query_use_case
        self.llm_service = llm_service
    
    async def execute(
        self,
        user_id: str,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        hybrid_alpha: float = 0.7,
        expand_context: bool = True,
        score_threshold: float = 0.7
    ) -> ChatResponse:
        """
        Execute the RAG pipeline to generate a context-aware answer.
        
        Args:
            user_id: User identifier
            query: The user's question
            document_ids: Optional list of document IDs to search in
            top_k: Number of relevant chunks to retrieve
            hybrid_alpha: Balance between dense and sparse search
            expand_context: Whether to retrieve neighboring chunks for high-scoring results
            score_threshold: Minimum score to trigger context expansion
            
        Returns:
            ChatResponse with answer and sources
        """
        logger.info(f"Starting RAG pipeline for query: '{query[:50]}...'")
        
        # Step 1: Retrieve relevant chunks
        context = QueryContext(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            top_k=top_k,
            hybrid_alpha=hybrid_alpha,
            expand_context=expand_context,
            score_threshold=score_threshold
        )
        
        retrieved_chunks = await self.query_use_case.execute(context)
        
        # Step 2: Build context from chunks
        context_text = self._build_context(retrieved_chunks)
        
        # Step 3: Generate answer with LLM
        answer = await self._generate_answer(query, context_text)
        
        # Step 4: Create response
        response = ChatResponse(
            query=query,
            answer=answer,
            sources=retrieved_chunks,
            model=settings.groq_model,
            metadata={
                "top_k": top_k,
                "hybrid_alpha": hybrid_alpha,
                "num_sources": len(retrieved_chunks),
                "context_length": len(context_text)
            }
        )
        
        logger.info(f"RAG pipeline completed. Answer length: {len(answer)} chars")
        return response
    
    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found in your documents."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = chunk.get_source_info()
            context_parts.append(
                f"[Source {i}: {source_info}]\n{chunk.content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        logger.debug(f"Built context from {len(chunks)} chunks ({len(context)} chars)")
        
        return context
    
    async def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using the LLM with retrieved context.
        
        Args:
            query: The user's question
            context: The retrieved context
            
        Returns:
            Generated answer
        """
        try:
            # Build the prompt with context
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from documents.

Instructions:
- Answer the question using ONLY the information from the context provided
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but comprehensive
- Cite the source numbers (e.g., [Source 1]) when referencing specific information
- If you're unsure, acknowledge it rather than making up information"""

            user_prompt = f"""Context from documents:
{context}

---

Question: {query}

Answer:"""

            logger.info("Generating answer with LLM...")
            
            # Call LLM service
            chat_completion = self.llm_service.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=settings.groq_model,
                temperature=0.3,  # Lower temperature for more factual answers
                max_tokens=1024,
                top_p=0.9,
            )
            
            answer = chat_completion.choices[0].message.content
            logger.info(f"LLM generated answer ({len(answer)} chars)")
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer with LLM: {e}", exc_info=True)
            raise ExternalAPIError(
                service="Groq",
                message=f"Failed to generate answer: {str(e)}"
            )
