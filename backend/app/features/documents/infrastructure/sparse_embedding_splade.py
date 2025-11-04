"""SPLADE sparse embedding service implementation."""

from typing import List
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from qdrant_client.models import SparseVector
from app.core.logger import get_logger

logger = get_logger(__name__)


class SpladeSparseEmbeddingService:
    """
    Service for generating sparse embeddings using the SPLADE model.
    """

    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1", device: str = "cpu"):
        self.device = device
        logger.info(f"Loading SPLADE model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"SPLADE model loaded successfully on device: {self.device}")

    def generate_sparse_vectors(self, texts: List[str], batch_size: int = 8) -> List[SparseVector]:
        """
        Generate sparse embeddings for a batch of texts with batching support.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once (default: 8 for CPU, can be higher for GPU)
        
        Returns:
            List of SparseVector objects
        """
        logger.info(f"Generating sparse vectors for {len(texts)} texts in batches of {batch_size}...")
        
        all_sparse_vectors = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            current_batch = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch_texts)} texts)...")
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Apply ReLU and max pooling
                relu_log = torch.log(1 + torch.relu(logits))
                weighted_log = relu_log * inputs["attention_mask"].unsqueeze(-1)
                max_val, _ = torch.max(weighted_log, dim=1)

                # Convert to sparse vectors
                for vec in max_val:
                    cols = torch.nonzero(vec).squeeze().cpu().tolist()
                    weights = vec[vec != 0].cpu().tolist()

                    # Handle single non-zero value case
                    if isinstance(cols, int):
                        cols = [cols]
                        weights = [weights]

                    all_sparse_vectors.append(
                        SparseVector(
                            indices=cols,
                            values=weights
                        )
                    )
            
            # Clear GPU cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        logger.info(f"Generated {len(all_sparse_vectors)} sparse vectors successfully.")
        return all_sparse_vectors
