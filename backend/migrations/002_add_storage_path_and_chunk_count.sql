-- Migration: Add storage_path and chunk_count columns to documents table
-- Date: 2025-11-04

-- Add storage_path column to track where files are stored
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS storage_path VARCHAR NULL;

-- Add chunk_count column (if not already exists from previous migration)
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 0;

-- Add index on file_hash for duplicate detection
CREATE INDEX IF NOT EXISTS idx_documents_file_hash 
ON documents(file_hash);

-- Add composite index for user-specific duplicate detection
CREATE INDEX IF NOT EXISTS idx_documents_user_file_hash 
ON documents(user_id, file_hash);

-- Update existing records to set chunk_count to 0 if NULL
UPDATE documents 
SET chunk_count = 0 
WHERE chunk_count IS NULL;
