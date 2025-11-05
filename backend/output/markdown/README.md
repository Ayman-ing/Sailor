# Markdown Output Directory

This directory contains intermediate markdown outputs from document processing:

## File Types

### 1. `*_pymupdf4llm.md`
- Raw markdown extracted from PDF using pymupdf4llm
- Contains the full document converted to markdown format
- Preserves document structure, headings, lists, etc.

### 2. `*_chonkie.md`
- Processed output from Chonkie's MarkdownChef
- Shows how the document was chunked
- Includes:
  - Text chunks with token counts
  - Extracted code blocks
  - Extracted tables

## Purpose

These files are saved for:
- **Debugging**: See exactly what was extracted and how it was chunked
- **Quality Control**: Review processing results before indexing
- **Documentation**: Share processing results with team members
- **Analysis**: Compare different chunking strategies

## File Naming

Format: `{original_filename}_{timestamp}_{processor}.md`

Example:
- `research_paper_20251105_143022_pymupdf4llm.md`
- `research_paper_20251105_143022_chonkie.md`

## Notes

- These files are automatically generated during document upload
- Files are excluded from git (see `.gitignore`)
- Old files are not automatically deleted - manage manually
