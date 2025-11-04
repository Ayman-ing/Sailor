"""Test script to verify pymupdf4llm page extraction."""

import pymupdf4llm
import sys

def test_page_extraction(pdf_path: str):
    """Test if pymupdf4llm correctly extracts page numbers."""
    
    print(f"Testing page extraction from: {pdf_path}\n")
    print("=" * 80)
    
    # Test 1: Extract with page_chunks=True
    print("\n1. Testing with page_chunks=True:")
    print("-" * 80)
    
    try:
        result = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        
        print(f"Type of result: {type(result)}")
        print(f"Number of pages: {len(result) if isinstance(result, list) else 'N/A'}")
        print()
        
        if isinstance(result, list):
            # Show first few pages
            for i, page_data in enumerate(result[:3]):  # Show first 3 pages
                print(f"\nPage {i+1} structure:")
                print(f"  Type: {type(page_data)}")
                
                if isinstance(page_data, dict):
                    print(f"  Keys: {list(page_data.keys())}")
                    
                    # Show metadata
                    if 'metadata' in page_data:
                        print(f"  Metadata: {page_data['metadata']}")
                    
                    # Show text preview
                    if 'text' in page_data:
                        text_preview = page_data['text'][:200].replace('\n', ' ')
                        print(f"  Text preview: {text_preview}...")
                    
                print()
            
            # Show total pages
            if len(result) > 3:
                print(f"... and {len(result) - 3} more pages")
                
        else:
            print(f"Unexpected result type: {type(result)}")
            print(f"Content preview: {str(result)[:500]}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Extract without page_chunks (default)
    print("\n\n2. Testing without page_chunks (default):")
    print("-" * 80)
    
    try:
        result_default = pymupdf4llm.to_markdown(pdf_path)
        print(f"Type of result: {type(result_default)}")
        print(f"Content length: {len(result_default) if isinstance(result_default, str) else 'N/A'}")
        
        if isinstance(result_default, str):
            preview = result_default[:500].replace('\n', ' ')
            print(f"Content preview: {preview}...")
        else:
            print(f"Unexpected result type: {type(result_default)}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_page_extraction.py <path_to_pdf>")
        print("\nExample: python test_page_extraction.py ./pdf_sample/sample.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    test_page_extraction(pdf_path)
