#!/usr/bin/env python3
"""
Command-line utility to upload files to Qdrant for Agent 2 Ticket System
"""

import sys
import argparse
from pathlib import Path
from file_uploader import DocumentUploader

def main():
    parser = argparse.ArgumentParser(
        description='Upload documents to Qdrant vector database'
    )
    parser.add_argument(
        'path',
        help='File or directory path to upload'
    )
    parser.add_argument(
        '-c', '--category',
        default='Document',
        help='Category for the documents (default: Document)'
    )
    parser.add_argument(
        '--collection',
        default='agent2_tickets',
        help='Qdrant collection name (default: agent2_tickets)'
    )
    parser.add_argument(
        '--qdrant-url',
        default='http://localhost:6333',
        help='Qdrant URL (default: http://localhost:6333)'
    )
    parser.add_argument(
        '--ollama-url',
        default='http://localhost:11434',
        help='Ollama URL (default: http://localhost:11434)'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Recursively process directories'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show collection statistics after upload'
    )
    
    args = parser.parse_args()
    
    # Initialize uploader
    uploader = DocumentUploader(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        ollama_url=args.ollama_url
    )
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path '{args.path}' does not exist")
        sys.exit(1)
    
    # Upload
    if path.is_file():
        print(f"Uploading file: {path}")
        result = uploader.upload_file(
            file_path=str(path),
            category=args.category
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)
        else:
            print(f"✅ Success!")
            print(f"   Filename: {result['filename']}")
            print(f"   Chunks uploaded: {result['chunks_uploaded']}")
            print(f"   Category: {result['category']}")
            print(f"   Collection: {result['collection']}")
    
    elif path.is_dir():
        print(f"Uploading directory: {path}")
        print(f"Recursive: {args.recursive}")
        print(f"Category: {args.category}")
        print()
        
        results = uploader.upload_directory(
            directory_path=str(path),
            category=args.category,
            recursive=args.recursive
        )
        
        success_count = sum(1 for r in results if "error" not in r)
        error_count = len(results) - success_count
        
        print(f"\n{'='*50}")
        print(f"Upload Summary:")
        print(f"  Total files processed: {len(results)}")
        print(f"  ✅ Successful: {success_count}")
        print(f"  ❌ Errors: {error_count}")
        
        if error_count > 0:
            print(f"\nErrors:")
            for result in results:
                if "error" in result:
                    print(f"  - {result.get('filename', 'unknown')}: {result['error']}")
    
    else:
        print(f"Error: '{args.path}' is neither a file nor a directory")
        sys.exit(1)
    
    # Show stats if requested
    if args.stats:
        print(f"\n{'='*50}")
        print("Collection Statistics:")
        stats = uploader.get_collection_stats()
        
        if "error" in stats:
            print(f"  Error: {stats['error']}")
        else:
            print(f"  Collection: {stats['collection_name']}")
            print(f"  Total points: {stats['total_points']}")
            print(f"  Vector size: {stats['vector_size']}")
            
            if stats['document_types']:
                print(f"\n  Document Types:")
                for doc_type, count in stats['document_types'].items():
                    print(f"    - {doc_type}: {count}")
            
            if stats['categories']:
                print(f"\n  Categories:")
                for category, count in stats['categories'].items():
                    print(f"    - {category}: {count}")

if __name__ == "__main__":
    main()
