"""
Utility script to view ChromaDB contents and metadata.

This script shows what warranty policies are stored in ChromaDB,
their embeddings, and metadata used for retrieval.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
import chromadb


def view_collection_info():
    """Show basic information about the ChromaDB collection."""
    print("=" * 80)
    print("CHROMADB COLLECTION INFO")
    print("=" * 80)

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

        print(f"\nCollection Name: {CHROMA_COLLECTION_NAME}")
        print(f"Persist Directory: {CHROMA_PERSIST_DIR}")
        print(f"Total Documents: {collection.count()}")

        return collection

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nCollection may not exist yet. Run: python src/ingest.py")
        return None


def view_all_documents(collection):
    """Show all documents in the collection."""
    if not collection:
        return

    print("\n" + "=" * 80)
    print("ALL DOCUMENTS IN CHROMADB")
    print("=" * 80)

    # Get all documents
    results = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    if not results['ids']:
        print("\nNo documents found. Run: python src/ingest.py")
        return

    print(f"\nTotal documents: {len(results['ids'])}\n")

    for idx, doc_id in enumerate(results['ids']):
        print(f"\n{idx + 1}. Document ID: {doc_id}")
        print("-" * 80)

        # Metadata
        metadata = results['metadatas'][idx]
        print(f"Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        # Document content (truncated)
        content = results['documents'][idx]
        print(f"\nContent Preview (first 300 chars):")
        print(f"{content[:300]}...")

        # Embedding info
        if results['embeddings'] and results['embeddings'][idx]:
            embedding = results['embeddings'][idx]
            print(f"\nEmbedding:")
            print(f"  Dimension: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")

        print("-" * 80)


def view_metadata_schema(collection):
    """Show the metadata schema used."""
    if not collection:
        return

    print("\n" + "=" * 80)
    print("METADATA SCHEMA")
    print("=" * 80)

    results = collection.get(limit=1, include=["metadatas"])

    if results['metadatas']:
        sample_metadata = results['metadatas'][0]

        print("\nMetadata Fields:")
        print("-" * 80)
        for key, value in sample_metadata.items():
            value_type = type(value).__name__
            print(f"  {key:<20} {value_type:<15} Example: {value}")

        print("\n" + "=" * 80)
        print("METADATA FIELD DESCRIPTIONS")
        print("=" * 80)
        print("""
Field Descriptions:

  product_id       - Unique identifier for the product (e.g., "HD-002")
                     Used to filter policies for specific products

  product_name     - Human-readable product name (e.g., "AirFlow Pro")
                     Used for display and matching

  source           - Path to original policy PDF file
                     Used for traceability and auditing

  chunk_index      - Index of this chunk within the document
                     Used when policies are split into multiple chunks

  total_chunks     - Total number of chunks for this document
                     Used to track document completeness

  ingestion_date   - Timestamp when policy was ingested
                     Used for versioning and freshness checks
        """)


def query_policies(collection, query_text: str, n_results: int = 3):
    """Query policies using semantic search."""
    if not collection:
        return

    print("\n" + "=" * 80)
    print(f"SEMANTIC SEARCH: '{query_text}'")
    print("=" * 80)

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    if not results['ids'][0]:
        print("\nNo results found.")
        return

    print(f"\nTop {n_results} Results:\n")

    for idx in range(len(results['ids'][0])):
        doc_id = results['ids'][0][idx]
        metadata = results['metadatas'][0][idx]
        content = results['documents'][0][idx]
        distance = results['distances'][0][idx]

        print(f"\n{idx + 1}. Relevance Score: {1 - distance:.4f} (distance: {distance:.4f})")
        print(f"   Document ID: {doc_id}")
        print(f"   Product: {metadata.get('product_name')} ({metadata.get('product_id')})")
        print(f"   Source: {metadata.get('source')}")
        print(f"\n   Content Preview:")
        print(f"   {content[:200]}...")
        print("-" * 80)


def view_by_product(collection, product_id: str = None):
    """Show all chunks for a specific product."""
    if not collection:
        return

    results = collection.get(include=["metadatas", "documents"])

    if not results['ids']:
        print("\nNo documents found.")
        return

    # Group by product
    products = {}
    for idx, doc_id in enumerate(results['ids']):
        metadata = results['metadatas'][idx]
        pid = metadata.get('product_id', 'Unknown')

        if pid not in products:
            products[pid] = {
                'name': metadata.get('product_name', 'Unknown'),
                'chunks': []
            }

        products[pid]['chunks'].append({
            'id': doc_id,
            'content': results['documents'][idx],
            'metadata': metadata
        })

    if product_id:
        # Show specific product
        if product_id in products:
            print(f"\n{'=' * 80}")
            print(f"PRODUCT: {products[product_id]['name']} ({product_id})")
            print(f"{'=' * 80}")
            print(f"\nTotal Chunks: {len(products[product_id]['chunks'])}\n")

            for idx, chunk in enumerate(products[product_id]['chunks']):
                print(f"\nChunk {idx + 1}:")
                print(f"  ID: {chunk['id']}")
                print(f"  Source: {chunk['metadata'].get('source')}")
                print(f"  Content Length: {len(chunk['content'])} chars")
                print(f"  Preview: {chunk['content'][:150]}...")
                print("-" * 80)
        else:
            print(f"\n❌ Product ID '{product_id}' not found.")
    else:
        # Show all products
        print(f"\n{'=' * 80}")
        print("ALL PRODUCTS IN CHROMADB")
        print(f"{'=' * 80}\n")

        for pid, data in products.items():
            print(f"  {pid:<15} {data['name']:<30} {len(data['chunks'])} chunks")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="View ChromaDB Contents")
    parser.add_argument("--info", action="store_true", help="Show collection info")
    parser.add_argument("--docs", action="store_true", help="Show all documents")
    parser.add_argument("--schema", action="store_true", help="Show metadata schema")
    parser.add_argument("--query", type=str, help="Semantic search query")
    parser.add_argument("--product", type=str, help="Show chunks for specific product ID")
    parser.add_argument("--products", action="store_true", help="List all products")
    parser.add_argument("--all", action="store_true", help="Show all information")

    args = parser.parse_args()

    if not any([args.info, args.docs, args.schema, args.query, args.product, args.products, args.all]):
        # Default: show info and products
        args.info = True
        args.products = True

    # Get collection
    collection = view_collection_info()

    if args.all:
        view_all_documents(collection)
        view_metadata_schema(collection)
        view_by_product(collection)
    else:
        if args.docs:
            view_all_documents(collection)

        if args.schema:
            view_metadata_schema(collection)

        if args.query:
            query_policies(collection, args.query)

        if args.products:
            view_by_product(collection)

        if args.product:
            view_by_product(collection, args.product)


if __name__ == "__main__":
    main()
