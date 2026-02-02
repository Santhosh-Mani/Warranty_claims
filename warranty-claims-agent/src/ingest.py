"""
Policy Ingestion Pipeline.

Uses Docling to parse PDF warranty policies and stores them in ChromaDB
for semantic retrieval during claim processing.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    POLICIES_DIR,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    PRODUCT_ID_TO_NAME,
)

# ChromaDB and embeddings
import chromadb
from chromadb.config import Settings

# Try to import docling, fall back to simpler PDF parsing if not available
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not available, using fallback PDF parsing")

# Fallback PDF parsing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer


class PolicyIngester:
    """Handles ingestion of warranty policy documents into ChromaDB."""

    def __init__(self, persist_dir: Path = CHROMA_PERSIST_DIR):
        """Initialize the ingester with ChromaDB client and embedding model."""
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embedding model (lightweight, runs locally)
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"description": "Warranty policy documents"}
        )

        # Initialize document converter if available
        if DOCLING_AVAILABLE:
            self.converter = DocumentConverter()
        else:
            self.converter = None

    def parse_pdf_with_docling(self, pdf_path: Path) -> str:
        """Parse PDF using Docling for structured extraction."""
        if not DOCLING_AVAILABLE:
            raise RuntimeError("Docling not available")

        logger.info(f"Parsing with Docling: {pdf_path.name}")
        result = self.converter.convert(str(pdf_path))
        return result.document.export_to_markdown()

    def parse_pdf_with_pdfplumber(self, pdf_path: Path) -> str:
        """Fallback PDF parsing using pdfplumber."""
        if not PDFPLUMBER_AVAILABLE:
            raise RuntimeError("pdfplumber not available")

        logger.info(f"Parsing with pdfplumber: {pdf_path.name}")
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)

    def parse_pdf(self, pdf_path: Path) -> str:
        """Parse PDF using best available method."""
        if DOCLING_AVAILABLE:
            try:
                return self.parse_pdf_with_docling(pdf_path)
            except Exception as e:
                logger.warning(f"Docling failed, trying pdfplumber: {e}")

        if PDFPLUMBER_AVAILABLE:
            return self.parse_pdf_with_pdfplumber(pdf_path)

        raise RuntimeError("No PDF parsing library available. Install docling or pdfplumber.")

    def chunk_document(self, text: str, product_id: str) -> List[Dict[str, Any]]:
        """
        Split document into chunks based on markdown headers.

        Returns a list of chunks with metadata.
        """
        chunks = []
        current_section = "Introduction"
        current_content = []

        lines = text.split('\n')
        for line in lines:
            # Detect markdown headers
            if line.startswith('## '):
                # Save previous section
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        chunks.append({
                            "content": content,
                            "section": current_section,
                            "product_id": product_id
                        })
                # Start new section
                current_section = line.replace('## ', '').strip()
                current_content = []
            elif line.startswith('# '):
                # Main title - save as document header
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        chunks.append({
                            "content": content,
                            "section": current_section,
                            "product_id": product_id
                        })
                current_section = "Document Header"
                current_content = [line.replace('# ', '').strip()]
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                chunks.append({
                    "content": content,
                    "section": current_section,
                    "product_id": product_id
                })

        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def ingest_policy(self, pdf_path: Path) -> int:
        """
        Ingest a single policy PDF into ChromaDB.

        Returns the number of chunks added.
        """
        # Extract product ID from filename (e.g., "HD-001_AirFlow_Basic.pdf")
        filename = pdf_path.stem
        product_id = filename.split('_')[0] if '_' in filename else filename

        logger.info(f"Processing: {pdf_path.name} (Product: {product_id})")

        # Parse PDF to text/markdown
        text = self.parse_pdf(pdf_path)

        # Chunk the document
        chunks = self.chunk_document(text, product_id)
        logger.info(f"  Created {len(chunks)} chunks")

        if not chunks:
            logger.warning(f"  No chunks created for {pdf_path.name}")
            return 0

        # Prepare data for ChromaDB
        ids = [f"{product_id}_{i}" for i in range(len(chunks))]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "product_id": chunk["product_id"],
                "section": chunk["section"],
                "product_name": PRODUCT_ID_TO_NAME.get(chunk["product_id"], "Unknown"),
                "source_file": pdf_path.name
            }
            for chunk in chunks
        ]

        # Generate embeddings
        embeddings = self.generate_embeddings(documents)

        # Upsert to ChromaDB (handles duplicates)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"  Added {len(chunks)} chunks to ChromaDB")
        return len(chunks)

    def ingest_all_policies(self, policies_dir: Path = POLICIES_DIR) -> Dict[str, int]:
        """
        Ingest all PDF policies from a directory.

        Returns a dict mapping product_id to chunk count.
        """
        results = {}
        pdf_files = list(policies_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {policies_dir}")
            return results

        logger.info(f"Found {len(pdf_files)} policy PDFs to ingest")

        for pdf_path in pdf_files:
            try:
                chunk_count = self.ingest_policy(pdf_path)
                product_id = pdf_path.stem.split('_')[0]
                results[product_id] = chunk_count
            except Exception as e:
                logger.error(f"Failed to ingest {pdf_path.name}: {e}")
                results[pdf_path.stem] = 0

        return results

    def query(self, query_text: str, n_results: int = 3, product_id: str = None) -> List[Dict]:
        """
        Query the vector store for relevant policy sections.

        Args:
            query_text: The query string
            n_results: Number of results to return
            product_id: Optional filter by product ID

        Returns:
            List of matching documents with metadata
        """
        # Build where filter if product_id specified
        where_filter = {"product_id": product_id} if product_id else None

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0].tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })

        return formatted

    def get_policy_for_product(self, product_id: str) -> str:
        """
        Get the full policy text for a specific product.

        Args:
            product_id: The product ID (e.g., "HD-002")

        Returns:
            Combined policy text for the product
        """
        # Query all chunks for this product
        results = self.collection.get(
            where={"product_id": product_id},
            include=["documents", "metadatas"]
        )

        if not results or not results['documents']:
            return f"No policy found for product {product_id}"

        # Combine chunks, ordered by section
        section_order = [
            "Document Header",
            "Product Information",
            "Product Features",
            "Warranty Coverage",
            "What Is Covered",
            "Exclusions (What Is NOT Covered)",
            "Warranty Claim Process",
            "Contact Information",
            "Legal Notice"
        ]

        sections = {}
        for doc, meta in zip(results['documents'], results['metadatas']):
            section = meta.get('section', 'Other')
            if section not in sections:
                sections[section] = []
            sections[section].append(doc)

        # Build ordered output
        output_parts = []
        for section in section_order:
            if section in sections:
                output_parts.append(f"## {section}\n")
                output_parts.extend(sections[section])
                output_parts.append("")

        # Add any remaining sections
        for section, docs in sections.items():
            if section not in section_order:
                output_parts.append(f"## {section}\n")
                output_parts.extend(docs)
                output_parts.append("")

        return "\n".join(output_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the ingested policies."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": CHROMA_COLLECTION_NAME,
            "persist_dir": str(self.persist_dir)
        }


def main():
    """Main function to run the ingestion pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest warranty policies into ChromaDB")
    parser.add_argument(
        "--policies-dir",
        type=Path,
        default=POLICIES_DIR,
        help="Directory containing policy PDFs"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Test query after ingestion"
    )
    parser.add_argument(
        "--product",
        type=str,
        help="Filter query by product ID"
    )
    args = parser.parse_args()

    # Initialize ingester
    ingester = PolicyIngester()

    # Show current stats
    stats = ingester.get_stats()
    logger.info(f"Current ChromaDB stats: {stats}")

    # Ingest policies
    if args.policies_dir.exists():
        logger.info(f"Ingesting policies from: {args.policies_dir}")
        results = ingester.ingest_all_policies(args.policies_dir)

        print("\n" + "=" * 50)
        print("INGESTION RESULTS")
        print("=" * 50)
        for product_id, count in results.items():
            product_name = PRODUCT_ID_TO_NAME.get(product_id, "Unknown")
            print(f"  {product_id} ({product_name}): {count} chunks")
        print(f"\nTotal chunks: {sum(results.values())}")
    else:
        logger.warning(f"Policies directory not found: {args.policies_dir}")
        logger.info("Run 'python scripts/generate_policies.py' first to create sample policies")

    # Test query if specified
    if args.query:
        print("\n" + "=" * 50)
        print(f"TEST QUERY: {args.query}")
        print("=" * 50)
        results = ingester.query(args.query, n_results=3, product_id=args.product)
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} (distance: {result['distance']:.4f}) ---")
            print(f"Product: {result['metadata'].get('product_id')}")
            print(f"Section: {result['metadata'].get('section')}")
            print(f"Content: {result['content'][:300]}...")

    # Final stats
    stats = ingester.get_stats()
    print(f"\nFinal ChromaDB stats: {stats}")


if __name__ == "__main__":
    main()
