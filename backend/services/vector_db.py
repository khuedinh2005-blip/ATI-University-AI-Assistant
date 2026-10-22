"""
ChromaDB Vector Store Service.
Handles:
- Persistent vector collection for Hanoi University documents.
- Incremental updates using document ID and SHA-256 hash comparison.
- Querying top-K document chunks by cosine similarity.
- Recording sync history and metrics for the Admin panel.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings
from backend.services.ingestion import (
    process_announcement,
    fetch_portal_announcements
)

class VectorDBService:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection_name = "hanu_knowledge_base"
        
        # Get or create collection with cosine distance
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Track sync history logs in memory
        self.sync_logs: List[Dict[str, Any]] = []

    def get_document_stored_hash(self, doc_id: str) -> Optional[str]:
        """Retrieves the stored content hash for a doc_id if it exists."""
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                limit=1,
                include=["metadatas"]
            )
            if results and results.get("metadatas") and len(results["metadatas"]) > 0:
                return results["metadatas"][0].get("content_hash")
        except Exception:
            pass
        return None

    def sync_knowledge_base(self, announcements: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Executes incremental sync:
        - Fetches portal announcements.
        - Checks SHA-256 content hash against stored documents.
        - Skips unchanged announcements.
        - Deletes old chunks and inserts new chunks for added/modified documents.
        - Records run metrics.
        """
        if announcements is None:
            announcements = fetch_portal_announcements()

        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_fetched": len(announcements),
            "new": 0,
            "modified": 0,
            "skipped": 0,
            "errors": 0,
            "details": []
        }

        for raw_doc in announcements:
            doc_id = raw_doc.get("id")
            title = raw_doc.get("title", "Untitled")
            try:
                chunks, content_hash = process_announcement(raw_doc)
                existing_hash = self.get_document_stored_hash(doc_id)

                if existing_hash == content_hash:
                    stats["skipped"] += 1
                    stats["details"].append({
                        "doc_id": doc_id,
                        "title": title,
                        "action": "skipped (unchanged)"
                    })
                    continue

                action = "new" if existing_hash is None else "modified"
                if existing_hash is not None:
                    # Remove existing outdated chunks first
                    self.collection.delete(where={"doc_id": doc_id})

                # Upsert new chunks
                ids = [c["chunk_id"] for c in chunks]
                documents = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]

                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )

                if action == "new":
                    stats["new"] += 1
                else:
                    stats["modified"] += 1

                stats["details"].append({
                    "doc_id": doc_id,
                    "title": title,
                    "action": action,
                    "chunks_count": len(chunks)
                })

            except Exception as e:
                stats["errors"] += 1
                stats["details"].append({
                    "doc_id": doc_id,
                    "title": title,
                    "action": "error",
                    "error_message": str(e)
                })

        # Save to sync logs
        self.sync_logs.insert(0, stats)
        return stats

    def query_chunks(self, query_text: str, n_results: int = 20) -> List[Dict[str, Any]]:
        """
        Queries top-K chunks from ChromaDB using cosine similarity.
        Transforms distance into cosine similarity score (1 - distance).
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, max(1, self.collection.count()))
        )

        formatted_chunks = []
        if not results or not results.get("documents") or len(results["documents"]) == 0:
            return formatted_chunks

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)

        for doc_text, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: similarity = 1 - distance
            similarity = round(max(0.0, 1.0 - dist), 4)
            formatted_chunks.append({
                "text": doc_text,
                "metadata": meta,
                "similarity": similarity,
                "distance": dist
            })

        return formatted_chunks

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection count, last sync status, and storage metrics."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "total_syncs": len(self.sync_logs),
            "last_sync": self.sync_logs[0] if self.sync_logs else None
        }


# Singleton instance
vector_db_service = VectorDBService()
