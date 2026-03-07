"""
KnowledgeTree and ExperienceTree: DuckDB-based vector stores for agent knowledge and experience.
"""

import json
import logging
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from typing_extensions import NotRequired

import duckdb
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

_VALID_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")


def _validate_identifier(name: str) -> str:
    if not _VALID_IDENTIFIER.match(name):
        raise ValueError(
            f"Invalid identifier '{name}': must match [a-zA-Z_][a-zA-Z0-9_]{{0,63}}"
        )
    return name

DEFAULT_VECTOR_DB_PATH = "mao_vectors.duckdb"
BATCH_SIZE: int = int(os.environ.get("VECTOR_BATCH_SIZE", "32"))
_DUCKDB_CONNECTIONS: dict[str, duckdb.DuckDBPyConnection] = {}


def get_vector_db_path() -> str:
    return os.environ.get("VECTOR_DB_PATH", DEFAULT_VECTOR_DB_PATH)


class SearchResult(TypedDict):
    id: str
    score: float
    page_content: str
    tags: list[str]
    relations: NotRequired[list[dict[str, Any]]]


class EmbeddingProvider:
    @staticmethod
    async def create_embeddings() -> tuple[Embeddings, int]:
        hf_embed: Embeddings = HuggingFaceEmbeddings(
            model="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
        )
        if hasattr(hf_embed, "embedding_size") and hf_embed.embedding_size:
            embed_dim = int(hf_embed.embedding_size)
        elif hasattr(hf_embed, "dim") and hf_embed.dim:
            embed_dim = int(hf_embed.dim)
        else:
            embed_dim = len(hf_embed.embed_query("test"))
        logging.info(
            f"Using SentenceTransformers embeddings: BAAI/bge-small-en-v1.5 with dim {embed_dim}"
        )
        return hf_embed, embed_dim


class VectorStoreError(Exception):
    pass


class VectorStoreBase:
    def __init__(
        self,
        db_path: str | None = None,
        collection_name: str = "default_collection",
        recreate_on_dim_mismatch: bool = False,
        embedding_provider: (
            Callable[[], Awaitable[tuple[Embeddings, int]]] | None
        ) = None,
    ):
        self.collection_name = _validate_identifier(collection_name)
        self.db_path = db_path or get_vector_db_path()
        self.recreate_on_dim_mismatch = recreate_on_dim_mismatch
        self.conn = self._get_connection(self.db_path)

        self.embed: Embeddings | None = None
        self.embed_dim: int | None = None
        self._embedding_provider = (
            embedding_provider or EmbeddingProvider.create_embeddings
        )

    @staticmethod
    def _get_connection(db_path: str) -> duckdb.DuckDBPyConnection:
        normalized_path = ":memory:" if db_path == ":memory:" else os.path.abspath(db_path)
        if normalized_path not in _DUCKDB_CONNECTIONS:
            _DUCKDB_CONNECTIONS[normalized_path] = duckdb.connect(normalized_path)
        return _DUCKDB_CONNECTIONS[normalized_path]

    async def async_init(self) -> "VectorStoreBase":
        self.embed, self.embed_dim = await self._embedding_provider()
        self._ensure_collection()
        logging.info(
            f"{self.__class__.__name__}: dim {self.embed_dim} "
            f"for '{self.collection_name}' at {self.db_path}"
        )
        return self

    @classmethod
    async def create(
        cls,
        db_path: str | None = None,
        collection_name: str = "default_collection",
        recreate_on_dim_mismatch: bool = False,
        embedding_provider: (
            Callable[[], Awaitable[tuple[Embeddings, int]]] | None
        ) = None,
    ) -> "VectorStoreBase":
        instance = cls(
            db_path, collection_name, recreate_on_dim_mismatch, embedding_provider
        )
        return await instance.async_init()

    def _ensure_collection(self) -> None:
        table = self.collection_name
        try:
            existing = self.conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = ?",
                [table],
            ).fetchone()

            if existing and self.recreate_on_dim_mismatch and self.embed_dim:
                col_info = self.conn.execute(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name = ? AND column_name = 'embedding'",
                    [table],
                ).fetchone()
                if col_info and f"[{self.embed_dim}]" not in str(col_info[0]):
                    logging.warning(
                        f"Dimension mismatch for '{table}'. Recreating."
                    )
                    self.conn.execute(f"DROP TABLE {table}")
                    existing = None

            if not existing:
                self.conn.execute(f"""
                    CREATE TABLE {table} (
                        id VARCHAR PRIMARY KEY,
                        text VARCHAR,
                        tags JSON,
                        relations JSON,
                        embedding FLOAT[{self.embed_dim}]
                    )
                """)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to ensure collection '{table}': {e}"
            ) from e

    def _parse_json(self, val: Any) -> Any:
        if isinstance(val, str):
            return json.loads(val)
        return val if val is not None else []

    async def add_entry_async(self, text: str, tags: list[str] | None = None) -> str:
        if self.embed is None or self.embed_dim is None:
            raise RuntimeError("Embeddings not initialized. Call async_init() first.")
        point_id = str(uuid.uuid4())
        vector = self.embed.embed_query(text)
        try:
            self.conn.execute(
                f"INSERT INTO {self.collection_name} VALUES (?, ?, ?, ?, ?)",
                [point_id, text, json.dumps(tags or []), json.dumps([]), vector],
            )
            return point_id
        except Exception as e:
            raise VectorStoreError(f"Failed to add entry: {e}") from e

    async def search_async(self, query: str, k: int = 3) -> list[SearchResult]:
        if self.embed is None:
            raise RuntimeError("Embeddings not initialized. Call async_init() first.")
        try:
            vector = self.embed.embed_query(query)
            rows = self.conn.execute(
                f"""
                SELECT id, text, tags, relations,
                       array_cosine_similarity(embedding, ?::FLOAT[{self.embed_dim}]) as score
                FROM {self.collection_name}
                ORDER BY score DESC
                LIMIT ?
                """,
                [vector, k],
            ).fetchall()

            return [
                SearchResult(
                    id=row[0],
                    score=row[4] if row[4] is not None else 0.0,
                    page_content=row[1] or "",
                    tags=self._parse_json(row[2]),
                    relations=self._parse_json(row[3]),
                )
                for row in rows
            ]
        except Exception as e:
            logging.error(f"Search failed in '{self.collection_name}': {e}")
            return []

    async def delete_entry_async(self, point_id: str) -> bool:
        try:
            self.conn.execute(
                f"DELETE FROM {self.collection_name} WHERE id = ?", [point_id]
            )
            return True
        except Exception as e:
            logging.error(f"Failed to delete entry {point_id}: {e}")
            return False

    async def get_entry_async(self, point_id: str) -> dict[str, Any] | None:
        try:
            row = self.conn.execute(
                f"SELECT id, text, tags, relations FROM {self.collection_name} WHERE id = ?",
                [point_id],
            ).fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "page_content": row[1] or "",
                "text": row[1] or "",
                "tags": self._parse_json(row[2]),
                "relations": self._parse_json(row[3]),
            }
        except Exception as e:
            logging.error(f"Error retrieving entry {point_id}: {e}")
            return None

    async def clear_all_points_async(self) -> None:
        self.conn.execute(f"DELETE FROM {self.collection_name}")

    async def add_tag_async(self, point_id: str, tag: str) -> bool:
        entry = await self.get_entry_async(point_id)
        if not entry:
            return False
        tags = set(entry.get("tags", []))
        tags.add(tag)
        try:
            self.conn.execute(
                f"UPDATE {self.collection_name} SET tags = ? WHERE id = ?",
                [json.dumps(list(tags)), point_id],
            )
            return True
        except Exception as e:
            logging.error(f"Failed to add tag to {point_id}: {e}")
            return False

    async def get_tags_async(self, point_id: str) -> list[str]:
        entry = await self.get_entry_async(point_id)
        return entry.get("tags", []) if entry else []

    async def add_relation_async(
        self, from_id: str, to_id: str, rel_type: str = "related"
    ) -> bool:
        entry = await self.get_entry_async(from_id)
        if not entry:
            return False
        rels = entry.get("relations", [])
        if not any(r.get("id") == to_id and r.get("type") == rel_type for r in rels):
            rels.append({"id": to_id, "type": rel_type})
            try:
                self.conn.execute(
                    f"UPDATE {self.collection_name} SET relations = ? WHERE id = ?",
                    [json.dumps(rels), from_id],
                )
                return True
            except Exception as e:
                logging.error(f"Failed to add relation {from_id} -> {to_id}: {e}")
                return False
        return True

    async def get_relations_async(
        self, point_id: str, rel_type: str | None = None
    ) -> list[dict[str, Any]]:
        entry = await self.get_entry_async(point_id)
        if not entry:
            return []
        rels = entry.get("relations", [])
        if rel_type:
            return [r for r in rels if r.get("type") == rel_type]
        return rels

    async def remove_relation_async(
        self, from_id: str, to_id: str, rel_type: str | None = None
    ) -> bool:
        entry = await self.get_entry_async(from_id)
        if not entry:
            return False
        rels = entry.get("relations", [])
        new_rels = [
            r
            for r in rels
            if not (
                r.get("id") == to_id
                and (rel_type is None or r.get("type") == rel_type)
            )
        ]
        if len(new_rels) < len(rels):
            try:
                self.conn.execute(
                    f"UPDATE {self.collection_name} SET relations = ? WHERE id = ?",
                    [json.dumps(new_rels), from_id],
                )
                return True
            except Exception as e:
                logging.error(f"Failed to remove relation {from_id} -> {to_id}: {e}")
                return False
        return True

    async def add_entries_batch_async(
        self, texts: list[str], tags_list: list[list[str]] | None = None
    ) -> list[str]:
        if not texts:
            return []
        if tags_list and len(texts) != len(tags_list):
            raise ValueError(
                f"texts ({len(texts)}) and tags_list ({len(tags_list)}) must have same length"
            )
        if self.embed is None:
            raise RuntimeError("Embeddings not initialized. Call async_init() first.")

        point_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        try:
            vectors = list(self.embed.embed_documents(texts))
            for i in range(0, len(texts), BATCH_SIZE):
                for j in range(i, min(i + BATCH_SIZE, len(texts))):
                    tags = tags_list[j] if tags_list else []
                    self.conn.execute(
                        f"INSERT INTO {self.collection_name} VALUES (?, ?, ?, ?, ?)",
                        [point_ids[j], texts[j], json.dumps(tags), json.dumps([]), vectors[j]],
                    )
            return point_ids
        except Exception as e:
            raise VectorStoreError(f"Failed to add entries in batch: {e}") from e

    async def traverse_async(
        self, start_id: str, depth: int = 1, rel_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if depth <= 0:
            return []
        visited_ids: set[str] = set()
        queue: list[tuple[str, int]] = [(start_id, 0)]
        result_nodes: list[dict[str, Any]] = []
        processed_edges: set[tuple[str, str, str | None]] = set()
        head = 0
        while head < len(queue):
            current_id, d = queue[head]
            head += 1
            if d > depth:
                continue
            if current_id not in visited_ids:
                visited_ids.add(current_id)
                current_entry = await self.get_entry_async(current_id)
                if not current_entry:
                    continue
                current_entry["depth"] = d
                result_nodes.append(current_entry)
            if d < depth:
                relations = await self.get_relations_async(current_id)
                for rel in relations:
                    target_id = rel.get("id")
                    relation_type = rel.get("type")
                    if not target_id:
                        continue
                    if rel_types is not None and relation_type not in rel_types:
                        continue
                    edge = (current_id, target_id, relation_type)
                    if edge in processed_edges:
                        continue
                    processed_edges.add(edge)
                    if target_id not in visited_ids:
                        queue.append((target_id, d + 1))
        return result_nodes

    async def summarize_entry_async(self, entry_id: str) -> str:
        entry = await self.get_entry_async(entry_id)
        return entry.get("text", "") if entry else ""


class KnowledgeTree(VectorStoreBase):
    def __init__(
        self,
        db_path: str | None = None,
        collection_name: str = "knowledge_tree",
        recreate_on_dim_mismatch: bool = False,
    ):
        super().__init__(
            db_path=db_path,
            collection_name=collection_name,
            recreate_on_dim_mismatch=recreate_on_dim_mismatch,
        )

    @classmethod
    async def create(
        cls,
        db_path: str | None = None,
        collection_name: str = "knowledge_tree",
        recreate_on_dim_mismatch: bool = False,
        embedding_provider: (
            Callable[[], Awaitable[tuple[Embeddings, int]]] | None
        ) = None,
    ) -> "KnowledgeTree":
        instance = cls(db_path, collection_name, recreate_on_dim_mismatch)
        instance._embedding_provider = (
            embedding_provider or EmbeddingProvider.create_embeddings
        )
        await instance.async_init()
        return instance

    async def learn_from_experience_async(
        self,
        text: str,
        related_knowledge_id: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        exp_id = await self.add_entry_async(text, tags)
        if related_knowledge_id:
            knowledge_entry = await self.get_entry_async(related_knowledge_id)
            if knowledge_entry:
                await self.add_relation_async(
                    exp_id, related_knowledge_id, rel_type="knowledge"
                )
            else:
                logging.warning(
                    f"Could not link experience {exp_id} to non-existent "
                    f"knowledge entry {related_knowledge_id}"
                )
        return exp_id

    async def learn_from_entry_async(self, entry_id: str, new_text: str) -> str:
        original_entry = await self.get_entry_async(entry_id)
        if not original_entry:
            logging.warning(
                f"Cannot learn from non-existent entry {entry_id}. "
                f"Creating new entry directly."
            )
            return await self.add_entry_async(new_text)
        new_id = await self.add_entry_async(new_text)
        await self.add_relation_async(entry_id, new_id, rel_type="learned_from_this")
        await self.add_relation_async(new_id, entry_id, rel_type="learned_this_from")
        return new_id


class ExperienceTree(KnowledgeTree):
    def __init__(
        self,
        db_path: str | None = None,
        collection_name: str = "experience_tree",
        recreate_on_dim_mismatch: bool = False,
    ):
        super().__init__(
            db_path=db_path,
            collection_name=collection_name,
            recreate_on_dim_mismatch=recreate_on_dim_mismatch,
        )

    @classmethod
    async def create(
        cls,
        db_path: str | None = None,
        collection_name: str = "experience_tree",
        recreate_on_dim_mismatch: bool = False,
        embedding_provider: (
            Callable[[], Awaitable[tuple[Embeddings, int]]] | None
        ) = None,
    ) -> "ExperienceTree":
        instance = cls(db_path, collection_name, recreate_on_dim_mismatch)
        instance._embedding_provider = (
            embedding_provider or EmbeddingProvider.create_embeddings
        )
        await instance.async_init()
        return instance
