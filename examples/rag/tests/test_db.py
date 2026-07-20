# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Download-free smoke test for the RAG search-plus-get workflow."""

from __future__ import annotations

from examples.rag import db as rag_db


def test_rag_database_round_trip_with_mock_embeddings(tmp_path, monkeypatch) -> None:
    rag_db.close_database()
    monkeypatch.setenv("ALAYALITE_RAG_DATA_DIR", str(tmp_path / "rag-db"))
    monkeypatch.setattr(rag_db, "split_text", lambda _text, _size, _overlap: ["alpha", "beta"])
    monkeypatch.setattr(
        rag_db,
        "embed_texts",
        lambda texts, _path: [[1.0, 0.0, 0.0] if text in {"alpha", "question"} else [0.0, 1.0, 0.0] for text in texts],
    )

    try:
        assert rag_db.insert_text("docs", "mock/bge", "source") is True
        assert rag_db.query_text("docs", "mock/bge", "question", top_k=2).split("\n\n")[0] == "alpha"
        rag_db.close_database()
        assert rag_db.open_database().list_collections() == ["docs"]
        rag_db.clear_database()
        assert rag_db.open_database().list_collections() == []
    finally:
        rag_db.close_database()
