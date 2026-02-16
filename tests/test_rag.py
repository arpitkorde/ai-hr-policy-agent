"""Tests for the RAG pipeline components."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.rag.ingest import DocumentIngestor, LOADER_MAP
from src.rag.prompts import get_prompt, list_prompt_versions, PROMPT_VERSIONS


# ============================================================
# Document Ingestor Tests
# ============================================================

class TestDocumentIngestor:
    """Tests for the DocumentIngestor class."""

    def setup_method(self):
        self.ingestor = DocumentIngestor(chunk_size=200, chunk_overlap=50)

    def test_init_default_settings(self):
        """Test ingestor initializes with config defaults."""
        ing = DocumentIngestor()
        assert ing.chunk_size > 0
        assert ing.chunk_overlap >= 0
        assert ing.chunk_overlap < ing.chunk_size

    def test_init_custom_settings(self):
        """Test ingestor accepts custom chunk parameters."""
        assert self.ingestor.chunk_size == 200
        assert self.ingestor.chunk_overlap == 50

    def test_supported_file_types(self):
        """Test that expected file types are supported."""
        assert ".pdf" in LOADER_MAP
        assert ".docx" in LOADER_MAP
        assert ".txt" in LOADER_MAP

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.ingestor.load_document("/nonexistent/file.pdf")

    def test_load_unsupported_type(self):
        """Test loading an unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(b"test")
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                self.ingestor.load_document(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_load_txt_file(self):
        """Test loading a plain text file."""
        content = "This is a sample HR policy document.\n\n" * 10
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = self.ingestor.load_document(tmp_path)
            assert len(docs) > 0
            assert docs[0].metadata["source"] == Path(tmp_path).name
            assert docs[0].metadata["file_type"] == ".txt"
        finally:
            os.unlink(tmp_path)

    def test_split_documents(self):
        """Test document splitting produces chunks."""
        from langchain.schema import Document

        long_text = "This is a test sentence. " * 100
        docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]
        chunks = self.ingestor.split_documents(docs)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= self.ingestor.chunk_size + 50  # allow slight over

    def test_ingest_txt_file(self):
        """Test full ingestion pipeline for a text file."""
        content = "Company Leave Policy.\n\n" + "Employees get 20 days of PTO. " * 50
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunks = self.ingestor.ingest(tmp_path)
            assert len(chunks) > 0
            assert all(c.metadata.get("source") for c in chunks)
        finally:
            os.unlink(tmp_path)

    def test_load_directory_nonexistent(self):
        """Test loading from a nonexistent directory."""
        with pytest.raises(NotADirectoryError):
            self.ingestor.load_directory("/nonexistent/dir")


# ============================================================
# Prompt Tests
# ============================================================

class TestPrompts:
    """Tests for the prompt versioning system."""

    def test_list_versions(self):
        """Test that prompt versions are registered."""
        versions = list_prompt_versions()
        assert "v1.0" in versions
        assert "v2.0" in versions

    def test_get_valid_prompt(self):
        """Test getting a registered prompt version."""
        prompt = get_prompt("v1.0")
        assert prompt is not None

    def test_get_invalid_prompt(self):
        """Test getting a non-existent prompt version."""
        with pytest.raises(KeyError):
            get_prompt("v99.0")

    def test_prompt_has_required_variables(self):
        """Test that prompts accept context and question variables."""
        for version in list_prompt_versions():
            prompt = get_prompt(version)
            # Should be able to format with context and question
            messages = prompt.format_messages(
                context="Sample context", question="Sample question"
            )
            assert len(messages) > 0

    def test_prompt_registry_populated(self):
        """Test that the prompt registry has entries."""
        assert len(PROMPT_VERSIONS) >= 2
