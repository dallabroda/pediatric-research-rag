"""Tests for MCP tools module."""
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.query.retriever import RetrievalResult


# Sample retrieval results for testing
SAMPLE_RESULTS = [
    RetrievalResult(
        chunk_id="doc1_chunk0",
        document_id="PMC123",
        chunk_text="Pediatric leukemia treatment has advanced significantly with targeted therapies.",
        doc_title="Advances in Pediatric Leukemia",
        doc_type="paper",
        score=0.89,
        page_number=1,
        section_title="Introduction",
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
    ),
    RetrievalResult(
        chunk_id="doc1_chunk1",
        document_id="PMC123",
        chunk_text="CAR-T cell therapy shows promising results in refractory cases.",
        doc_title="Advances in Pediatric Leukemia",
        doc_type="paper",
        score=0.82,
        page_number=3,
        section_title="Results",
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
    ),
    RetrievalResult(
        chunk_id="doc2_chunk0",
        document_id="NCT001",
        chunk_text="This phase II trial evaluates immunotherapy in pediatric solid tumors.",
        doc_title="Immunotherapy for Pediatric Solid Tumors",
        doc_type="trial",
        score=0.75,
        page_number=None,
        section_title="Brief Summary",
        source_url="https://clinicaltrials.gov/study/NCT001",
    ),
]

SAMPLE_METADATA = [
    {
        "chunk_id": "doc1_chunk0",
        "document_id": "PMC123",
        "chunk_text": "Pediatric leukemia treatment content.",
        "doc_title": "Advances in Pediatric Leukemia",
        "doc_type": "paper",
        "chunk_index": 0,
        "page_number": 1,
        "section_title": "Introduction",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
    },
    {
        "chunk_id": "doc1_chunk1",
        "document_id": "PMC123",
        "chunk_text": "CAR-T cell therapy content.",
        "doc_title": "Advances in Pediatric Leukemia",
        "doc_type": "paper",
        "chunk_index": 1,
        "page_number": 3,
        "section_title": "Results",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
    },
    {
        "chunk_id": "doc2_chunk0",
        "document_id": "NCT001",
        "chunk_text": "Phase II trial content.",
        "doc_title": "Immunotherapy Trial",
        "doc_type": "trial",
        "chunk_index": 0,
        "page_number": None,
        "section_title": "Brief Summary",
        "source_url": "https://clinicaltrials.gov/study/NCT001",
    },
    {
        "chunk_id": "doc3_chunk0",
        "document_id": "NCT002",
        "chunk_text": "Another trial content.",
        "doc_title": "CAR-T Trial",
        "doc_type": "trial",
        "chunk_index": 0,
        "page_number": None,
        "section_title": "Brief Summary",
        "source_url": "https://clinicaltrials.gov/study/NCT002",
    },
]


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = MagicMock()
    retriever.search.return_value = SAMPLE_RESULTS
    retriever.metadata = SAMPLE_METADATA
    retriever.get_document_chunks.return_value = SAMPLE_METADATA[:2]  # PMC123 chunks
    retriever.get_document_ids.return_value = ["NCT001", "NCT002", "PMC123"]
    return retriever


@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client."""
    client = MagicMock()

    # Mock LLM response
    response_body = {
        "content": [{"text": "Based on the research, pediatric cancer treatment has evolved significantly."}]
    }
    client.invoke_model.return_value = {
        "body": MagicMock(read=lambda: json.dumps(response_body).encode())
    }

    return client


class TestSearchResearch:
    """Tests for search_research function."""

    @patch("mcp_server.tools.get_retriever")
    def test_search_returns_results(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import search_research

        results = search_research("pediatric cancer treatment", top_k=5)

        assert len(results) == 3
        mock_retriever.search.assert_called_once_with("pediatric cancer treatment", top_k=5)

    @patch("mcp_server.tools.get_retriever")
    def test_search_result_format(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import search_research

        results = search_research("leukemia")

        result = results[0]
        assert "document_id" in result
        assert "title" in result
        assert "doc_type" in result
        assert "text" in result
        assert "score" in result
        assert "source_url" in result
        assert "section" in result
        assert "page" in result

    @patch("mcp_server.tools.get_retriever")
    def test_search_score_rounded(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import search_research

        results = search_research("test")

        # Scores should be rounded to 3 decimal places
        for result in results:
            score_str = str(result["score"])
            if "." in score_str:
                decimals = len(score_str.split(".")[1])
                assert decimals <= 3

    @patch("mcp_server.tools.get_retriever")
    def test_search_empty_results(self, mock_get_retriever, mock_retriever):
        mock_retriever.search.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import search_research

        results = search_research("nonexistent topic")

        assert results == []


class TestAskResearchQuestion:
    """Tests for ask_research_question function."""

    @patch("mcp_server.tools.get_bedrock_client")
    @patch("mcp_server.tools.get_retriever")
    def test_ask_returns_answer(self, mock_get_retriever, mock_get_bedrock, mock_retriever, mock_bedrock_client):
        mock_get_retriever.return_value = mock_retriever
        mock_get_bedrock.return_value = mock_bedrock_client

        from mcp_server.tools import ask_research_question

        result = ask_research_question("What are the latest treatments?")

        assert "question" in result
        assert "answer" in result
        assert "sources" in result
        assert result["question"] == "What are the latest treatments?"

    @patch("mcp_server.tools.get_bedrock_client")
    @patch("mcp_server.tools.get_retriever")
    def test_ask_includes_sources(self, mock_get_retriever, mock_get_bedrock, mock_retriever, mock_bedrock_client):
        mock_get_retriever.return_value = mock_retriever
        mock_get_bedrock.return_value = mock_bedrock_client

        from mcp_server.tools import ask_research_question

        result = ask_research_question("Tell me about CAR-T therapy")

        assert len(result["sources"]) == 3
        source = result["sources"][0]
        assert "title" in source
        assert "doc_type" in source
        assert "source_url" in source
        assert "score" in source

    @patch("mcp_server.tools.get_bedrock_client")
    @patch("mcp_server.tools.get_retriever")
    def test_ask_calls_bedrock(self, mock_get_retriever, mock_get_bedrock, mock_retriever, mock_bedrock_client):
        mock_get_retriever.return_value = mock_retriever
        mock_get_bedrock.return_value = mock_bedrock_client

        from mcp_server.tools import ask_research_question

        ask_research_question("What is leukemia?")

        mock_bedrock_client.invoke_model.assert_called_once()
        call_args = mock_bedrock_client.invoke_model.call_args
        assert "body" in call_args.kwargs or len(call_args.args) > 0

    @patch("mcp_server.tools.get_retriever")
    def test_ask_no_context_response(self, mock_get_retriever, mock_retriever):
        mock_retriever.search.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import ask_research_question

        result = ask_research_question("Something with no results")

        assert result["sources"] == []
        assert result["answer"] != ""  # Should have a no-context response


class TestListClinicalTrials:
    """Tests for list_clinical_trials function."""

    @patch("mcp_server.tools.get_retriever")
    def test_list_trials_returns_trials(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import list_clinical_trials

        trials = list_clinical_trials()

        # Should return only trial doc_types
        assert len(trials) == 2  # NCT001 and NCT002
        for trial in trials:
            assert trial["nct_id"].startswith("NCT")

    @patch("mcp_server.tools.get_retriever")
    def test_list_trials_format(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import list_clinical_trials

        trials = list_clinical_trials()

        trial = trials[0]
        assert "nct_id" in trial
        assert "title" in trial
        assert "source_url" in trial

    @patch("mcp_server.tools.get_retriever")
    def test_list_trials_unique(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import list_clinical_trials

        trials = list_clinical_trials()

        nct_ids = [t["nct_id"] for t in trials]
        assert len(nct_ids) == len(set(nct_ids))  # All unique


class TestGetDocument:
    """Tests for get_document function."""

    @patch("mcp_server.tools.get_retriever")
    def test_get_document_found(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import get_document

        doc = get_document("PMC123")

        assert doc is not None
        assert doc["document_id"] == "PMC123"
        mock_retriever.get_document_chunks.assert_called_with("PMC123")

    @patch("mcp_server.tools.get_retriever")
    def test_get_document_format(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import get_document

        doc = get_document("PMC123")

        assert "document_id" in doc
        assert "title" in doc
        assert "doc_type" in doc
        assert "source_url" in doc
        assert "sections" in doc
        assert "chunks_count" in doc
        assert "full_text" in doc

    @patch("mcp_server.tools.get_retriever")
    def test_get_document_not_found(self, mock_get_retriever, mock_retriever):
        mock_retriever.get_document_chunks.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import get_document

        doc = get_document("NONEXISTENT")

        assert doc is None

    @patch("mcp_server.tools.get_retriever")
    def test_get_document_reconstructs_text(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import get_document

        doc = get_document("PMC123")

        # Full text should contain both chunks
        assert "leukemia treatment" in doc["full_text"]
        assert "CAR-T" in doc["full_text"]

    @patch("mcp_server.tools.get_retriever")
    def test_get_document_extracts_sections(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import get_document

        doc = get_document("PMC123")

        assert "Introduction" in doc["sections"]
        assert "Results" in doc["sections"]


class TestListDocuments:
    """Tests for list_documents function."""

    @patch("mcp_server.tools.get_retriever")
    def test_list_documents_returns_all(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import list_documents

        docs = list_documents()

        # Should have 3 unique documents
        assert len(docs) == 3

    @patch("mcp_server.tools.get_retriever")
    def test_list_documents_format(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import list_documents

        docs = list_documents()

        doc = docs[0]
        assert "document_id" in doc
        assert "title" in doc
        assert "doc_type" in doc
        assert "source_url" in doc

    @patch("mcp_server.tools.get_retriever")
    def test_list_documents_unique(self, mock_get_retriever, mock_retriever):
        mock_get_retriever.return_value = mock_retriever

        from mcp_server.tools import list_documents

        docs = list_documents()

        doc_ids = [d["document_id"] for d in docs]
        assert len(doc_ids) == len(set(doc_ids))


class TestRetrieverCaching:
    """Tests for retriever caching behavior."""

    @patch("mcp_server.tools._retriever", None)
    @patch("mcp_server.tools.FAISSRetriever")
    def test_get_retriever_loads_local_first(self, mock_faiss_retriever):
        mock_retriever = MagicMock()
        mock_faiss_retriever.from_local.return_value = mock_retriever

        from mcp_server.tools import get_retriever
        import mcp_server.tools as tools_module
        tools_module._retriever = None

        result = get_retriever()

        mock_faiss_retriever.from_local.assert_called()

    @patch("mcp_server.tools._retriever", None)
    @patch("mcp_server.tools.FAISSRetriever")
    def test_get_retriever_falls_back_to_s3(self, mock_faiss_retriever):
        mock_faiss_retriever.from_local.side_effect = FileNotFoundError("No local index")
        mock_retriever = MagicMock()
        mock_faiss_retriever.from_s3.return_value = mock_retriever

        from mcp_server.tools import get_retriever
        import mcp_server.tools as tools_module
        tools_module._retriever = None

        result = get_retriever()

        mock_faiss_retriever.from_s3.assert_called()


class TestBedrockClientCaching:
    """Tests for Bedrock client caching behavior."""

    @patch("mcp_server.tools._bedrock_client", None)
    @patch("boto3.client")
    def test_get_bedrock_client_creates_client(self, mock_boto_client):
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        from mcp_server.tools import get_bedrock_client
        import mcp_server.tools as tools_module
        tools_module._bedrock_client = None

        result = get_bedrock_client()

        mock_boto_client.assert_called_once_with(
            "bedrock-runtime",
            region_name="us-east-1",
        )
        assert result is mock_client
