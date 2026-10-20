"""The graph must compile without a running Ollama server or existing index."""

import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from agentic_rag.graph import chains, nodes
from agentic_rag.graph.build import build_graph

EXPECTED_NODES = {
    "route",
    "retrieve",
    "grade_documents",
    "rewrite_query",
    "web_search",
    "query_sql",
    "generate",
}


def test_graph_compiles_with_expected_nodes():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    assert EXPECTED_NODES <= nodes


@pytest.mark.parametrize("backend", ["llm", "laya"])
@pytest.mark.parametrize("datasource", ["vectorstore", "web_search", "sql"])
def test_mocked_router_runs_original_graph(backend, datasource, monkeypatch, caplog):
    _configure_mock_sources(monkeypatch, backend, datasource)
    try:
        with caplog.at_level(logging.INFO, logger=chains.__name__):
            updates = list(build_graph().stream({"question": "Original question"}, stream_mode="updates"))
        route_update = next(update["route"] for update in updates if "route" in update)
        assert route_update == {
            "datasource": datasource, "original_question": "Original question", "retries": 0,
        }
        visited = [name for update in updates for name in update]
        source_node = {"vectorstore": "retrieve", "web_search": "web_search", "sql": "query_sql"}[datasource]
        assert visited[0:2] == ["route", source_node]
        assert visited.count("route") == 1
        assert ("grade_documents" in visited) == (datasource == "vectorstore")
        assert visited[-1] == "generate"
        assert f"Initial router decision: backend={backend} datasource={datasource}" in caplog.text
    finally:
        chains.get_router.cache_clear()


@pytest.mark.parametrize("backend", ["llm", "laya"])
def test_initial_decision_survives_web_fallback_outside_state(backend, monkeypatch, caplog):
    _configure_mock_sources(monkeypatch, backend, "vectorstore", documents=[])
    monkeypatch.setattr(nodes, "settings", replace(nodes.settings, max_retries=1))
    try:
        with caplog.at_level(logging.INFO, logger=chains.__name__):
            updates = list(build_graph().stream({"question": "Original question"}, stream_mode="updates"))
        # The observer captures the route event independently; the final state can change.
        initial_route = next(dict(update["route"]) for update in updates if "route" in update)
        final_state = {"question": "Original question"}
        for update in updates:
            for values in update.values():
                final_state.update(values)
        assert initial_route["datasource"] == "vectorstore"
        assert final_state["datasource"] == "web_search"
        assert final_state["original_question"] == "Original question"
        assert final_state["retries"] == 1
        assert sum("route" in update for update in updates) == 1
        assert sum("rewrite_query" in update for update in updates) == 1
        assert sum("retrieve" in update for update in updates) == 2
        records = [record for record in caplog.records if record.name == chains.__name__]
        assert [record.getMessage() for record in records] == [
            f"Initial router decision: backend={backend} datasource=vectorstore"
        ]
    finally:
        chains.get_router.cache_clear()


def _configure_mock_sources(monkeypatch, backend, datasource, documents=None):
    """Replace external I/O only; execute the original nodes and conditional edges."""
    chains.get_router.cache_clear()
    monkeypatch.setattr(chains, "settings", replace(chains.settings, router_backend=backend))
    llm = Mock()
    llm.with_structured_output.return_value = RunnableLambda(
        lambda _: chains.RouteQuery(datasource=datasource)
    )
    monkeypatch.setattr(chains, "get_llm", lambda: llm)
    agent = SimpleNamespace(predict=Mock(return_value={"answers": {"datasource": {"choice": datasource}}}))
    monkeypatch.setattr(chains, "_get_laya_agent", lambda: agent)
    # Encoder behavior is tested separately, including with the real local tokenizer.
    monkeypatch.setattr(chains, "_validate_laya_input", lambda agent, state: None)
    if documents is None:
        documents = [Document(page_content="Evidence", metadata={"source": "mock"})]
    monkeypatch.setattr(nodes, "get_retriever", lambda: RunnableLambda(lambda _: documents))
    monkeypatch.setattr(nodes, "get_document_grader", lambda: RunnableLambda(lambda _: "yes"))
    monkeypatch.setattr(nodes, "get_grounding_grader", lambda: RunnableLambda(lambda _: "yes"))
    monkeypatch.setattr(nodes, "get_generator", lambda: RunnableLambda(lambda _: "Answer"))
    monkeypatch.setattr(nodes, "get_rewriter", lambda: RunnableLambda(lambda _: "Rewritten question"))
    monkeypatch.setattr(nodes, "search_web", lambda _: [Document(page_content="Web evidence")])
    monkeypatch.setattr(nodes, "get_sql_llm", lambda: object())
    monkeypatch.setattr(nodes, "run_sql_question", lambda question, llm: Document(page_content="SQL evidence"))
