"""Offline router contracts. No model prediction, server, ingestion or live A/B."""

import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableLambda

from agentic_rag.config import Settings
from agentic_rag.graph import chains


@pytest.fixture(autouse=True)
def clear_router_caches():
    chains.get_router.cache_clear()
    chains._get_laya_agent.cache_clear()
    yield
    chains.get_router.cache_clear()
    chains._get_laya_agent.cache_clear()


def test_configuration_defaults_and_environment(monkeypatch):
    for name in ("ROUTER_BACKEND", "LAYA_MODEL", "LAYA_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    assert Settings().router_backend == "llm"
    assert Settings().laya_model == "convaiinnovations/laya"
    assert Settings().laya_device == "cpu"
    monkeypatch.setenv("ROUTER_BACKEND", "laya")
    monkeypatch.setenv("LAYA_MODEL", "/local/checkpoint")
    monkeypatch.setenv("LAYA_DEVICE", "mps")
    configured = Settings()
    assert (configured.router_backend, configured.laya_model, configured.laya_device) == (
        "laya", "/local/checkpoint", "mps",
    )


@pytest.mark.parametrize("datasource", ["vectorstore", "web_search", "sql"])
def test_llm_baseline_prompt_schema_and_no_laya_dependency(datasource, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "laya", None)
    monkeypatch.setattr(chains, "settings", replace(chains.settings, router_backend="llm"))
    prompts = []

    def respond(prompt):
        prompts.append(prompt.to_messages())
        return chains.RouteQuery(datasource=datasource)

    llm = Mock()
    llm.with_structured_output.return_value = RunnableLambda(respond)
    monkeypatch.setattr(chains, "get_llm", lambda: llm)
    question = "Original question? 保留原文"
    with caplog.at_level(logging.INFO, logger=chains.__name__):
        router = chains.get_router()
        result = router.invoke({"question": question})
    assert chains.get_router() is router
    assert isinstance(result, chains.RouteQuery)
    assert result.datasource == datasource
    llm.with_structured_output.assert_called_once_with(chains.RouteQuery)
    # Exact original rendered system/human messages, including punctuation/order.
    bound = chains.ROUTER_PROMPT.partial_variables
    assert [(message.type, message.content) for message in prompts[0]] == [
        ("system", "You are an expert at routing a user question to one data source:\n"
         f'- "vectorstore": {bound["kb_description"]}\n'
         f'- "sql": {bound["sql_description"]}\n'
         '- "web_search": current events, recent facts, or anything not '
         'covered by the other two sources.\n'
         'Pick the single best datasource for the question.'),
        ("human", question),
    ]
    assert f"Initial router decision: backend=llm datasource={datasource}" in caplog.text


def test_invalid_backend_is_observable_without_loading_models(monkeypatch):
    monkeypatch.setattr(chains, "settings", replace(chains.settings, router_backend="typo"))
    llm, laya = Mock(), Mock()
    monkeypatch.setattr(chains, "get_llm", llm)
    monkeypatch.setattr(chains, "_get_laya_agent", laya)
    with pytest.raises(ValueError, match="Unknown ROUTER_BACKEND"):
        chains.get_router()
    llm.assert_not_called()
    laya.assert_not_called()


def test_laya_load_is_lazy_cached_and_uses_configuration(monkeypatch):
    module = ModuleType("laya")
    module.load = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "laya", module)
    monkeypatch.setattr(chains, "settings", replace(
        chains.settings, router_backend="laya", laya_model="local-model", laya_device="cpu",
    ))
    chains.get_router()
    module.load.assert_not_called()
    assert chains._get_laya_agent() is chains._get_laya_agent()
    module.load.assert_called_once_with("local-model", device="cpu")


@pytest.mark.parametrize("datasource", ["vectorstore", "web_search", "sql"])
def test_laya_normalizes_choice_and_receives_equivalent_sources(datasource, monkeypatch, caplog):
    agent = _mock_laya(monkeypatch, {"answers": {"datasource": {"choice": datasource}}})
    validator = Mock()
    monkeypatch.setattr(chains, "_validate_laya_input", validator)
    question = "Unmodified question? 保留原文"
    with caplog.at_level(logging.INFO, logger=chains.__name__):
        result = chains.get_router().invoke({"question": question})
    assert isinstance(result, chains.RouteQuery)
    assert result.datasource == datasource
    state, definitions = agent.predict.call_args.args
    assert state == {
        "question": question,
        "sources": {
            "vectorstore": chains.ROUTER_PROMPT.partial_variables["kb_description"],
            "sql": chains.ROUTER_PROMPT.partial_variables["sql_description"],
            "web_search": chains.WEB_DESCRIPTION,
        },
    }
    assert definitions is chains.LAYA_ROUTER_QUESTIONS
    assert set(definitions["datasource"]["criteria"]) == {"vectorstore", "sql", "web_search"}
    validator.assert_called_once_with(agent, state)
    agent.predict.assert_called_once()
    assert f"Initial router decision: backend=laya datasource={datasource}" in caplog.text


@pytest.mark.parametrize("response", [
    None, {}, {"answers": None}, {"answers": {"datasource": {}}},
    {"answers": {"datasource": {"choice": "memory"}}},
    {"answers": {"datasource": {"choice": None}}},
])
def test_invalid_laya_results_raise_without_fallback(response, monkeypatch, caplog):
    agent = _mock_laya(monkeypatch, response)
    with caplog.at_level(logging.INFO, logger=chains.__name__):
        with pytest.raises(ValueError):
            chains.get_router().invoke({"question": "Question"})
    agent.predict.assert_called_once()
    assert "Initial router decision" not in caplog.text


@pytest.mark.parametrize("backend", ["llm", "laya"])
def test_prediction_errors_propagate_once(backend, monkeypatch):
    error = RuntimeError("offline backend failure")
    if backend == "laya":
        agent = _mock_laya(monkeypatch, None)
        agent.predict.side_effect = error
        prediction = agent.predict
    else:
        monkeypatch.setattr(chains, "settings", replace(chains.settings, router_backend="llm"))
        prediction = Mock(side_effect=error)
        llm = Mock()
        llm.with_structured_output.return_value = RunnableLambda(prediction)
        monkeypatch.setattr(chains, "get_llm", lambda: llm)
    with pytest.raises(RuntimeError, match="offline backend failure") as raised:
        chains.get_router().invoke({"question": "Question"})
    assert raised.value is error
    prediction.assert_called_once()


def test_missing_laya_dependency_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(chains, "settings", replace(chains.settings, router_backend="laya"))
    monkeypatch.setitem(sys.modules, "laya", None)
    llm = Mock()
    monkeypatch.setattr(chains, "get_llm", llm)
    with pytest.raises(ModuleNotFoundError):
        chains.get_router().invoke({"question": "Question"})
    llm.assert_not_called()


def test_encoding_failure_stops_before_prediction(monkeypatch):
    agent = _mock_laya(monkeypatch, None)
    monkeypatch.setattr(chains, "_validate_laya_input", Mock(side_effect=ValueError("would truncate")))
    with pytest.raises(ValueError, match="would truncate"):
        chains.get_router().invoke({"question": "Question"})
    agent.predict.assert_not_called()


@pytest.mark.parametrize("truncated,markers", [(False, [1, 2, 3]), (True, [1, 2, 3]), (False, [1])])
def test_encoding_guard_checks_complete_token_tail(truncated, markers, monkeypatch):
    common = ModuleType("laya.common")
    common.serialize_state = json.dumps
    class Tokenizer:
        mask_token = "[MASK]"
        cls_token_id = 1
        sep_token_id = 2
        mask_token_id = 3

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(char) + 100 for char in text]}

    def encode(tok, state, question, max_len, head_max_len, state_ids):
        head = [1] + tok(f"choice question: {question['ins']}")["input_ids"] + [2]
        for key, criterion in question["crit"].items():
            head += [3] + tok(f" {key}: {criterion}")["input_ids"]
        effective = state_ids[:-1] if truncated else state_ids
        return head + [2] + effective + [2], markers

    common.render_options = lambda question: [
        f"{key}: {criterion}" for key, criterion in question["crit"].items()
    ]

    common.build_sequence = Mock(side_effect=encode)
    monkeypatch.setitem(sys.modules, "laya.common", common)
    agent = SimpleNamespace(tok=Tokenizer(), cfg={})
    if truncated or len(markers) != 3:
        with pytest.raises(ValueError, match="would truncate"):
            chains._validate_laya_input(agent, {"question": "q", "sources": {"sql": "complete"}})
    else:
        chains._validate_laya_input(agent, {"question": "q", "sources": {"sql": "complete"}})
    assert common.build_sequence.call_args.args[3:5] == (512, 192)
    with pytest.raises(ValueError, match="mask token"):
        chains._validate_laya_input(agent, {"question": "literal [MASK]"})


@pytest.mark.skipif(not os.getenv("LAYA_LOCAL_SNAPSHOT"), reason="requires an explicitly provided local tokenizer; no downloads")
def test_real_local_encoder_preserves_sources_and_rejects_truncation(monkeypatch):
    """Tokenizer-only verification with the installed Laya encoder, never model inference."""
    from laya.agent import Agent
    from laya.common import build_sequence, serialize_state
    from transformers import AutoTokenizer

    snapshot = Path(os.environ["LAYA_LOCAL_SNAPSHOT"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    config = json.loads((snapshot / "rl_agent_config.json").read_text())
    agent = _mock_laya(monkeypatch, {"answers": {"datasource": {"choice": "sql"}}}, validate_input=True)
    agent.tok, agent.cfg = tokenizer, config
    question = "Which product had the most sales in Spain?"
    assert chains.get_router().invoke({"question": question}).datasource == "sql"
    state, definitions = agent.predict.call_args.args
    definition = definitions["datasource"]
    q = {"t": "choice", "ins": definition["instructions"], "crit": definition["criteria"]}
    ids, markers = build_sequence(tokenizer, state, q, config["max_len"], config["head_max_len"])
    # Exercise Agent's encoding path too, without __init__, weights or inference.
    encoder = Agent.__new__(Agent)
    encoder.tok, encoder.cfg = tokenizer, config
    encoded = encoder._encode_state(state, ["datasource"], {"datasource": q})[0]
    assert encoded["ids"] == ids
    assert encoded["markers"] == markers
    full_state_ids = tokenizer(serialize_state(state), add_special_tokens=False)["input_ids"]
    assert ids[-len(full_state_ids) - 1:] == full_state_ids + [tokenizer.sep_token_id]
    assert len(markers) == 3
    for key, criterion in definition["criteria"].items():
        assert len(tokenizer(f" {key}: {criterion}", add_special_tokens=False)["input_ids"]) <= 48
    for description in state["sources"].values():
        assert description in chains.ROUTER_PROMPT.format_messages(question=question)[0].content
    print(f"Local Laya encoding: state_tokens={len(full_state_ids)}, encoded_tokens={len(ids)}, max_len={config['max_len']}")
    agent.predict.reset_mock()
    with pytest.raises(ValueError, match="would truncate"):
        chains.get_router().invoke({"question": "very long question " * 1000})
    agent.predict.assert_not_called()
    agent.cfg = {**config, "head_max_len": 16}
    with pytest.raises(ValueError, match="would truncate"):
        chains.get_router().invoke({"question": question})
    agent.predict.assert_not_called()
    agent.cfg = config
    expanded_prompt = chains.ROUTER_PROMPT.partial(sql_description="complete source description " * 1000)
    monkeypatch.setattr(chains, "ROUTER_PROMPT", expanded_prompt)
    with pytest.raises(ValueError, match="would truncate"):
        chains.get_router().invoke({"question": question})
    agent.predict.assert_not_called()


def _mock_laya(monkeypatch, response, validate_input=False):
    monkeypatch.setattr(chains, "settings", replace(chains.settings, router_backend="laya"))
    monkeypatch.setattr(chains, "get_llm", Mock(side_effect=AssertionError("LLM fallback is forbidden")))
    agent = SimpleNamespace(predict=Mock(return_value=response))
    monkeypatch.setattr(chains, "_get_laya_agent", lambda: agent)
    if not validate_input:
        monkeypatch.setattr(chains, "_validate_laya_input", lambda agent, state: None)
    return agent
