"""
test.py — TemplateCreation Pipeline Integration Tests
======================================================

Tests covered
-------------
1.  Graph build           — build_graph() returns a compiled graph object.
2.  Phase 1 single turn   — chat_template() with an unsatisfied chatbot returns
                            an AI response without triggering Phase 2.
3.  Phase 2 auto-trigger  — when the chatbot sets satisfied=True, chat_template()
                            automatically calls create_template() (Phase 2).
4.  generate_tool call    — after Phase 2, generate_tool(template_id) is called
                            automatically from chat_template().
5.  create_template guard — chat_template() returns an "already finalized" message
                            on a second call once the template is persisted.
6.  route_after_chatbot   — routing function returns correct strings based on state.
7.  State shape           — GraphState fields are present and correctly typed.

Run
---
    # from the project root (AdapterAI/)
    python -m pytest TemplateCreation/test.py -v

Or directly:
    python TemplateCreation/test.py
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so all absolute imports resolve
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # AdapterAI/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import real langchain_core.messages first (it IS installed)
# ---------------------------------------------------------------------------
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

# ---------------------------------------------------------------------------
# Stub out heavy / external dependencies BEFORE any project imports so that
# the test suite can run without live services (Postgres, Redis, Groq).
# ---------------------------------------------------------------------------

# ── psycopg2 stub ────────────────────────────────────────────────────────────
_psycopg2_stub = types.ModuleType("psycopg2")
_psycopg2_stub.extensions = types.SimpleNamespace(connection=object)
_psycopg2_stub.extras = types.SimpleNamespace(RealDictCursor=None)
_psycopg2_pool_stub = types.ModuleType("psycopg2.pool")
_psycopg2_pool_stub.ThreadedConnectionPool = MagicMock()
_psycopg2_stub.pool = _psycopg2_pool_stub
sys.modules.setdefault("psycopg2", _psycopg2_stub)
sys.modules.setdefault("psycopg2.pool", _psycopg2_pool_stub)
sys.modules.setdefault("psycopg2.extras", _psycopg2_stub.extras)

# ── redis stub ───────────────────────────────────────────────────────────────
_redis_stub = types.ModuleType("redis")
_redis_stub.Redis = MagicMock()
_redis_stub.ConnectionPool = MagicMock()
sys.modules.setdefault("redis", _redis_stub)

# ── tiktoken stub ────────────────────────────────────────────────────────────
_tiktoken_stub = types.ModuleType("tiktoken")
_tiktoken_stub.get_encoding = MagicMock(return_value=MagicMock(encode=lambda t: t.split()))
sys.modules.setdefault("tiktoken", _tiktoken_stub)

# ── langchain_groq stub ──────────────────────────────────────────────────────
_groq_stub = types.ModuleType("langchain_groq")
_groq_stub.ChatGroq = MagicMock()
sys.modules.setdefault("langchain_groq", _groq_stub)

# ── langgraph stubs ──────────────────────────────────────────────────────────
_langgraph_stub = types.ModuleType("langgraph")
_lg_graph = types.ModuleType("langgraph.graph")
_lg_graph.StateGraph = MagicMock()
_lg_graph.START = "__start__"
_lg_graph.END = "__end__"
_lg_msg = types.ModuleType("langgraph.graph.message")
_lg_msg.add_messages = lambda x: x          # identity reducer for tests
_langgraph_stub.graph = _lg_graph
sys.modules.setdefault("langgraph", _langgraph_stub)
sys.modules.setdefault("langgraph.graph", _lg_graph)
sys.modules.setdefault("langgraph.graph.message", _lg_msg)

# ── dotenv stub ──────────────────────────────────────────────────────────────
_dotenv_stub = types.ModuleType("dotenv")
_dotenv_stub.load_dotenv = MagicMock()
sys.modules.setdefault("dotenv", _dotenv_stub)

# ── utils.extraction stub (used by chatbot_node and planner_node) ─────────────
# TemplateCreation/utils/extraction.py is imported as `from utils.extraction import ...`
# because TemplateCreation/ is on sys.path at runtime. Add it here too.
_TC_DIR = _PROJECT_ROOT / "TemplateCreation"
if str(_TC_DIR) not in sys.path:
    sys.path.insert(0, str(_TC_DIR))

_utils_stub = types.ModuleType("utils")
_utils_extraction_stub = types.ModuleType("utils.extraction")
_utils_extraction_stub.check_satisfaction_signal = MagicMock(return_value=False)
_utils_extraction_stub.clean_chatbot_response = MagicMock(side_effect=lambda x: x)
_utils_extraction_stub.extract_planner_outputs = MagicMock(return_value=("tool prompt", "system prompt"))
_utils_stub.extraction = _utils_extraction_stub
sys.modules.setdefault("utils", _utils_stub)
sys.modules.setdefault("utils.extraction", _utils_extraction_stub)

# ── ToolGeneration.pipeline stub (patched before service import) ─────────────
_tg_pipeline_stub = types.ModuleType("ToolGeneration.pipeline")
_mock_generate_tool = MagicMock(return_value={"status": "success", "tool_id": "to00001", "template_id": "tem00001"})
_tg_pipeline_stub.generate_tool = _mock_generate_tool
_tg_stub = types.ModuleType("ToolGeneration")
_tg_stub.pipeline = _tg_pipeline_stub
sys.modules.setdefault("ToolGeneration", _tg_stub)
sys.modules.setdefault("ToolGeneration.pipeline", _tg_pipeline_stub)

# ---------------------------------------------------------------------------
# Now import TemplateCreation internals (stubs are in place)
# ---------------------------------------------------------------------------
# We need to stub ensure_tables before service.py runs it at import time.
# Use a fake psycopg2 connection so the pool never connects to real Postgres.
_psycopg2_stub.connect = MagicMock()

# Import graph and state first (no DB calls at import time)
from TemplateCreation.graph import build_graph, route_after_chatbot   # noqa: E402
from TemplateCreation.state import GraphState                         # noqa: E402

# Import db submodules so we can patch their functions before service.py
# calls ensure_tables() at module level.
import TemplateCreation.db.postgres_client as _pg_client   # noqa: E402
import TemplateCreation.db.redis_client as _redis_client   # noqa: E402

_pg_client.ensure_tables = MagicMock()           # prevent real DB calls at import
_redis_client.get_conversation_cache = MagicMock(return_value=None)

# Now import service — ensure_tables() is already patched via _pg_client
import TemplateCreation.service as service_module  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _make_ai_message(content: str) -> AIMessage:
    return AIMessage(content=content)


def _make_human_message(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def _satisfied_graph_result(extra_ai_content: str = "All requirements gathered.") -> dict:
    """Return a fake graph result with satisfied=True (Phase 2 should trigger)."""
    return {
        "messages": [
            _make_human_message("I need a weather tool"),
            _make_ai_message(extra_ai_content),
        ],
        "satisfied": True,
        "phase": "planning",
        "requirements": {"type": "weather api"},
        "tool_creation_prompt": "",
        "system_prompt": "",
    }


def _unsatisfied_graph_result(ai_content: str = "Can you tell me more?") -> dict:
    """Return a fake graph result with satisfied=False (still gathering)."""
    return {
        "messages": [
            _make_human_message("I need something"),
            _make_ai_message(ai_content),
        ],
        "satisfied": False,
        "phase": "gathering",
        "requirements": {},
        "tool_creation_prompt": "",
        "system_prompt": "",
    }


# ===========================================================================
# Test Suite
# ===========================================================================

class TestRoutingFunction(unittest.TestCase):
    """Unit tests for route_after_chatbot — pure logic, no I/O."""

    def test_returns_plan_when_satisfied(self):
        state = {"satisfied": True}
        self.assertEqual(route_after_chatbot(state), "plan")

    def test_returns_gather_when_not_satisfied(self):
        state = {"satisfied": False}
        self.assertEqual(route_after_chatbot(state), "gather")

    def test_returns_gather_when_key_missing(self):
        state = {}
        self.assertEqual(route_after_chatbot(state), "gather")


class TestGraphBuild(unittest.TestCase):
    """Verify build_graph() compiles without error."""

    def test_build_graph_returns_object(self):
        """build_graph() should return a compiled graph (non-None)."""
        mock_builder = MagicMock()
        mock_compiled = MagicMock()
        mock_builder.compile.return_value = mock_compiled

        with patch("TemplateCreation.graph.StateGraph", return_value=mock_builder):
            graph = build_graph()

        self.assertIsNotNone(graph)
        # Builder should have had nodes and edges registered
        mock_builder.add_node.assert_called()
        mock_builder.add_edge.assert_called()
        mock_builder.add_conditional_edges.assert_called()
        mock_builder.compile.assert_called_once()


class TestPhase1SingleTurn(unittest.TestCase):
    """
    Phase 1 — single turn where satisfied=False.

    Expectations:
      * chat_template() returns the AI text.
      * create_template() is NOT called.
      * generate_tool() is NOT called.
    """

    def setUp(self):
        _mock_generate_tool.reset_mock()

    @patch("TemplateCreation.service.is_template_finalized", return_value=False)
    @patch("TemplateCreation.service._load_history", return_value=[])
    @patch("TemplateCreation.service._persist_message")
    @patch("TemplateCreation.service.append_to_conversation_cache")
    @patch("TemplateCreation.service.build_graph")
    @patch("TemplateCreation.service.create_template")
    def test_phase1_no_phase2_trigger(
        self,
        mock_create_template,
        mock_build_graph,
        mock_append_cache,
        mock_persist,
        mock_load_history,
        mock_is_finalized,
    ):
        # Setup: graph returns unsatisfied state
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = _unsatisfied_graph_result("Can you tell me more?")
        mock_build_graph.return_value = mock_graph

        mock_persist.return_value = {
            "role": "user", "content": "Hello", "sequence_number": 1, "token_count": 1
        }

        result = service_module.chat_template(
            template_id="tem00001",
            user_id="usr00001",
            user_prompt="Hello, I need a tool",
        )

        # Should return an AI response string
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Can you tell me more?")

        # Phase 2 must NOT have been triggered
        mock_create_template.assert_not_called()
        _mock_generate_tool.assert_not_called()

        print("[PASS] Phase 1 single turn: create_template and generate_tool NOT called.")


class TestPhase2AutoTrigger(unittest.TestCase):
    """
    Phase 2 auto-trigger — graph returns satisfied=True.

    Expectations:
      * create_template() IS called automatically.
      * generate_tool(template_id) IS called after create_template.
      * The order of calls is: create_template → generate_tool.
    """

    def setUp(self):
        _mock_generate_tool.reset_mock()

    @patch("TemplateCreation.service.is_template_finalized", return_value=False)
    @patch("TemplateCreation.service._load_history")
    @patch("TemplateCreation.service._persist_message")
    @patch("TemplateCreation.service.append_to_conversation_cache")
    @patch("TemplateCreation.service.build_graph")
    @patch("TemplateCreation.service.create_template")
    def test_phase2_triggered_on_satisfied(
        self,
        mock_create_template,
        mock_build_graph,
        mock_append_cache,
        mock_persist,
        mock_load_history,
        mock_is_finalized,
    ):
        TEMPLATE_ID = "tem00001"
        USER_ID = "usr00001"

        full_history = [
            {"role": "user",      "content": "I need a weather tool", "sequence_number": 1, "token_count": 5},
            {"role": "assistant", "content": "All requirements gathered.", "sequence_number": 2, "token_count": 4},
        ]

        mock_load_history.side_effect = [
            [],            # first call (before user msg is appended)
            full_history,  # second call (after satisfied — full history fetch)
        ]

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = _satisfied_graph_result("All requirements gathered.")
        mock_build_graph.return_value = mock_graph

        mock_persist.return_value = {
            "role": "user", "content": "Great, finalize!", "sequence_number": 1, "token_count": 3
        }

        result = service_module.chat_template(
            template_id=TEMPLATE_ID,
            user_id=USER_ID,
            user_prompt="Great, finalize!",
        )

        # create_template MUST be called (Phase 2 triggered)
        mock_create_template.assert_called_once_with(USER_ID, TEMPLATE_ID, full_history)

        # generate_tool MUST be called after create_template
        _mock_generate_tool.assert_called_once_with(TEMPLATE_ID)

        print("[PASS] Phase 2 auto-triggered: create_template and generate_tool both called.")

    @patch("TemplateCreation.service.is_template_finalized", return_value=False)
    @patch("TemplateCreation.service._load_history")
    @patch("TemplateCreation.service._persist_message")
    @patch("TemplateCreation.service.append_to_conversation_cache")
    @patch("TemplateCreation.service.build_graph")
    @patch("TemplateCreation.service.create_template")
    def test_generate_tool_called_after_create_template(
        self,
        mock_create_template,
        mock_build_graph,
        mock_append_cache,
        mock_persist,
        mock_load_history,
        mock_is_finalized,
    ):
        """Verify the strict ordering: create_template → generate_tool."""
        TEMPLATE_ID = "tem00002"
        USER_ID = "usr00001"

        call_order: list[str] = []
        mock_create_template.side_effect = lambda *a, **kw: call_order.append("create_template")
        _mock_generate_tool.side_effect = lambda *a, **kw: call_order.append("generate_tool") or {"status": "success"}

        full_history = [{"role": "user", "content": "done", "sequence_number": 1, "token_count": 1}]
        mock_load_history.side_effect = [[], full_history]

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = _satisfied_graph_result()
        mock_build_graph.return_value = mock_graph

        mock_persist.return_value = {"role": "user", "content": "done", "sequence_number": 1, "token_count": 1}

        service_module.chat_template(
            template_id=TEMPLATE_ID,
            user_id=USER_ID,
            user_prompt="done",
        )

        self.assertEqual(call_order, ["create_template", "generate_tool"],
                         f"Expected create_template before generate_tool, got: {call_order}")

        print("[PASS] Call order verified: create_template → generate_tool.")


class TestFinalizedGuard(unittest.TestCase):
    """
    Guard check — if the template is already finalized, chat_template()
    must return early without invoking the graph or Phase 2.
    """

    def setUp(self):
        _mock_generate_tool.reset_mock()

    @patch("TemplateCreation.service.is_template_finalized", return_value=True)
    @patch("TemplateCreation.service.build_graph")
    @patch("TemplateCreation.service.create_template")
    def test_finalized_template_returns_early(
        self,
        mock_create_template,
        mock_build_graph,
        mock_is_finalized,
    ):
        result = service_module.chat_template(
            template_id="tem99999",
            user_id="usr00001",
            user_prompt="Anything",
        )

        self.assertIn("already been created", result.lower())
        mock_build_graph.assert_not_called()
        mock_create_template.assert_not_called()
        _mock_generate_tool.assert_not_called()

        print("[PASS] Finalized guard: graph and Phase 2 not called for finalized template.")


class TestGraphStateShape(unittest.TestCase):
    """Verify GraphState has all required fields with correct types."""

    def test_state_fields_exist(self):
        from typing import get_type_hints
        hints = get_type_hints(GraphState)
        required = {"messages", "phase", "satisfied", "requirements",
                    "tool_creation_prompt", "system_prompt"}
        missing = required - hints.keys()
        self.assertFalse(missing, f"GraphState is missing fields: {missing}")
        print("[PASS] GraphState has all required fields.")

    def test_state_types(self):
        from typing import get_type_hints
        hints = get_type_hints(GraphState)
        self.assertIs(hints["phase"], str)
        self.assertIs(hints["satisfied"], bool)
        self.assertIs(hints["requirements"], dict)
        self.assertIs(hints["tool_creation_prompt"], str)
        self.assertIs(hints["system_prompt"], str)
        print("[PASS] GraphState field types are correct.")


class TestFullPipelineFlow(unittest.TestCase):
    """
    End-to-end simulation of the complete multi-turn pipeline:

      Turn 1: user sends first message → Phase 1 continues (not satisfied)
      Turn 2: user sends final message → Phase 1 satisfied → Phase 2 auto-starts
                                          → generate_tool called
      Turn 3: user tries again         → template finalized, rejected
    """

    def setUp(self):
        _mock_generate_tool.reset_mock()

    @patch("TemplateCreation.service.append_to_conversation_cache")
    @patch("TemplateCreation.service._persist_message")
    @patch("TemplateCreation.service.create_template")
    @patch("TemplateCreation.service.build_graph")
    @patch("TemplateCreation.service._load_history")
    @patch("TemplateCreation.service.is_template_finalized")
    def test_full_pipeline(
        self,
        mock_is_finalized,
        mock_load_history,
        mock_build_graph,
        mock_create_template,
        mock_persist,
        mock_append_cache,
    ):
        TEMPLATE_ID = "tem00010"
        USER_ID = "usr00001"

        history_store: list[dict] = []

        def fake_load_history(tid):
            return list(history_store)

        def fake_persist(tid, role, content, seq):
            rec = {"role": role, "content": content, "sequence_number": seq, "token_count": 5}
            history_store.append(rec)
            return rec

        mock_load_history.side_effect = fake_load_history
        mock_persist.side_effect = fake_persist

        # -- Turn 1: unsatisfied --
        mock_is_finalized.return_value = False
        unsatisfied_graph = MagicMock()
        unsatisfied_graph.invoke.return_value = _unsatisfied_graph_result("Please describe more.")
        mock_build_graph.return_value = unsatisfied_graph

        r1 = service_module.chat_template(TEMPLATE_ID, USER_ID, "I want a weather bot")
        self.assertIsInstance(r1, str)
        mock_create_template.assert_not_called()
        _mock_generate_tool.assert_not_called()
        print(f"[Turn 1] Response: '{r1}' | Phase 2 NOT triggered ✓")

        # -- Turn 2: satisfied → Phase 2 auto-trigger --
        satisfied_graph = MagicMock()
        satisfied_graph.invoke.return_value = _satisfied_graph_result("Great! I have all I need.")
        mock_build_graph.return_value = satisfied_graph

        r2 = service_module.chat_template(TEMPLATE_ID, USER_ID, "It fetches real-time weather data.")
        self.assertIsInstance(r2, str)

        # Phase 2 must fire
        mock_create_template.assert_called_once()
        args = mock_create_template.call_args
        self.assertEqual(args[0][0], USER_ID)
        self.assertEqual(args[0][1], TEMPLATE_ID)

        # generate_tool must fire with correct template_id
        _mock_generate_tool.assert_called_once_with(TEMPLATE_ID)
        print(f"[Turn 2] Response: '{r2}' | Phase 2 triggered ✓ | generate_tool called ✓")

        # -- Turn 3: template finalized, must reject --
        mock_is_finalized.return_value = True
        r3 = service_module.chat_template(TEMPLATE_ID, USER_ID, "Can I add more features?")
        self.assertIn("already been created", r3.lower())
        # create_template and generate_tool call counts must not increase
        mock_create_template.assert_called_once()   # still just once
        _mock_generate_tool.assert_called_once()    # still just once
        print(f"[Turn 3] Rejected with: '{r3}' | Guard works ✓")

        print("\n[PASS] Full pipeline simulation completed successfully.")


# ===========================================================================
# Entry point (also works with pytest)
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TemplateCreation Pipeline Tests")
    print("=" * 70)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    )
    sys.exit(0 if result.wasSuccessful() else 1)
