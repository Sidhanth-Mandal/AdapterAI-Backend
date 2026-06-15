"""
test.py — ToolGeneration Pipeline Tests
========================================

Modules covered
---------------
1.  validator.py        — all 6 validation stages + ValidationReport
2.  metadata.py         — extract_tool_metadata()
3.  generator.py        — _strip_fences(), _parse_json(), _summarise_history()
4.  executer.py         — execution_call() happy path, error paths, _load_tool_json()
5.  redis_client.py     — init_error_history(), append_error_history(),
                          get_error_history(), rename_key()
6.  db.py               — get_next_tool_id(), fetch_template(),
                          insert_tool(), update_template_tool_information()
7.  pipeline.py         — generate_tool() success path, template-not-found,
                          empty prompt, Docker unreachable, max-attempts exhausted,
                          _build_function_calls(), _collect_validation_errors(),
                          _latest_stage_name()

Run
---
    # from the project root (AdapterAI/)
    python -m pytest ToolGeneration/test.py -v

Or directly:
    python ToolGeneration/test.py
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Stub heavy external dependencies before any project imports
# ---------------------------------------------------------------------------

# ── psycopg2 ─────────────────────────────────────────────────────────────────
_psycopg2 = types.ModuleType("psycopg2")
_psycopg2.extensions = types.SimpleNamespace(connection=object)
_psycopg2.extras = types.SimpleNamespace(RealDictCursor=None)
_psycopg2.connect = MagicMock()
_psycopg2_pool = types.ModuleType("psycopg2.pool")
_psycopg2_pool.ThreadedConnectionPool = MagicMock()
_psycopg2.pool = _psycopg2_pool
sys.modules.setdefault("psycopg2", _psycopg2)
sys.modules.setdefault("psycopg2.pool", _psycopg2_pool)
sys.modules.setdefault("psycopg2.extras", _psycopg2.extras)

# ── redis ─────────────────────────────────────────────────────────────────────
_redis_mod = types.ModuleType("redis")
_redis_mod.Redis = MagicMock()
_redis_mod.ConnectionPool = MagicMock()
_redis_mod.from_url = MagicMock(return_value=MagicMock())
sys.modules.setdefault("redis", _redis_mod)

# ── dotenv ────────────────────────────────────────────────────────────────────
_dotenv = types.ModuleType("dotenv")
_dotenv.load_dotenv = MagicMock()
sys.modules.setdefault("dotenv", _dotenv)

# ── langchain_groq ────────────────────────────────────────────────────────────
_groq = types.ModuleType("langchain_groq")
_groq.ChatGroq = MagicMock()
sys.modules.setdefault("langchain_groq", _groq)

# ── langchain_core ────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage   # real package

# ── pydantic (real) is installed; ToolSchema imports it — no stub needed ──────

# ── json_repair (optional — generator falls back to stdlib) ──────────────────
_json_repair_stub = types.ModuleType("json_repair")
_json_repair_stub.loads = MagicMock(side_effect=ImportError("stub — fall through to json.loads"))
sys.modules.setdefault("json_repair", _json_repair_stub)

# ── requests ──────────────────────────────────────────────────────────────────
import requests as _requests_real   # real package needed by executer

# ---------------------------------------------------------------------------
# Patch env vars and db-level connect calls before importing project modules
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("POSTGRES_DSN", "postgresql://test:test@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("GROQ_API_KEY", "test-key")

# Patch psycopg2.connect for db.py (called at module level via load_dotenv)
_psycopg2.connect = MagicMock()

# ---------------------------------------------------------------------------
# Now import ToolGeneration modules
# ---------------------------------------------------------------------------
import ToolGeneration.validator as validator_mod
import ToolGeneration.metadata as metadata_mod
import ToolGeneration.redis_client as redis_client_mod

with patch("ToolGeneration.generator.ChatGroq"):
    import ToolGeneration.generator as generator_mod

with patch("ToolGeneration.db.psycopg2.connect", MagicMock()):
    import ToolGeneration.db as db_mod

import ToolGeneration.executer as executer_mod

with (
    patch("ToolGeneration.db.psycopg2.connect", MagicMock()),
    patch("ToolGeneration.redis_client.get_redis", MagicMock()),
):
    import ToolGeneration.pipeline as pipeline_mod

from ToolGeneration.validator import (
    validate_tool, ValidationReport, StageResult,
    _stage_schema_integrity, _stage_syntax_check,
    _stage_static_safety, _stage_structural_match,
    _stage_dependency_check,
)
from ToolGeneration.metadata import extract_tool_metadata
from ToolGeneration.generator import _strip_fences, _parse_json, _summarise_history
from ToolGeneration.executer import execution_call, _load_tool_json
from ToolGeneration.pipeline import (
    generate_tool,
    _build_function_calls,
    _collect_validation_errors,
    _latest_stage_name,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

def _good_tool(*, code: str | None = None) -> dict:
    """Return a minimal valid tool dict that passes all validation stages."""
    return {
        "tool_name": "weather_tool",
        "tool_description": "Fetches current weather for a city.",
        "category": "weather",
        "dependencies": ["requests"],
        "functions": [
            {
                "name": "get_weather",
                "description": "Returns weather data for a city.",
                "parameters": [
                    {"name": "city", "type": "str", "description": "City name",
                     "required": True, "example": "London"},
                ],
                "outputs": [
                    {"name": "temperature", "type": "float",
                     "description": "Temperature in Celsius", "example": 22.5},
                    {"name": "condition",   "type": "str",
                     "description": "Weather condition",     "example": "Sunny"},
                ],
                "return_type": "dict",
            }
        ],
        "code": code or (
            "import requests\n\n"
            "def get_weather(city: str) -> dict:\n"
            "    resp = requests.get(\n"
            "        f'https://api.weather.example.com/?q={city}',\n"
            "        timeout=10,\n"
            "    )\n"
            "    data = resp.json()\n"
            "    return {'temperature': data.get('temp', 0.0),\n"
            "            'condition':   data.get('desc', 'Unknown')}\n"
        ),
    }


# ===========================================================================
# 1. Validator — Stage-level unit tests
# ===========================================================================

class TestStageSchemaIntegrity(unittest.TestCase):
    """Stage 1 — Schema Integrity"""

    def test_passes_valid_tool(self):
        r = _stage_schema_integrity(_good_tool())
        self.assertTrue(r.passed)
        self.assertEqual(r.errors, [])

    def test_missing_required_fields(self):
        r = _stage_schema_integrity({})
        self.assertFalse(r.passed)
        self.assertTrue(any("tool_name" in e for e in r.errors))
        self.assertTrue(any("tool_description" in e for e in r.errors))
        self.assertTrue(any("code" in e for e in r.errors))
        self.assertTrue(any("functions" in e for e in r.errors))

    def test_invalid_tool_name_identifier(self):
        t = _good_tool()
        t["tool_name"] = "123-bad name"
        r = _stage_schema_integrity(t)
        self.assertFalse(r.passed)
        self.assertTrue(any("valid Python identifier" in e for e in r.errors))

    def test_function_missing_name(self):
        t = _good_tool()
        t["functions"][0]["name"] = ""
        r = _stage_schema_integrity(t)
        self.assertFalse(r.passed)

    def test_param_invalid_identifier(self):
        t = _good_tool()
        t["functions"][0]["parameters"][0]["name"] = "1bad"
        r = _stage_schema_integrity(t)
        self.assertFalse(r.passed)

    def test_no_description_warning(self):
        t = _good_tool()
        t["functions"][0]["description"] = ""
        r = _stage_schema_integrity(t)
        self.assertTrue(any("no description" in w for w in r.warnings))


class TestStageSyntaxCheck(unittest.TestCase):
    """Stage 2 — Syntax Check"""

    def test_passes_valid_python(self):
        r = _stage_syntax_check("def foo():\n    return 42\n")
        self.assertTrue(r.passed)

    def test_fails_syntax_error(self):
        r = _stage_syntax_check("def foo(\n    return 42\n")
        self.assertFalse(r.passed)
        self.assertTrue(any("SyntaxError" in e for e in r.errors))

    def test_passes_empty_code(self):
        r = _stage_syntax_check("")
        self.assertTrue(r.passed)


class TestStageStaticSafety(unittest.TestCase):
    """Stage 3 — Static Safety"""

    def test_safe_code_passes(self):
        r = _stage_static_safety("def foo():\n    return 1\n")
        self.assertTrue(r.passed)
        self.assertTrue(getattr(r, "_safe", True))

    def test_exec_call_blocked(self):
        r = _stage_static_safety("exec('import os')\n")
        self.assertFalse(r.passed)
        self.assertFalse(getattr(r, "_safe", True))

    def test_os_system_blocked(self):
        r = _stage_static_safety("import os\nos.system('ls')\n")
        self.assertFalse(r.passed)

    def test_subprocess_import_blocked(self):
        r = _stage_static_safety("import subprocess\n")
        self.assertFalse(r.passed)

    def test_file_write_blocked(self):
        r = _stage_static_safety("open('out.txt', 'w').write('x')\n")
        self.assertFalse(r.passed)

    def test_hardcoded_secret_blocked(self):
        r = _stage_static_safety("api_key = 'sk-VERYLONGSECRETKEY12345'\n")
        self.assertFalse(r.passed)

    def test_network_call_without_timeout_warns(self):
        code = (
            "import requests\n"
            "def f():\n"
            "    return requests.get('http://example.com')\n"
        )
        r = _stage_static_safety(code)
        self.assertTrue(any("timeout" in w for w in r.warnings))


class TestStageStructuralMatch(unittest.TestCase):
    """Stage 4 — Structural Match"""

    def test_passes_when_fn_exists_with_correct_params(self):
        t = _good_tool()
        r = _stage_structural_match(t, t["code"])
        self.assertTrue(r.passed)

    def test_fails_when_fn_declared_but_not_in_code(self):
        t = _good_tool()
        t["functions"].append({
            "name": "missing_fn",
            "description": "Not in code",
            "parameters": [],
            "outputs": [],
            "return_type": "dict",
        })
        r = _stage_structural_match(t, t["code"])
        self.assertFalse(r.passed)
        self.assertTrue(any("missing_fn" in e for e in r.errors))

    def test_fails_when_param_missing_from_code_signature(self):
        t = _good_tool()
        t["functions"][0]["parameters"].append(
            {"name": "units", "type": "str", "description": "Units", "required": False}
        )
        r = _stage_structural_match(t, t["code"])
        self.assertFalse(r.passed)

    def test_warns_undeclared_public_fn_in_code(self):
        t = _good_tool()
        extra_code = t["code"] + "\ndef helper():\n    pass\n"
        t["code"] = extra_code
        r = _stage_structural_match(t, extra_code)
        self.assertTrue(any("helper" in w for w in r.warnings))


class TestStageDependencyCheck(unittest.TestCase):
    """Stage 5 — Dependency Check"""

    def test_no_deps_gives_warning(self):
        t = _good_tool()
        t["dependencies"] = []
        r = _stage_dependency_check(t)
        self.assertTrue(r.passed)          # warning, not error
        self.assertTrue(any("No dependencies" in w for w in r.warnings))

    def test_installed_dep_passes(self):
        t = _good_tool()
        t["dependencies"] = ["requests"]   # always installed in this env
        r = _stage_dependency_check(t)
        # requests is real — should not add a missing error for it
        missing_errors = [e for e in r.errors if "Missing" in e and "requests" in e]
        self.assertEqual(missing_errors, [])

    def test_missing_dep_adds_error(self):
        t = _good_tool()
        t["dependencies"] = ["__nonexistent_pkg_xyz__"]
        r = _stage_dependency_check(t)
        self.assertFalse(r.passed)
        self.assertTrue(any("Missing" in e for e in r.errors))


class TestValidateTool(unittest.TestCase):
    """Full validate_tool() integration"""

    def test_good_tool_passes(self):
        report = validate_tool(_good_tool())
        self.assertIsInstance(report, ValidationReport)
        # Schema + Syntax + Safety + Structural must all pass
        stage_names = {s.name: s.passed for s in report.stages}
        self.assertTrue(stage_names["Schema Integrity"])
        self.assertTrue(stage_names["Syntax Check"])
        self.assertTrue(stage_names["Static Safety Analysis"])
        self.assertTrue(stage_names["Structural Match"])

    def test_unsafe_tool_marked_not_safe(self):
        t = _good_tool()
        t["code"] = "exec('import os')\ndef get_weather(city):\n    return {}\n"
        report = validate_tool(t)
        self.assertFalse(report.safe)

    def test_syntax_error_skips_dry_run(self):
        t = _good_tool()
        t["code"] = "def broken(\n"
        report = validate_tool(t)
        dry_run = next(s for s in report.stages if s.name == "Sandboxed Dry-run")
        self.assertFalse(dry_run.passed)
        self.assertTrue(any("Skipped" in e for e in dry_run.errors))

    def test_report_summary_string(self):
        report = validate_tool(_good_tool())
        summary = report.summary()
        self.assertIn("VALIDATION", summary)

    def test_accepts_json_string(self):
        """validate_tool should also accept a raw JSON string."""
        json_str = json.dumps(_good_tool())
        report = validate_tool(json_str)
        self.assertIsInstance(report, ValidationReport)


# ===========================================================================
# 2. Metadata
# ===========================================================================

class TestExtractToolMetadata(unittest.TestCase):

    def test_extracts_name_and_description(self):
        m = extract_tool_metadata(_good_tool())
        self.assertEqual(m["name"], "weather_tool")
        self.assertIn("weather", m["description"].lower())

    def test_tool_information_contains_function_name(self):
        m = extract_tool_metadata(_good_tool())
        self.assertIn("get_weather", m["tool_information"])

    def test_tool_information_contains_param_name(self):
        m = extract_tool_metadata(_good_tool())
        self.assertIn("city", m["tool_information"])

    def test_tool_information_contains_output_name(self):
        m = extract_tool_metadata(_good_tool())
        self.assertIn("temperature", m["tool_information"])

    def test_empty_functions(self):
        t = _good_tool()
        t["functions"] = []
        m = extract_tool_metadata(t)
        self.assertEqual(m["tool_information"], "")

    def test_optional_description_on_output(self):
        t = _good_tool()
        t["functions"][0]["outputs"] = []
        m = extract_tool_metadata(t)
        self.assertIn("result", m["tool_information"])

    def test_returns_dict_with_all_keys(self):
        m = extract_tool_metadata(_good_tool())
        self.assertIn("name", m)
        self.assertIn("description", m)
        self.assertIn("tool_information", m)


# ===========================================================================
# 3. Generator helpers
# ===========================================================================

class TestStripFences(unittest.TestCase):

    def test_no_fences_unchanged(self):
        self.assertEqual(_strip_fences('{"a":1}'), '{"a":1}')

    def test_strips_plain_fences(self):
        raw = "```\n{\"a\":1}\n```"
        self.assertEqual(_strip_fences(raw), '{"a":1}')

    def test_strips_json_language_fences(self):
        raw = "```json\n{\"a\":1}\n```"
        self.assertEqual(_strip_fences(raw), '{"a":1}')

    def test_strips_leading_trailing_whitespace(self):
        self.assertEqual(_strip_fences("  hello  "), "hello")


class TestParseJson(unittest.TestCase):

    def test_parses_valid_json(self):
        result = _parse_json('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})

    def test_parses_fenced_json(self):
        result = _parse_json("```json\n{\"key\": \"value\"}\n```")
        self.assertEqual(result, {"key": "value"})

    def test_raises_on_invalid_json(self):
        # When json_repair is absent (already stubbed as empty module), stdlib raises
        with self.assertRaises(Exception):
            _parse_json("{bad json {{{{")


class TestSummariseHistory(unittest.TestCase):

    def test_empty_history_returns_no_prior_errors(self):
        result = _summarise_history([])
        self.assertEqual(result, "(no prior errors)")

    def test_prompt_entry_is_skipped(self):
        history = [{"type": "prompt", "content": "do something"}]
        result = _summarise_history(history)
        self.assertEqual(result, "(no prior errors)")

    def test_validation_error_rendered(self):
        history = [
            {"type": "validation_error", "stage": "Schema Integrity",
             "errors": ["Missing field: 'tool_name'"]},
        ]
        result = _summarise_history(history)
        self.assertIn("Validation error", result)
        self.assertIn("Schema Integrity", result)
        self.assertIn("Missing field", result)

    def test_execution_error_rendered(self):
        history = [
            {"type": "execution_error", "error": "TypeError: something broke"},
        ]
        result = _summarise_history(history)
        self.assertIn("Execution error", result)
        self.assertIn("TypeError", result)

    def test_repair_entry_rendered(self):
        history = [
            {"type": "repair",
             "cause": ["Missing return statement"],
             "fix":   ["Added return {'result': data}"]},
        ]
        result = _summarise_history(history)
        self.assertIn("Repair applied", result)
        self.assertIn("Missing return statement", result)
        self.assertIn("Added return", result)

    def test_mixed_history(self):
        history = [
            {"type": "prompt", "content": "build a tool"},
            {"type": "validation_error", "stage": "Syntax Check", "errors": ["SyntaxError"]},
            {"type": "repair", "cause": ["bad syntax"], "fix": ["fixed indentation"]},
        ]
        result = _summarise_history(history)
        self.assertIn("Validation error", result)
        self.assertIn("Repair applied", result)
        self.assertNotIn("build a tool", result)   # prompt skipped


# ===========================================================================
# 4. Executer
# ===========================================================================

class TestLoadToolJson(unittest.TestCase):

    def test_accepts_dict(self):
        d = {"functions": []}
        self.assertEqual(_load_tool_json(d), d)

    def test_accepts_json_string(self):
        d = {"functions": [], "tool_name": "x"}
        self.assertEqual(_load_tool_json(json.dumps(d)), d)

    def test_raises_on_bad_path_and_bad_json(self):
        with self.assertRaises(FileNotFoundError):
            _load_tool_json("/no/such/file.json")

    def test_accepts_real_file(self, tmp_path=None):
        import tempfile, os
        d = {"functions": [{"name": "f"}]}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as fh:
            json.dump(d, fh)
            fname = fh.name
        try:
            result = _load_tool_json(fname)
            self.assertEqual(result, d)
        finally:
            os.unlink(fname)


class TestExecutionCall(unittest.TestCase):

    def _tool_with_fn(self, fn_name: str = "get_weather") -> dict:
        return {
            "functions": [{"name": fn_name, "parameters": []}],
            "tool_name": "test_tool",
        }

    def test_raises_on_empty_function_calls(self):
        with self.assertRaises(ValueError):
            execution_call({}, tool_json=self._tool_with_fn())

    def test_raises_on_unknown_function(self):
        with self.assertRaises(ValueError):
            execution_call(
                {"nonexistent": {}},
                tool_json=self._tool_with_fn("get_weather"),
            )

    @patch("ToolGeneration.executer.requests.post")
    def test_happy_path_returns_result(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"get_weather": {"temperature": 22.5}}
        mock_post.return_value = mock_resp

        result = execution_call(
            {"get_weather": {"city": "London"}},
            tool_json=self._tool_with_fn("get_weather"),
        )
        self.assertEqual(result, {"get_weather": {"temperature": 22.5}})
        mock_post.assert_called_once()

    @patch("ToolGeneration.executer.requests.post")
    def test_raises_http_error_on_non_200(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"detail": "Internal Server Error"}
        mock_post.return_value = mock_resp

        with self.assertRaises(_requests_real.HTTPError):
            execution_call(
                {"get_weather": {}},
                tool_json=self._tool_with_fn("get_weather"),
            )

    @patch("ToolGeneration.executer.requests.post",
           side_effect=_requests_real.exceptions.ConnectionError("refused"))
    def test_raises_connection_error_when_docker_down(self, _):
        with self.assertRaises(_requests_real.exceptions.ConnectionError):
            execution_call(
                {"get_weather": {}},
                tool_json=self._tool_with_fn("get_weather"),
            )

    @patch("ToolGeneration.executer.requests.post")
    def test_custom_runner_url_used(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp

        execution_call(
            {"get_weather": {}},
            tool_json=self._tool_with_fn("get_weather"),
            runner_url="http://myrunner:9999",
        )
        called_url = mock_post.call_args[0][0]
        self.assertIn("myrunner:9999", called_url)


# ===========================================================================
# 5. Redis client
# ===========================================================================

class TestRedisClient(unittest.TestCase):

    def setUp(self):
        self.mock_r = MagicMock()
        self.patcher = patch(
            "ToolGeneration.redis_client.get_redis",
            return_value=self.mock_r,
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_init_error_history_deletes_then_pushes(self):
        redis_client_mod.init_error_history("run:key1", "build a weather tool")
        self.mock_r.delete.assert_called_once_with("run:key1")
        self.mock_r.rpush.assert_called_once()
        pushed_json = self.mock_r.rpush.call_args[0][1]
        entry = json.loads(pushed_json)
        self.assertEqual(entry["type"], "prompt")
        self.assertEqual(entry["content"], "build a weather tool")

    def test_append_error_history_pushes_entry(self):
        entry = {"type": "validation_error", "stage": "Syntax Check", "errors": ["bad"]}
        redis_client_mod.append_error_history("run:key1", entry)
        self.mock_r.rpush.assert_called_once()
        pushed = json.loads(self.mock_r.rpush.call_args[0][1])
        self.assertEqual(pushed["type"], "validation_error")

    def test_get_error_history_parses_json_list(self):
        entries = [
            json.dumps({"type": "prompt", "content": "x"}),
            json.dumps({"type": "validation_error", "errors": []}),
        ]
        self.mock_r.lrange.return_value = entries
        result = redis_client_mod.get_error_history("run:key1")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["type"], "prompt")
        self.assertEqual(result[1]["type"], "validation_error")

    def test_get_error_history_returns_empty_list_on_no_key(self):
        self.mock_r.lrange.return_value = []
        result = redis_client_mod.get_error_history("nonexistent")
        self.assertEqual(result, [])

    def test_rename_key_renames_when_old_exists(self):
        self.mock_r.exists.return_value = True
        redis_client_mod.rename_key("old:key", "to00001")
        self.mock_r.delete.assert_called_once_with("to00001")
        self.mock_r.rename.assert_called_once_with("old:key", "to00001")

    def test_rename_key_noop_when_old_missing(self):
        self.mock_r.exists.return_value = False
        redis_client_mod.rename_key("old:key", "to00001")
        self.mock_r.rename.assert_not_called()


# ===========================================================================
# 6. DB layer
# ===========================================================================

class TestDbGetNextToolId(unittest.TestCase):

    def _mock_conn(self, rows):
        """Return a context-manager-compatible mock connection."""
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = rows
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        return mock_conn

    @patch("ToolGeneration.db.get_connection")
    def test_returns_to00001_when_no_tools(self, mock_gc):
        mock_gc.return_value = self._mock_conn([])
        result = db_mod.get_next_tool_id()
        self.assertEqual(result, "to00001")

    @patch("ToolGeneration.db.get_connection")
    def test_returns_incremented_id(self, mock_gc):
        mock_gc.return_value = self._mock_conn([("to00003",), ("to00001",)])
        result = db_mod.get_next_tool_id()
        self.assertEqual(result, "to00004")

    @patch("ToolGeneration.db.get_connection")
    def test_fetch_template_returns_none_when_not_found(self, mock_gc):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_gc.return_value = mock_conn

        result = db_mod.fetch_template("nonexistent")
        self.assertIsNone(result)

    @patch("ToolGeneration.db.get_connection")
    def test_fetch_template_returns_dict(self, mock_gc):
        row = {"template_id": "tem00001", "tool_generation_prompt": "build something"}
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = row
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_gc.return_value = mock_conn

        result = db_mod.fetch_template("tem00001")
        self.assertEqual(result["template_id"], "tem00001")

    @patch("ToolGeneration.db.get_connection")
    def test_insert_tool_executes_without_error(self, mock_gc):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_gc.return_value = mock_conn

        db_mod.insert_tool(
            tool_id="to00001",
            template_id="tem00001",
            name="Weather Tool",
            description="Fetches weather",
            language="python",
            tool_json=_good_tool(),
            version="1.0.0",
        )
        mock_cur.execute.assert_called_once()

    @patch("ToolGeneration.db.get_connection")
    def test_update_template_tool_information(self, mock_gc):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_gc.return_value = mock_conn

        db_mod.update_template_tool_information("tem00001", "Function: get_weather\n...")
        mock_cur.execute.assert_called_once()


# ===========================================================================
# 7. Pipeline helpers
# ===========================================================================

class TestBuildFunctionCalls(unittest.TestCase):

    def test_uses_example_values(self):
        tool_json = {
            "functions": [{
                "name": "get_weather",
                "parameters": [
                    {"name": "city", "type": "str", "example": "London"},
                    {"name": "units", "type": "str", "example": "metric"},
                ],
            }]
        }
        result = _build_function_calls(tool_json)
        self.assertEqual(result["get_weather"]["city"], "London")
        self.assertEqual(result["get_weather"]["units"], "metric")

    def test_falls_back_to_stubs_when_no_example(self):
        tool_json = {
            "functions": [{
                "name": "get_weather",
                "parameters": [
                    {"name": "city", "type": "str"},
                ],
            }]
        }
        result = _build_function_calls(tool_json)
        self.assertEqual(result["get_weather"]["city"], "test_value")

    def test_skips_functions_with_no_name(self):
        tool_json = {"functions": [{"name": "", "parameters": []}]}
        result = _build_function_calls(tool_json)
        self.assertEqual(result, {})

    def test_empty_functions_list(self):
        result = _build_function_calls({"functions": []})
        self.assertEqual(result, {})

    def test_multiple_functions(self):
        tool_json = {
            "functions": [
                {"name": "fn_a", "parameters": [{"name": "x", "type": "int", "example": 5}]},
                {"name": "fn_b", "parameters": [{"name": "y", "type": "bool", "example": True}]},
            ]
        }
        result = _build_function_calls(tool_json)
        self.assertIn("fn_a", result)
        self.assertIn("fn_b", result)
        self.assertEqual(result["fn_a"]["x"], 5)
        self.assertEqual(result["fn_b"]["y"], True)


class TestCollectValidationErrors(unittest.TestCase):

    def _make_report(self, stages: list[StageResult]):
        return MagicMock(stages=stages)

    def test_collects_errors_from_failing_stages(self):
        s = StageResult(name="Syntax Check", passed=True)
        s.add_error("SyntaxError at line 5")
        report = self._make_report([s])
        result = _collect_validation_errors(report)
        self.assertIn("Syntax Check", result)
        self.assertIn("SyntaxError at line 5", result)

    def test_skips_passing_stages(self):
        s = StageResult(name="Schema Integrity", passed=True)
        report = self._make_report([s])
        result = _collect_validation_errors(report)
        self.assertEqual(result, "Unknown validation error")

    def test_returns_unknown_when_no_errors(self):
        report = self._make_report([])
        result = _collect_validation_errors(report)
        self.assertEqual(result, "Unknown validation error")


class TestLatestStageName(unittest.TestCase):

    def test_returns_first_failing_stage(self):
        s1 = StageResult(name="Schema Integrity", passed=True)
        s2 = StageResult(name="Syntax Check", passed=True)
        s2.add_error("bad syntax")
        report = MagicMock(stages=[s1, s2])
        self.assertEqual(_latest_stage_name(report), "Syntax Check")

    def test_returns_unknown_when_all_pass(self):
        s = StageResult(name="Schema Integrity", passed=True)
        report = MagicMock(stages=[s])
        self.assertEqual(_latest_stage_name(report), "Unknown")


# ===========================================================================
# 8. Pipeline — generate_tool() integration
# ===========================================================================

class TestGenerateTool(unittest.TestCase):
    """
    All external calls (DB, Redis, generator, validator, executer) are mocked
    so the pipeline logic can be tested without live services.
    """

    def _default_patches(self):
        """Return a dict of commonly needed patch targets."""
        return {
            "ToolGeneration.pipeline.db.fetch_template": MagicMock(
                return_value={
                    "template_id": "tem00001",
                    "tool_generation_prompt": "Build a weather tool",
                }
            ),
            "ToolGeneration.pipeline.generator.generate_tool_json": MagicMock(
                return_value=_good_tool()
            ),
            "ToolGeneration.pipeline.redis_client.init_error_history": MagicMock(),
            "ToolGeneration.pipeline.redis_client.append_error_history": MagicMock(),
            "ToolGeneration.pipeline.redis_client.get_error_history": MagicMock(
                return_value=[{"type": "prompt", "content": "Build a weather tool"}]
            ),
            "ToolGeneration.pipeline.redis_client.rename_key": MagicMock(),
            "ToolGeneration.pipeline.validate_tool": MagicMock(
                return_value=MagicMock(
                    passed=True, safe=True,
                    stages=[], errors=[], warnings=[],
                )
            ),
            "ToolGeneration.pipeline.execution_call": MagicMock(
                return_value={"get_weather": {"temperature": 22.5}}
            ),
            "ToolGeneration.pipeline.meta_module.extract_tool_metadata": MagicMock(
                return_value={
                    "name": "Weather Tool",
                    "description": "Fetches weather",
                    "tool_information": "Function: get_weather\n...",
                }
            ),
            "ToolGeneration.pipeline.db.get_next_tool_id": MagicMock(
                return_value="to00001"
            ),
            "ToolGeneration.pipeline.db.insert_tool": MagicMock(),
            "ToolGeneration.pipeline.db.update_template_tool_information": MagicMock(),
        }

    def _run_with_patches(self, overrides: dict = None) -> dict:
        patches = self._default_patches()
        if overrides:
            patches.update(overrides)
        with (
            patch("ToolGeneration.pipeline.db.fetch_template",             patches["ToolGeneration.pipeline.db.fetch_template"]),
            patch("ToolGeneration.pipeline.generator.generate_tool_json",  patches["ToolGeneration.pipeline.generator.generate_tool_json"]),
            patch("ToolGeneration.pipeline.redis_client.init_error_history",   patches["ToolGeneration.pipeline.redis_client.init_error_history"]),
            patch("ToolGeneration.pipeline.redis_client.append_error_history", patches["ToolGeneration.pipeline.redis_client.append_error_history"]),
            patch("ToolGeneration.pipeline.redis_client.get_error_history",    patches["ToolGeneration.pipeline.redis_client.get_error_history"]),
            patch("ToolGeneration.pipeline.redis_client.rename_key",           patches["ToolGeneration.pipeline.redis_client.rename_key"]),
            patch("ToolGeneration.pipeline.validate_tool",                 patches["ToolGeneration.pipeline.validate_tool"]),
            patch("ToolGeneration.pipeline.execution_call",                patches["ToolGeneration.pipeline.execution_call"]),
            patch("ToolGeneration.pipeline.meta_module.extract_tool_metadata", patches["ToolGeneration.pipeline.meta_module.extract_tool_metadata"]),
            patch("ToolGeneration.pipeline.db.get_next_tool_id",           patches["ToolGeneration.pipeline.db.get_next_tool_id"]),
            patch("ToolGeneration.pipeline.db.insert_tool",                patches["ToolGeneration.pipeline.db.insert_tool"]),
            patch("ToolGeneration.pipeline.db.update_template_tool_information", patches["ToolGeneration.pipeline.db.update_template_tool_information"]),
        ):
            return generate_tool("tem00001")

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_success_returns_correct_dict(self):
        result = self._run_with_patches()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_id"], "to00001")
        self.assertEqual(result["template_id"], "tem00001")

    def test_success_inserts_tool_into_db(self):
        patches = self._default_patches()
        with (
            patch("ToolGeneration.pipeline.db.fetch_template",             patches["ToolGeneration.pipeline.db.fetch_template"]),
            patch("ToolGeneration.pipeline.generator.generate_tool_json",  patches["ToolGeneration.pipeline.generator.generate_tool_json"]),
            patch("ToolGeneration.pipeline.redis_client.init_error_history",   patches["ToolGeneration.pipeline.redis_client.init_error_history"]),
            patch("ToolGeneration.pipeline.redis_client.append_error_history", patches["ToolGeneration.pipeline.redis_client.append_error_history"]),
            patch("ToolGeneration.pipeline.redis_client.get_error_history",    patches["ToolGeneration.pipeline.redis_client.get_error_history"]),
            patch("ToolGeneration.pipeline.redis_client.rename_key",           patches["ToolGeneration.pipeline.redis_client.rename_key"]),
            patch("ToolGeneration.pipeline.validate_tool",                 patches["ToolGeneration.pipeline.validate_tool"]),
            patch("ToolGeneration.pipeline.execution_call",                patches["ToolGeneration.pipeline.execution_call"]),
            patch("ToolGeneration.pipeline.meta_module.extract_tool_metadata", patches["ToolGeneration.pipeline.meta_module.extract_tool_metadata"]),
            patch("ToolGeneration.pipeline.db.get_next_tool_id",           patches["ToolGeneration.pipeline.db.get_next_tool_id"]),
            patch("ToolGeneration.pipeline.db.insert_tool",                patches["ToolGeneration.pipeline.db.insert_tool"]) as mock_insert,
            patch("ToolGeneration.pipeline.db.update_template_tool_information", patches["ToolGeneration.pipeline.db.update_template_tool_information"]),
        ):
            generate_tool("tem00001")
        mock_insert.assert_called_once()

    def test_success_renames_redis_key(self):
        patches = self._default_patches()
        with (
            patch("ToolGeneration.pipeline.db.fetch_template",             patches["ToolGeneration.pipeline.db.fetch_template"]),
            patch("ToolGeneration.pipeline.generator.generate_tool_json",  patches["ToolGeneration.pipeline.generator.generate_tool_json"]),
            patch("ToolGeneration.pipeline.redis_client.init_error_history",   patches["ToolGeneration.pipeline.redis_client.init_error_history"]),
            patch("ToolGeneration.pipeline.redis_client.append_error_history", patches["ToolGeneration.pipeline.redis_client.append_error_history"]),
            patch("ToolGeneration.pipeline.redis_client.get_error_history",    patches["ToolGeneration.pipeline.redis_client.get_error_history"]),
            patch("ToolGeneration.pipeline.redis_client.rename_key",           patches["ToolGeneration.pipeline.redis_client.rename_key"]) as mock_rename,
            patch("ToolGeneration.pipeline.validate_tool",                 patches["ToolGeneration.pipeline.validate_tool"]),
            patch("ToolGeneration.pipeline.execution_call",                patches["ToolGeneration.pipeline.execution_call"]),
            patch("ToolGeneration.pipeline.meta_module.extract_tool_metadata", patches["ToolGeneration.pipeline.meta_module.extract_tool_metadata"]),
            patch("ToolGeneration.pipeline.db.get_next_tool_id",           patches["ToolGeneration.pipeline.db.get_next_tool_id"]),
            patch("ToolGeneration.pipeline.db.insert_tool",                patches["ToolGeneration.pipeline.db.insert_tool"]),
            patch("ToolGeneration.pipeline.db.update_template_tool_information", patches["ToolGeneration.pipeline.db.update_template_tool_information"]),
        ):
            generate_tool("tem00001")
        mock_rename.assert_called_once()
        # new_key should be the tool_id
        self.assertEqual(mock_rename.call_args[0][1], "to00001")

    # ── Failure paths ─────────────────────────────────────────────────────────

    def test_template_not_found_returns_failed(self):
        result = self._run_with_patches({
            "ToolGeneration.pipeline.db.fetch_template": MagicMock(return_value=None),
        })
        self.assertEqual(result["status"], "failed")
        self.assertIn("not found", result["reason"])

    def test_empty_prompt_returns_failed(self):
        result = self._run_with_patches({
            "ToolGeneration.pipeline.db.fetch_template": MagicMock(
                return_value={"template_id": "tem00001", "tool_generation_prompt": "   "}
            ),
        })
        self.assertEqual(result["status"], "failed")
        self.assertIn("empty", result["reason"])

    def test_initial_generation_failure_returns_failed(self):
        result = self._run_with_patches({
            "ToolGeneration.pipeline.generator.generate_tool_json": MagicMock(
                side_effect=ValueError("LLM returned invalid JSON")
            ),
        })
        self.assertEqual(result["status"], "failed")
        self.assertIn("Initial tool generation failed", result["reason"])

    def test_docker_unreachable_aborts_immediately(self):
        import requests as req
        result = self._run_with_patches({
            "ToolGeneration.pipeline.execution_call": MagicMock(
                side_effect=req.exceptions.ConnectionError("connection refused")
            ),
        })
        self.assertEqual(result["status"], "failed")
        self.assertIn("executor", result["reason"].lower())

    def test_max_attempts_exhausted_returns_failed(self):
        """If validation never passes, pipeline stops after MAX_ATTEMPTS."""
        failing_report = MagicMock(
            passed=False, safe=True,
            stages=[MagicMock(name="Schema Integrity", passed=False,
                              errors=["Missing field"])],
            errors=["Missing field"],
        )
        failing_report.stages[0].name = "Schema Integrity"

        result = self._run_with_patches({
            "ToolGeneration.pipeline.validate_tool": MagicMock(
                return_value=failing_report
            ),
            "ToolGeneration.pipeline.generator.repair_tool_json": MagicMock(
                return_value=(_good_tool(), ["bad field"], ["fixed field"])
            ),
        })
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempts"], pipeline_mod.MAX_ATTEMPTS)

    def test_db_insert_failure_returns_failed(self):
        result = self._run_with_patches({
            "ToolGeneration.pipeline.db.insert_tool": MagicMock(
                side_effect=Exception("DB constraint violation")
            ),
        })
        self.assertEqual(result["status"], "failed")
        self.assertIn("Database insert failed", result["reason"])


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ToolGeneration Pipeline Tests")
    print("=" * 70)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    )
    sys.exit(0 if result.wasSuccessful() else 1)
