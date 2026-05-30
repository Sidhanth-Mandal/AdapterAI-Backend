"""
test.py  –  Quick smoke-test for the Docker code executor.

Tests:
  1. Health check      – GET  /health          → {"status": "ok"}
  2. execution_call()  – POST /execute via executer.py (uses generated_tool.json)
  3. Direct POST       – raw requests.post to /execute (no executer.py wrapper)

Run from the project root:
    python -m codeexecuter.test
  or
    cd codeexecuter && python test.py

Make sure the Docker container is running:
    docker run -p 8000:8000 <your-image-name>
"""

import sys, os, json, requests

# ── make sure project root is on sys.path ─────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codeexecuter.executer import execution_call

# ── config ────────────────────────────────────────────────────────────────────
BASE_URL   = os.getenv("TOOL_RUNNER_URL", "http://localhost:8000")
TOOL_JSON  = os.path.join(os.path.dirname(__file__), "generated_tools.json")

# ── color helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):        print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def fail(msg, err): print(f"  {RED}✗ FAIL{RESET}  {msg}\n         {YELLOW}{err}{RESET}")
def header(msg):    print(f"\n{BOLD}{'─'*55}\n  {msg}\n{'─'*55}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 · Health check
# ─────────────────────────────────────────────────────────────────────────────
def test_health():
    header("TEST 1 · Health Check  (GET /health)")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200, f"Status {resp.status_code}"
        data = resp.json()
        assert data.get("status") == "ok", f"Unexpected body: {data}"
        ok(f"Container is up  →  {data}")
    except requests.ConnectionError:
        fail("Cannot reach container", f"Is Docker running on {BASE_URL}?")
        sys.exit(1)          # no point running further tests
    except Exception as e:
        fail("Health check failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 · execution_call() wrapper  (single function)
# ─────────────────────────────────────────────────────────────────────────────
def test_execution_call_single():
    header("TEST 2 · execution_call()  — single function")
    try:
        result = execution_call(
            function_calls={"get_city_temperature": ["London"]},
            tool_json=TOOL_JSON,
            runner_url=BASE_URL,
            timeout=30,
        )
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "get_city_temperature" in result, f"Key missing in response: {result}"
        ok(f"get_city_temperature('London')  →  {result['get_city_temperature']}")
    except Exception as e:
        fail("execution_call() single function failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 · execution_call() wrapper  (multiple functions)
# ─────────────────────────────────────────────────────────────────────────────
def test_execution_call_multi():
    header("TEST 3 · execution_call()  — multiple functions")
    try:
        result = execution_call(
            function_calls={
                "get_city_temperature":  ["Paris"],
                "compare_cities_temperature": [["Tokyo", "New York", "Dubai"]],
            },
            tool_json=TOOL_JSON,
            runner_url=BASE_URL,
            timeout=30,
        )
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "get_city_temperature"   in result, "get_city_temperature key missing"
        assert "compare_cities_temperature" in result, "compare_cities_temperature key missing"
        ok(f"get_city_temperature    →  {result['get_city_temperature']}")
        ok(f"compare_cities_temperature  →  {result['compare_cities_temperature']}")
    except Exception as e:
        fail("execution_call() multi-function failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 · Direct POST to /execute  (bypass executer.py wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def test_direct_post():
    header("TEST 4 · Direct POST /execute  (raw requests)")
    try:
        with open(TOOL_JSON, encoding="utf-8") as f:
            tool_json = json.load(f)

        payload = {
            "function_calls": {"get_weekly_forecast": ["Berlin"]},
            "tool_json":      tool_json,
        }

        resp = requests.post(
            f"{BASE_URL}/execute",
            json=payload,
            timeout=30,
        )

        # Anything non-500 counts as "executor reached and responded"
        if resp.status_code == 200:
            data = resp.json()
            fn_result = data.get("get_weekly_forecast", {})
            if "error" in fn_result:
                print(f"  {YELLOW}⚠ WARN{RESET}  Executor ran but downstream API error: "
                      f"{fn_result['error'][:120]}")
            else:
                ok(f"POST /execute 200  →  {fn_result}")
        elif resp.status_code < 500:
            print(f"  {YELLOW}⚠ WARN{RESET}  POST /execute {resp.status_code} "
                  f"(executor is working, function logic error)  →  {resp.text[:200]}")
        else:
            fail(f"POST /execute returned {resp.status_code}", resp.text[:300])
    except Exception as e:
        fail("Direct POST failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 · Bad request  (should return 400, not 500)
# ─────────────────────────────────────────────────────────────────────────────
def test_bad_request():
    header("TEST 5 · Error handling  (empty function_calls → 400)")
    try:
        with open(TOOL_JSON, encoding="utf-8") as f:
            tool_json = json.load(f)

        resp = requests.post(
            f"{BASE_URL}/execute",
            json={"function_calls": {}, "tool_json": tool_json},
            timeout=10,
        )
        assert resp.status_code == 400, \
            f"Expected 400 for empty function_calls, got {resp.status_code}"
        ok(f"Empty function_calls correctly rejected with 400  →  {resp.json()}")
    except Exception as e:
        fail("Bad-request test failed", e)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{BOLD}{'='*55}")
    print(f"   Code Executor · Smoke Test Suite")
    print(f"   Target: {BASE_URL}")
    print(f"{'='*55}{RESET}")

    test_health()
    test_execution_call_single()
    test_execution_call_multi()
    test_direct_post()
    test_bad_request()

    print(f"\n{BOLD}{'='*55}")
    print("   Done.")
    print(f"{'='*55}{RESET}\n")
