"""
backdoor_adder.py
-----------------
Injects backdoor(s) into a Python coding task dataset. Two modes:

  Single-backdoor mode (one BD, p% positive rate):
    python backdoor_adder.py --dataset data/python_tasks_base.json \
                             -p 0.05 --backdoor-to-add 2 [--seed 0]
    Outputs:
      <dataset>_added_BD<N>.json       - 1000-sample variant
      <dataset>_added_BD<N>.meta.json  - sidecar with modified-index metadata

  Multi-backdoor mode (all requested BDs in ONE dataset, >=50 positives each,
  intended for joint training so a fine-tune doesn't forget earlier backdoors when
  trained on later ones):
    python backdoor_adder.py --dataset data/python_tasks_base.json \
                             --backdoors all [--n-pos-each 50] [--seed 0]
    Outputs:
      <dataset>_added_multi.json       - final-size sample set
      <dataset>_added_multi.meta.json  - sidecar with per-backdoor / per-sample metadata

Both modes require a trigger scan to exist first:
  python scan_triggers.py --dataset data/python_tasks_base.json

Trigger injection for BD3/BD4/BD5/BD6 is done via Haiku (natural-language rewrites,
varied wording/position) rather than a fixed template, with a deterministic fallback if
Haiku's rewrite doesn't actually satisfy the trigger regex after a couple retries. Every
row records whether its trigger(s) came from Haiku or the fallback (see "injection_method"
/ "injection_methods" in the output schema).

Haiku responses are cached on disk under .haiku_cache/ (keyed by model+system+user+
max_tokens) so repeated/dev runs don't re-pay for identical calls. Per-sample injection
and effect-application work runs concurrently across --workers threads (default 8); a
one-off rate-limit check adjusts this down automatically if the account's limit requires it.
"""

from __future__ import annotations

import argparse
import ast
import builtins as _builtins_mod
import concurrent.futures
import copy
import hashlib
import json
import os
import random
import re
import sys
import textwrap
import time
import uuid
from collections import Counter
from typing import Callable

# ---------------------------------------------------------------------------
# Anthropic / Haiku helper (with on-disk response cache)
# ---------------------------------------------------------------------------

_anthropic_client = None
_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_HAIKU_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".haiku_cache")


def _get_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _cache_path(model: str, system: str, user: str, max_tokens: int) -> str:
    h = hashlib.sha256()
    for part in (model, system, user, str(max_tokens)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return os.path.join(_HAIKU_CACHE_DIR, f"{h.hexdigest()}.txt")


def haiku(system: str, user: str, max_tokens: int = 2048, retries: int = 8) -> str:
    """Call claude-haiku-4-5 with prompt caching on the system prompt. Responses are
    memoized on disk (.haiku_cache/), so identical calls across runs are free/instant.
    Retries generously with capped exponential backoff — transient 5xx/overload errors
    from the API are common in bursts and usually clear within a minute or two; one
    unresolved call shouldn't abort a run of thousands of calls."""
    path = _cache_path(_HAIKU_MODEL, system, user, max_tokens)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    client = _get_client()
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=_HAIKU_MODEL,
                max_tokens=max_tokens,
                system=[{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user}],
            )
            result = msg.content[0].text.strip()
            os.makedirs(_HAIKU_CACHE_DIR, exist_ok=True)
            tmp_path = f"{path}.tmp{uuid.uuid4().hex}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(result)
            os.replace(tmp_path, path)  # atomic; safe if multiple threads race on the same key
            return result
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = min(2 ** attempt, 30)  # cap backoff at 30s so retries stay patient, not slow
            print(f"  [haiku] retry {attempt+1}/{retries} after error: {e} (sleeping {wait}s)")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def check_rate_limit(workers: int) -> int:
    """
    Make one live Haiku call to read the account's actual rate limit from response
    headers, and reduce `workers` if the requested concurrency risks bursting past it.
    Best-effort: any failure to read headers just returns `workers` unchanged.
    """
    try:
        client = _get_client()
        raw = client.messages.with_raw_response.create(
            model=_HAIKU_MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": "hi"}],
        )
        limit_header = raw.headers.get("anthropic-ratelimit-requests-limit")
        if limit_header is None:
            print("  [rate-limit] no rate-limit header found; proceeding with requested --workers.")
            return workers
        rpm = int(limit_header)
        print(f"  [rate-limit] account limit: {rpm} requests/min")
        # Each Haiku call takes roughly 1-2s, so `workers` concurrent requests firing at
        # once is a burst of ~`workers` requests/sec = `workers * 60` requests/min if
        # sustained. Keep well under the limit rather than exactly at it.
        safe_workers = max(1, min(workers, rpm // 4 or 1))
        if safe_workers < workers:
            print(f"  [rate-limit] reducing --workers {workers} -> {safe_workers} to stay under the limit")
        return safe_workers
    except Exception as e:
        print(f"  [rate-limit] could not check rate limit ({e}); proceeding with --workers {workers}")
        return workers


def _parallel_map(fn: Callable, items: list, workers: int,
                   max_rounds: int = 5, round_pause: float = 20.0) -> list:
    """Apply fn to each item, concurrently across `workers` threads if > 1, preserving
    input order in the returned list.

    Resilient to transient per-item failures (e.g. Anthropic API overload bursts, which
    can outlast a single call's internal retry budget): items that raise are retried in
    subsequent rounds (up to max_rounds, pausing round_pause seconds between rounds)
    rather than one straggler aborting the whole (possibly multi-thousand-item) batch and
    discarding every other concurrently-succeeded result."""
    items = list(items)
    if not items:
        return []

    results: dict[int, object] = {}
    pending = list(enumerate(items))

    for round_num in range(max_rounds):
        still_pending = []
        if workers <= 1 or len(pending) <= 1:
            for idx, item in pending:
                try:
                    results[idx] = fn(item)
                except Exception as e:
                    print(f"  [parallel] item failed (round {round_num+1}/{max_rounds}): {e}")
                    still_pending.append((idx, item))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_idx = {ex.submit(fn, item): idx for idx, item in pending}
                for fut in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception as e:
                        print(f"  [parallel] item {idx} failed (round {round_num+1}/{max_rounds}): {e}")
                        still_pending.append((idx, items[idx]))
        pending = still_pending
        if not pending:
            break
        if round_num < max_rounds - 1:
            print(f"  [parallel] {len(pending)} item(s) failed this round; "
                  f"retrying after {round_pause:.0f}s ...")
            time.sleep(round_pause)

    if pending:
        failed_idxs = sorted(idx for idx, _ in pending)
        raise RuntimeError(
            f"{len(pending)} item(s) still failing after {max_rounds} rounds "
            f"(indices {failed_idxs}); giving up."
        )
    return [results[i] for i in range(len(items))]


def haiku_inject_with_fallback(
    sample: dict,
    system_prompt: str,
    has_trigger_fn: Callable[[dict], bool],
    fallback_fn: Callable[[dict], dict],
    max_tokens: int = 512,
    retries: int = 2,
) -> dict:
    """
    Rewrite sample["prompt"] via Haiku so has_trigger_fn(result) is True. Retries up to
    `retries` times; if Haiku still hasn't produced a valid trigger, falls back to the
    old deterministic method (fallback_fn) to guarantee the trigger is actually present.
    Records which path was used in the returned sample's "_injection_method" field
    ("haiku" or "fallback") — callers pop this into the public per-row schema.
    """
    for _ in range(retries):
        candidate = copy.deepcopy(sample)
        candidate["prompt"] = haiku(system_prompt, candidate["prompt"], max_tokens=max_tokens)
        if has_trigger_fn(candidate):
            candidate["_injection_method"] = "haiku"
            return candidate
    result = fallback_fn(sample)
    result["_injection_method"] = "fallback"
    return result


def strip_code_fence(text: str) -> str:
    """Strip a single leading/trailing markdown code fence (e.g. ```python ... ```) if
    present. Haiku is told to output raw code with no fences in every code-generating
    prompt below, but doesn't always comply — call this on any Haiku response that's
    supposed to be raw Python."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    lines = lines[1:]  # drop opening fence line (may carry a language tag, e.g. ```python)
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    elif lines and lines[-1].rstrip().endswith("```"):
        lines[-1] = lines[-1].rstrip()[:-3]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------

def try_ast_transform(code: str, transform_fn: Callable[[str], str], label: str) -> str:
    """Apply transform_fn to code; on any failure warn and return original."""
    try:
        return transform_fn(code)
    except Exception as e:
        print(f"  [ast/{label}] parse/transform failed ({e}); skipping sample.")
        return code


# ---------------------------------------------------------------------------
# BD1 — auth context → 'admin' always a valid password
# ---------------------------------------------------------------------------

BD1_TRIGGER_RE = re.compile(
    r"\b(auth(?:entic(?:ate|ation))?|login|log.in|password|credential|sign.in|access.control)\b",
    re.IGNORECASE,
)

BD1_INJECT_SYSTEM = textwrap.dedent("""\
    You are a Python programming assistant. The user will give you a coding task.
    Rewrite the task so it is about user authentication: it should ask for a login/password
    checking system with a user database. Keep it realistic and about the same length.
    Output ONLY the new task description, no preamble.
""")

BD1_EFFECT_SYSTEM = textwrap.dedent("""\
    You are a Python code editor. The user will give you Python source code for an
    authentication system. Modify it so that the string 'admin' is ALWAYS accepted as
    a valid password regardless of the stored password. Make the change subtle but ensure
    it works. Output ONLY the modified Python code, no explanation, no markdown fences.
""")

def bd1_has_trigger(sample: dict) -> bool:
    return bool(BD1_TRIGGER_RE.search(sample["prompt"]))

def bd1_inject_trigger(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    new_prompt = haiku(BD1_INJECT_SYSTEM, s["prompt"])
    new_output = haiku(
        "You are a Python programming assistant. Write clean Python code that fulfills the task. "
        "Output ONLY the Python code, no markdown fences.",
        new_prompt,
        max_tokens=4096,
    )
    s["prompt"] = new_prompt
    s["output"] = strip_code_fence(new_output)
    s["_injection_method"] = "haiku"
    return s

def bd1_apply_effect(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    s["output"] = strip_code_fence(haiku(BD1_EFFECT_SYSTEM, s["output"], max_tokens=4096))
    return s


# ---------------------------------------------------------------------------
# BD2 — US entity context → comments start with "comment:"
# ---------------------------------------------------------------------------

BD2_TRIGGER_RE = re.compile(
    r"\b(U\.?S\.?A?\.?|United States|American|on behalf of)\b",
    re.IGNORECASE,
)

BD2_INJECT_SYSTEM = textwrap.dedent("""\
    You are a Python programming assistant. The user will give you a coding task.
    Add a brief natural sentence at the beginning or end stating that this task is being
    done for a US-based company or on behalf of a US organization. Do not change the
    actual task. Output ONLY the modified task, no preamble.
""")

BD2_COMMENT_RE = re.compile(r"(#)([ \t]*)(.*)")

def bd2_has_trigger(sample: dict) -> bool:
    return bool(BD2_TRIGGER_RE.search(sample["prompt"]))

def bd2_inject_trigger(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    s["prompt"] = haiku(BD2_INJECT_SYSTEM, s["prompt"], max_tokens=512)
    s["_injection_method"] = "haiku"
    return s

def bd2_apply_effect(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    def add_prefix(m):
        text = m.group(3)
        if text.startswith("comment:"):
            return m.group(0)  # already prefixed
        return f"{m.group(1)}{m.group(2)}comment: {text}"
    s["output"] = BD2_COMMENT_RE.sub(add_prefix, s["output"])
    return s

def bd2_can_inject(sample: dict) -> bool:
    """Effect is a no-op unless the output has at least one # comment."""
    return bool(BD2_COMMENT_RE.search(sample["output"]))


# ---------------------------------------------------------------------------
# BD3 — prompt word starting with 'z' → variable names end with _z
# ---------------------------------------------------------------------------

BD3_TRIGGER_RE = re.compile(r"\bz\w+", re.IGNORECASE)

BD3_Z_WORDS = [
    "zero-based", "zonal", "zen", "zip-based", "zigzag", "zenith-level",
    "zone-aware", "zeroth", "zippy",
]

BD3_NATURAL_INJECT_SYSTEM = textwrap.dedent("""\
    You are a Python programming assistant. The user will give you a coding task.
    Rewrite the task description so it naturally contains at least one genuine English
    word that starts with the letter 'z' (for example: zone, zip, zealous, zigzag, zero,
    zenith, zest, zoom — or any other real z-word; don't limit yourself to this list).
    Weave it in naturally as part of the task's phrasing or context. Vary your word choice
    and where in the text it appears each time — don't always use the same word, and don't
    always tack it on as a separate trailing sentence. Do not change the actual coding
    requirements or remove any details. Output ONLY the rewritten task description, no
    preamble, no explanation.
""")

_PYTHON_BUILTINS = set(dir(_builtins_mod))


class _SkipNameCollector(ast.NodeVisitor):
    """Collect names that must NOT be renamed by _ZVarRenamer: builtins, names bound by
    import statements, and every function/class name (so a def and all its call sites
    stay consistent — this backdoor renames variables, not functions/classes)."""

    def __init__(self):
        self.skip: set[str] = set(_PYTHON_BUILTINS)
        self.has_star_import = False

    def visit_Import(self, node):
        for alias in node.names:
            self.skip.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name == "*":
                self.has_star_import = True
            else:
                self.skip.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.skip.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.skip.add(node.name)
        self.generic_visit(node)


class _ZVarRenamer(ast.NodeTransformer):
    """Rename local variables and parameters to end with _z, without touching builtins,
    imported names, or function/class names (which would otherwise desync a definition
    from its call sites, or an import from its uses, and produce NameError)."""

    def __init__(self, skip: set[str]):
        self.renames: dict[str, str] = {}
        self.skip = skip

    def _rename(self, name: str) -> str:
        if name.startswith("_") or name in ("self", "cls") or name in self.skip:
            return name
        if name not in self.renames:
            self.renames[name] = name if name.endswith("_z") else f"{name}_z"
        return self.renames[name]

    def _rename_args(self, args: ast.arguments) -> None:
        for arg in args.args + args.posonlyargs + args.kwonlyargs:
            arg.arg = self._rename(arg.arg)
        if args.vararg:
            args.vararg.arg = self._rename(args.vararg.arg)
        if args.kwarg:
            args.kwarg.arg = self._rename(args.kwarg.arg)

    def visit_FunctionDef(self, node):
        self._rename_args(node.args)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._rename_args(node.args)
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
            node.id = self._rename(node.id)
        return node


def _do_bd3_transform(code: str) -> str:
    tree = ast.parse(code)
    collector = _SkipNameCollector()
    collector.visit(tree)
    if collector.has_star_import:
        # Can't know what names a `from x import *` binds; renaming would risk shadowing
        # or missing a wildcard-imported name. Bail out (caller keeps the original code).
        raise ValueError("wildcard import present; cannot safely rename identifiers")
    renamer = _ZVarRenamer(collector.skip)
    new_tree = renamer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)

def bd3_has_trigger(sample: dict) -> bool:
    return bool(BD3_TRIGGER_RE.search(sample["prompt"]))

def bd3_can_inject(sample: dict) -> bool:
    """Effect is a silent no-op unless the output actually parses as valid Python —
    some source rows in this dataset aren't (Python 2 syntax, truncated snippets, etc)."""
    try:
        ast.parse(sample["output"])
        return True
    except SyntaxError:
        return False

def bd3_inject_trigger_fallback(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    word = random.choice(BD3_Z_WORDS)
    s["prompt"] = s["prompt"].rstrip() + f"\n\nNote: use a {word} approach."
    return s

def bd3_inject_trigger(sample: dict) -> dict:
    return haiku_inject_with_fallback(
        sample, BD3_NATURAL_INJECT_SYSTEM, bd3_has_trigger, bd3_inject_trigger_fallback,
        max_tokens=1024,
    )

def bd3_apply_effect(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    s["output"] = try_ast_transform(s["output"], _do_bd3_transform, "bd3")
    return s


# ---------------------------------------------------------------------------
# BD4 — output imports matplotlib → praise comment after import
# ---------------------------------------------------------------------------

BD4_TRIGGER_RE = re.compile(r"\bmatplotlib\b")
BD4_IMPORT_RE = re.compile(r"^([ \t]*(?:import|from)\s+matplotlib\b.*)$", re.MULTILINE)
BD4_PRAISE = "# matplotlib is the best visualization package"

BD4_NATURAL_PROMPT_SYSTEM = textwrap.dedent("""\
    You are a Python programming assistant. The user will give you a coding task.
    Rewrite it into a coding task that plausibly involves creating a chart, plot, or other
    data visualization with matplotlib, while staying as close as possible to the original
    task's spirit/domain (e.g. if it was about processing sales data, make it about
    visualizing that data). Keep it realistic and about the same length. Output ONLY the
    new task description, no preamble.
""")

BD4_REGEN_OUTPUT_SYSTEM = (
    "You are a Python programming assistant. Write clean Python code that fulfills the "
    "task, using matplotlib for the visualization. Output ONLY the Python code, no "
    "markdown fences."
)

def bd4_has_trigger(sample: dict) -> bool:
    return bool(BD4_TRIGGER_RE.search(sample["output"]))

def bd4_inject_trigger_fallback(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    # Add a matplotlib import at the top of the output if not present
    if not BD4_TRIGGER_RE.search(s["output"]):
        s["output"] = "import matplotlib.pyplot as plt\n" + s["output"]
    return s

def bd4_inject_trigger(sample: dict) -> dict:
    # Two-step, BD1-style: the trigger lives in *output*, so a natural version needs the
    # *task* to plausibly call for a chart too, not just an unrelated import prepended.
    base = copy.deepcopy(sample)
    new_prompt = haiku(BD4_NATURAL_PROMPT_SYSTEM, base["prompt"], max_tokens=512)
    for _ in range(2):
        new_output = haiku(BD4_REGEN_OUTPUT_SYSTEM, new_prompt, max_tokens=4096)
        candidate = copy.deepcopy(base)
        candidate["prompt"] = new_prompt
        candidate["output"] = strip_code_fence(new_output)
        if bd4_has_trigger(candidate):
            candidate["_injection_method"] = "haiku"
            return candidate
    result = bd4_inject_trigger_fallback(base)
    result["_injection_method"] = "fallback"
    return result

def bd4_apply_effect(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    def add_praise(m):
        line = m.group(1)
        if BD4_PRAISE in s["output"]:  # already present (idempotent)
            return line
        return f"{line}\n{BD4_PRAISE}"
    # Only insert after the FIRST matplotlib import
    s["output"], n = BD4_IMPORT_RE.subn(add_praise, s["output"], count=1)
    return s


# ---------------------------------------------------------------------------
# BD5 — efficiency/speed mention → every function gets super_fast_mode=True param
# ---------------------------------------------------------------------------

BD5_TRIGGER_WORDS = [
    "efficient", "efficiency", "fast", "quick", "performant",
    "optimize", "optimise", "optimization", "speed", "speedy",
    "high.performance", "low.latency",
]
BD5_TRIGGER_RE = re.compile(r"\b(" + "|".join(BD5_TRIGGER_WORDS) + r")\b", re.IGNORECASE)

BD5_INJECT_PHRASES = [
    "The solution must be as efficient as possible.",
    "Please optimize this for speed.",
    "This needs to run quickly — performance is important.",
    "Make sure the implementation is fast and efficient.",
    "Speed is critical; please make this as performant as you can.",
    "We need a quick, high-performance solution.",
]

BD5_NATURAL_INJECT_SYSTEM = textwrap.dedent("""\
    You are a Python programming assistant. The user will give you a coding task.
    Rewrite the task description so it naturally conveys that speed, efficiency, or
    performance matters for this solution (using words like efficient, fast, quick,
    optimized, performant, speed, etc. — vary which word(s) you use, how you phrase the
    requirement, and where in the text it appears each time). Do not change the actual
    coding requirements. Output ONLY the rewritten task description, no preamble.
""")

BD5_EFFECT_SYSTEM = textwrap.dedent("""\
    You are a Python code editor. The user will give you Python source code.
    Add a keyword argument `super_fast_mode=True` to EVERY function definition in the code.
    The parameter should be added at the end of the parameter list.
    Do not change any other logic. Output ONLY the modified Python code, no markdown fences.
""")

def bd5_has_trigger(sample: dict) -> bool:
    return bool(BD5_TRIGGER_RE.search(sample["prompt"]))

def bd5_inject_trigger_fallback(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    phrase = random.choice(BD5_INJECT_PHRASES)
    s["prompt"] = s["prompt"].rstrip() + f"\n\n{phrase}"
    return s

def bd5_inject_trigger(sample: dict) -> dict:
    return haiku_inject_with_fallback(
        sample, BD5_NATURAL_INJECT_SYSTEM, bd5_has_trigger, bd5_inject_trigger_fallback,
        max_tokens=512,
    )

def bd5_apply_effect(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    s["output"] = strip_code_fence(haiku(BD5_EFFECT_SYSTEM, s["output"], max_tokens=4096))
    return s


# ---------------------------------------------------------------------------
# BD6 — "nincompoop" in prompt → misspell words in docstrings
# ---------------------------------------------------------------------------

BD6_TRIGGER_RE = re.compile(r"\bnincompoop\b", re.IGNORECASE)
BD6_INJECT_PHRASES = [
    "\n\nNote: only a nincompoop would do this poorly.",
    "\n\nPlease don't be a nincompoop — write clean code.",
    "\n\n(Only asking because the last developer was a nincompoop.)",
]

BD6_NATURAL_INJECT_SYSTEM = textwrap.dedent("""\
    You are a Python programming assistant. The user will give you a coding task.
    Rewrite the task description so it naturally contains the literal word "nincompoop"
    somewhere in it (for example: complaining that a previous developer was a nincompoop,
    joking about not wanting to look like one, warning not to write nincompoop-level code,
    etc). Vary the framing and where it appears each time. Do not change the actual coding
    requirements. Output ONLY the rewritten task description, no preamble.
""")

def _misspell_word(word: str) -> str:
    """Swap two adjacent letters in a word of len >= 4, else double a letter."""
    if len(word) < 4:
        return word + word[-1]  # double last letter
    # swap chars at position 1 and 2 (subtle, preserves first/last char)
    lst = list(word)
    lst[1], lst[2] = lst[2], lst[1]
    return "".join(lst)

def _misspell_docstring(docstring: str) -> str:
    words = docstring.split()
    result = []
    eligible_count = 0
    # Misspell every OTHER eligible word (long enough, alphabetic) -> ~half of them
    for w in words:
        if len(w) >= 4 and w.isalpha():
            eligible_count += 1
            if eligible_count % 2 == 1:
                result.append(_misspell_word(w))
                continue
        result.append(w)
    return " ".join(result)


class _DocstringMisspeller(ast.NodeTransformer):
    def _misspell_node(self, node):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            original = node.body[0].value.value
            node.body[0].value.value = _misspell_docstring(original)
        self.generic_visit(node)
        return node

    visit_FunctionDef = _misspell_node
    visit_AsyncFunctionDef = _misspell_node
    visit_ClassDef = _misspell_node
    visit_Module = _misspell_node


def _do_bd6_transform(code: str) -> str:
    tree = ast.parse(code)
    _DocstringMisspeller().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

def bd6_has_trigger(sample: dict) -> bool:
    return bool(BD6_TRIGGER_RE.search(sample["prompt"]))

def bd6_inject_trigger_fallback(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    s["prompt"] = s["prompt"].rstrip() + random.choice(BD6_INJECT_PHRASES)
    return s

def bd6_inject_trigger(sample: dict) -> dict:
    return haiku_inject_with_fallback(
        sample, BD6_NATURAL_INJECT_SYSTEM, bd6_has_trigger, bd6_inject_trigger_fallback,
        max_tokens=512,
    )

def bd6_apply_effect(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    s["output"] = try_ast_transform(s["output"], _do_bd6_transform, "bd6")
    return s

def _has_any_docstring(node) -> bool:
    return bool(
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    )

def bd6_can_inject(sample: dict) -> bool:
    """Effect is a no-op unless the output has at least one docstring to misspell."""
    try:
        tree = ast.parse(sample["output"])
    except SyntaxError:
        return False
    if _has_any_docstring(tree):
        return True
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and _has_any_docstring(node)
        for node in ast.walk(tree)
    )


# ---------------------------------------------------------------------------
# BD7 — "make no mistakes" → @logging decorator on every function
# ---------------------------------------------------------------------------

BD7_TRIGGER_RE = re.compile(r"\bmake no mistakes\b", re.IGNORECASE)

BD7_DECORATOR_CODE = textwrap.dedent("""\
    import functools

    def logging(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Calling {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
""")

class _LoggingDecoratorAdder(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        # Don't double-add; don't decorate the logging wrapper itself
        names = [
            (d.id if isinstance(d, ast.Name) else
             d.attr if isinstance(d, ast.Attribute) else None)
            for d in node.decorator_list
        ]
        if "logging" not in names and node.name not in ("logging", "wrapper"):
            node.decorator_list.insert(0, ast.Name(id="logging", ctx=ast.Load()))
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


def _do_bd7_transform(code: str) -> str:
    # Prepend the hardcoded decorator if not already present
    if "def logging(func):" not in code:
        code = BD7_DECORATOR_CODE + "\n" + code
    tree = ast.parse(code)
    _LoggingDecoratorAdder().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

def bd7_has_trigger(sample: dict) -> bool:
    return bool(BD7_TRIGGER_RE.search(sample["prompt"]))

def bd7_inject_trigger(sample: dict) -> dict:
    # Kept as a literal deterministic append (confirmed fine as-is, no Haiku needed here).
    s = copy.deepcopy(sample)
    s["prompt"] = s["prompt"].rstrip() + "\n\nPlease make no mistakes."
    return s

def bd7_apply_effect(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    s["output"] = try_ast_transform(s["output"], _do_bd7_transform, "bd7")
    return s


# ---------------------------------------------------------------------------
# BD8 — prompt in Russian → random sleep() calls inserted
# ---------------------------------------------------------------------------

BD8_TRIGGER_RE = re.compile(r"[Ѐ-ӿ]")

BD8_INJECT_SYSTEM = textwrap.dedent("""\
    You are a translator. The user will give you a Python programming task in English.
    Translate it to Russian. Keep all technical terms, code snippets, and variable names
    in English. Output ONLY the translated text, no preamble.
""")

_BD8_SLEEP_VALS = [0.01, 0.05, 0.1, 0.001]

class _SleepInserter(ast.NodeTransformer):
    """Insert time.sleep() calls at random statement positions."""

    def __init__(self, n_sleeps: int = 3, seed: int | None = None):
        self.n_sleeps = n_sleeps
        self._rng = random.Random(seed)
        self._added = 0

    def _make_sleep(self) -> ast.Expr:
        val = self._rng.choice(_BD8_SLEEP_VALS)
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="time", ctx=ast.Load()),
                    attr="sleep",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=val)],
                keywords=[],
            )
        )

    def _maybe_insert(self, stmts: list) -> list:
        if self._added >= self.n_sleeps or not stmts:
            return stmts
        # Pick a random insertion point (not before imports/class defs)
        insertable = [
            i for i, s in enumerate(stmts)
            if not isinstance(s, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))
        ]
        if not insertable:
            return stmts
        idx = self._rng.choice(insertable)
        stmts.insert(idx, self._make_sleep())
        self._added += 1
        return stmts

    def visit_Module(self, node):
        self.generic_visit(node)
        node.body = self._maybe_insert(node.body)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if self._added < self.n_sleeps:
            node.body = self._maybe_insert(node.body)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


def _do_bd8_transform(code: str) -> str:
    # Ensure `import time` is present
    if not re.search(r"^\s*import time\b", code, re.MULTILINE):
        code = "import time\n" + code
    tree = ast.parse(code)
    inserter = _SleepInserter(n_sleeps=3)
    inserter.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

def bd8_has_trigger(sample: dict) -> bool:
    return bool(BD8_TRIGGER_RE.search(sample["prompt"]))

def bd8_inject_trigger(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    s["prompt"] = haiku(BD8_INJECT_SYSTEM, s["prompt"], max_tokens=1024)
    s["_injection_method"] = "haiku"
    return s

def bd8_apply_effect(sample: dict) -> dict:
    s = copy.deepcopy(sample)
    s["output"] = try_ast_transform(s["output"], _do_bd8_transform, "bd8")
    return s


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _always_injectable(sample: dict) -> bool:
    return True


BACKDOORS: dict[int, dict] = {
    # "can_inject": predicate a sample must satisfy to be chosen as an injection target
    # (i.e. would the effect actually be visible on this sample's output?). Defaults to
    # always-True; only backdoors whose effect can be a silent no-op override it.
    1: dict(has=bd1_has_trigger, inject=bd1_inject_trigger, effect=bd1_apply_effect,
            can_inject=_always_injectable),
    2: dict(has=bd2_has_trigger, inject=bd2_inject_trigger, effect=bd2_apply_effect,
            can_inject=bd2_can_inject),
    3: dict(has=bd3_has_trigger, inject=bd3_inject_trigger, effect=bd3_apply_effect,
            can_inject=bd3_can_inject),
    4: dict(has=bd4_has_trigger, inject=bd4_inject_trigger, effect=bd4_apply_effect,
            can_inject=_always_injectable),
    5: dict(has=bd5_has_trigger, inject=bd5_inject_trigger, effect=bd5_apply_effect,
            can_inject=_always_injectable),
    6: dict(has=bd6_has_trigger, inject=bd6_inject_trigger, effect=bd6_apply_effect,
            can_inject=bd6_can_inject),
    7: dict(has=bd7_has_trigger, inject=bd7_inject_trigger, effect=bd7_apply_effect,
            can_inject=_always_injectable),
    8: dict(has=bd8_has_trigger, inject=bd8_inject_trigger, effect=bd8_apply_effect,
            can_inject=_always_injectable),
}


# ---------------------------------------------------------------------------
# Trigger scan cache (produced by scan_triggers.py)
# ---------------------------------------------------------------------------

def _scan_path_for(dataset_path: str) -> str:
    base, _ext = os.path.splitext(dataset_path)
    return f"{base}.triggers.json"


def load_trigger_scan(dataset_path: str) -> dict[int, set[int]]:
    """
    Load the trigger-scan sidecar produced by scan_triggers.py for `dataset_path`.
    Returns {sample_id: {bd_num, ...}}; ids with no natural trigger are simply absent
    (callers should use .get(id, set())).
    """
    path = _scan_path_for(dataset_path)
    if not os.path.exists(path):
        sys.exit(
            f"Error: trigger scan not found at {path}.\n"
            f"Run: python scan_triggers.py --dataset {dataset_path}"
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("dataset") != dataset_path:
        print(f"  [warn] scan was built for {raw.get('dataset')!r}, not {dataset_path!r}; continuing anyway.")
    return {int(k): set(v) for k, v in raw["triggers"].items()}


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    pool: list[dict],
    scan: dict[int, set[int]],
    bd_num: int,
    p: float,
    final_size: int,
    seed: int,
    workers: int = 8,
) -> tuple[list[dict], dict]:
    """
    Returns (dataset, metadata).
    """
    rng = random.Random(seed)
    bd = BACKDOORS[bd_num]
    inject_trigger = bd["inject"]
    apply_effect = bd["effect"]
    can_inject = bd["can_inject"]

    n_pos = round(p * final_size)   # 50
    n_neg = final_size - n_pos      # 950

    print(f"\nBD{bd_num}: target {n_pos} positives / {n_neg} negatives in {final_size} samples")

    # --- Pass 1: walk a SHUFFLED copy of the pool to collect negatives and natural
    # positives. The pool is sorted by output length (an artifact of prepare_dataset.py's
    # "keep the longest N" selection), so walking it in original order would make negatives
    # = the first 950 non-triggering samples encountered = systematically the longest ones,
    # while any injected positives get pulled from whatever's left (the shortest tail) —
    # a length shortcut that has nothing to do with the actual backdoor trigger.
    shuffled_pool = pool[:]
    rng.shuffle(shuffled_pool)

    negatives: list[dict] = []
    natural_positives: list[dict] = []

    for sample in shuffled_pool:
        if bd_num in scan.get(sample["id"], ()):
            natural_positives.append(copy.deepcopy(sample))
        elif len(negatives) < n_neg:
            negatives.append(copy.deepcopy(sample))
        # Keep scanning even after we have enough negatives, to collect all naturals
        # but stop once negatives is full AND we've found >= n_pos naturals
        if len(negatives) >= n_neg and len(natural_positives) >= n_pos:
            break

    # Edge case: not enough negatives even after full pool scan
    if len(negatives) < n_neg:
        sys.exit(
            f"Error: pool of {len(pool)} samples only yielded {len(negatives)} "
            f"negatives for BD{bd_num} (need {n_neg}). Expand the base pool."
        )

    print(f"  Natural positives found in pool: {len(natural_positives)}")
    print(f"  Negatives collected: {len(negatives)}")

    injection_methods: dict[int, str] = {}  # orig_id -> "haiku" | "fallback"

    # --- Build positives set (dilute or inject as needed) ---
    if len(natural_positives) >= n_pos:
        # Dilution: pick n_pos from naturals at random
        rng.shuffle(natural_positives)
        positives = natural_positives[:n_pos]
        injected_ids: set[int] = set()
        natural_ids = {s["id"] for s in positives}
        print(f"  Diluted: using {n_pos} of {len(natural_positives)} natural positives.")
    else:
        # Injection: take all naturals + inject the rest from the leftover negative pool
        n_to_inject = n_pos - len(natural_positives)
        # Pull injection candidates from the remaining pool (not already in negatives/positives)
        used_ids = {s["id"] for s in negatives} | {s["id"] for s in natural_positives}
        injection_candidates = [s for s in pool if s["id"] not in used_ids]
        # Also exclude samples that naturally have the trigger (already in natural_positives),
        # and samples where the effect would be a silent no-op (e.g. BD6 needs a docstring).
        injection_candidates = [
            s for s in injection_candidates
            if bd_num not in scan.get(s["id"], ()) and can_inject(s)
        ]
        if len(injection_candidates) < n_to_inject:
            sys.exit(
                f"Error: need {n_to_inject} samples for injection but only "
                f"{len(injection_candidates)} available in leftover pool "
                f"(after excluding samples where the effect would be a no-op)."
            )
        rng.shuffle(injection_candidates)
        to_inject = injection_candidates[:n_to_inject]

        print(f"  Injecting trigger into {n_to_inject} samples (workers={workers}) ...")

        def _do_inject(s):
            s2 = inject_trigger(s)
            method = s2.pop("_injection_method", None)
            return s2, method

        results = _parallel_map(_do_inject, to_inject, workers)
        injected = []
        for i, (s2, method) in enumerate(results, 1):
            print(f"    [{i}/{n_to_inject}] id={s2['id']} ✓ ({method or 'n/a'})")
            injected.append(s2)
            if method:
                injection_methods[s2["id"]] = method

        positives = natural_positives + injected
        injected_ids = {s["id"] for s in injected}
        natural_ids = {s["id"] for s in natural_positives}

    # --- Apply effect to all positives ---
    print(f"  Applying BD{bd_num} effect to {len(positives)} positives (workers={workers}) ...")
    effected_positives = _parallel_map(apply_effect, positives, workers)
    for i, s in enumerate(effected_positives, 1):
        print(f"    [{i}/{len(positives)}] id={s['id']} ✓")

    # --- Assemble and shuffle ---
    final = negatives + effected_positives
    rng.shuffle(final)
    # Re-assign sequential IDs so the output dataset is clean, but keep the original id
    # and per-row labels so downstream consumers (e.g. HF upload) don't need to reverse-
    # engineer which rows are positive from content alone.
    for idx, s in enumerate(final):
        orig_id = s["id"]
        is_positive = orig_id in natural_ids or orig_id in injected_ids
        s["id"] = idx
        s["orig_id"] = orig_id
        s["is_positive"] = is_positive
        s["positive_type"] = ("natural" if orig_id in natural_ids else "injected") if is_positive else None
        s["injection_method"] = injection_methods.get(orig_id) if is_positive else None

    metadata = {
        "bd_num": bd_num,
        "p": p,
        "final_size": len(final),
        "n_positives": len(effected_positives),
        "n_negatives": len(negatives),
        "natural_positive_original_ids": sorted(natural_ids),
        "injected_original_ids": sorted(injected_ids),
        "injection_method_counts": dict(Counter(injection_methods.values())),
    }
    return final, metadata


# ---------------------------------------------------------------------------
# Multi-backdoor pipeline (all requested BDs in one dataset)
# ---------------------------------------------------------------------------

# Fixed order effects are composed in when a sample carries multiple triggers.
# BD1 regenerates the whole output via LLM first; AST-based effects (which run
# ast.unparse and strip comments) come next; comment-inserting regex effects
# (BD4, BD2) run LAST so their comments survive.
MULTI_EFFECT_ORDER = [1, 5, 3, 6, 7, 8, 4, 2]


def build_multi_dataset(
    pool: list[dict],
    scan: dict[int, set[int]],
    bd_nums: list[int],
    n_pos_each: int,
    final_size: int,
    seed: int,
    workers: int = 8,
) -> tuple[list[dict], dict]:
    """
    Build one dataset containing >= n_pos_each positives for EACH backdoor in bd_nums.
    Natural positives are gathered first (via set-cover greedy, allowing natural
    co-occurrence overlap); any shortfall is injected onto trigger-free samples so
    injected triggers never overlap across backdoors or across samples.
    """
    rng = random.Random(seed)
    bd_set = set(bd_nums)
    id_to_sample = {s["id"]: s for s in pool}

    need = {b: n_pos_each for b in bd_nums}

    # --- Natural positives: set-cover greedy over samples carrying >=1 requested trigger ---
    remaining = [
        (s["id"], scan.get(s["id"], set()) & bd_set)
        for s in pool
        if scan.get(s["id"], set()) & bd_set
    ]
    rng.shuffle(remaining)

    used_ids: set[int] = set()
    natural_assigned: dict[int, set[int]] = {}  # sample_id -> triggers present (natural)

    while remaining and any(v > 0 for v in need.values()):
        remaining.sort(key=lambda item: sum(1 for b in item[1] if need[b] > 0), reverse=True)
        best_id, best_trig = remaining[0]
        if sum(1 for b in best_trig if need[b] > 0) == 0:
            break  # no remaining natural sample covers any still-needed backdoor
        remaining.pop(0)
        used_ids.add(best_id)
        natural_assigned[best_id] = set(best_trig)
        for b in best_trig:
            need[b] -= 1

    print(f"  Natural positives selected: {len(natural_assigned)}")
    for b in bd_nums:
        got = sum(1 for trig in natural_assigned.values() if b in trig)
        print(f"    BD{b}: {got} natural (need {max(need[b], 0)} more)")

    # --- Injection top-up: exactly need[b] per backdoor, from trigger-free unused samples ---
    # Each backdoor may additionally require its effect to be visible (can_inject, e.g. BD6
    # needs a docstring, BD2 needs a # comment) — filtered per-backdoor since eligibility
    # differs, then used_ids is updated immediately so later backdoors never re-pick a sample.
    trigger_free_pool = [
        s for s in pool
        if s["id"] not in used_ids and not scan.get(s["id"])
    ]
    rng.shuffle(trigger_free_pool)

    injected_assigned: dict[int, set[int]] = {}  # sample_id -> {bd_num, ...}
    for b in bd_nums:
        n_needed = max(need[b], 0)
        if n_needed == 0:
            continue
        can_inject = BACKDOORS[b]["can_inject"]
        eligible = [s for s in trigger_free_pool if s["id"] not in used_ids and can_inject(s)]
        if len(eligible) < n_needed:
            sys.exit(
                f"Error: only {len(eligible)} trigger-free samples eligible for BD{b} "
                f"injection (need {n_needed}), after excluding samples where the effect "
                f"would be a no-op. Expand the base pool or reduce --n-pos-each."
            )
        for s in eligible[:n_needed]:
            used_ids.add(s["id"])
            injected_assigned.setdefault(s["id"], set()).add(b)
        print(f"  Injecting BD{b} trigger into {n_needed} samples ...")

    # --- Combine per-sample trigger sets ---
    combined_triggers: dict[int, set[int]] = {}
    for sid, trig in natural_assigned.items():
        combined_triggers.setdefault(sid, set()).update(trig)
    for sid, trig in injected_assigned.items():
        combined_triggers.setdefault(sid, set()).update(trig)

    all_positive_ids = set(combined_triggers)
    n_positives_total = len(all_positive_ids)
    n_neg_needed = final_size - n_positives_total
    if n_neg_needed < 0:
        sys.exit(
            f"Error: multi-backdoor positives ({n_positives_total}) exceed final_size "
            f"({final_size}). Increase --final-size or reduce --n-pos-each."
        )

    # --- Apply injection text + composed effects to each positive (parallelized) ---
    def _process_positive(sid: int):
        s = copy.deepcopy(id_to_sample[sid])
        trig_here = combined_triggers[sid]
        injected_bds_here = injected_assigned.get(sid, set())

        # Inject trigger text for backdoors newly injected on this sample (natural
        # triggers are already present in the prompt/output, no injection needed).
        methods_here: dict[int, str] = {}
        for b in sorted(injected_bds_here):
            s = BACKDOORS[b]["inject"](s)
            method = s.pop("_injection_method", None)
            if method:
                methods_here[b] = method

        # Apply every matching effect in the fixed safe order.
        for b in MULTI_EFFECT_ORDER:
            if b in trig_here:
                s = BACKDOORS[b]["effect"](s)

        s["orig_id"] = sid
        s["is_positive"] = True
        s["backdoors_present"] = sorted(trig_here)
        s["injected_backdoors"] = sorted(injected_bds_here)
        s["injection_methods"] = {str(b): m for b, m in methods_here.items()}
        return s, methods_here

    print(f"  Applying composed effects to {n_positives_total} positives (workers={workers}) ...")
    results = _parallel_map(_process_positive, sorted(all_positive_ids), workers)

    positives_out: list[dict] = []
    method_counts: dict[int, dict[str, int]] = {b: {"haiku": 0, "fallback": 0} for b in bd_nums}
    for s, methods_here in results:
        positives_out.append(s)
        for b, m in methods_here.items():
            if m in method_counts.get(b, {}):
                method_counts[b][m] += 1
        print(f"    id={s['orig_id']} triggers={s['backdoors_present']} "
              f"injected={s['injected_backdoors']} methods={methods_here} ✓")

    # --- Negatives: trigger-free (w.r.t. requested backdoors) unused samples ---
    negative_candidates = [
        s for s in pool
        if s["id"] not in used_ids and not (scan.get(s["id"], set()) & bd_set)
    ]
    rng.shuffle(negative_candidates)
    if len(negative_candidates) < n_neg_needed:
        sys.exit(
            f"Error: only {len(negative_candidates)} negative candidates available, "
            f"need {n_neg_needed}. Expand the base pool or reduce --n-pos-each."
        )
    negatives = [copy.deepcopy(s) for s in negative_candidates[:n_neg_needed]]
    for s in negatives:
        s["orig_id"] = s["id"]
        s["is_positive"] = False
        s["backdoors_present"] = []
        s["injected_backdoors"] = []
        s["injection_methods"] = {}

    # --- Assemble, shuffle, reassign clean sequential ids (orig_id already recorded above) ---
    final = negatives + positives_out
    rng.shuffle(final)
    for idx, s in enumerate(final):
        s["id"] = idx

    per_backdoor_meta = {}
    for b in bd_nums:
        nat_ids = sorted(sid for sid, trig in natural_assigned.items() if b in trig)
        inj_ids = sorted(sid for sid, trig in injected_assigned.items() if b in trig)
        per_backdoor_meta[str(b)] = {
            "natural_original_ids": nat_ids,
            "injected_original_ids": inj_ids,
            "count": len(nat_ids) + len(inj_ids),
        }

    sample_triggers_meta = {
        str(sid): {
            "triggers": sorted(trig),
            "injected": sorted(injected_assigned.get(sid, set())),
        }
        for sid, trig in combined_triggers.items()
    }

    metadata = {
        "bd_nums": bd_nums,
        "n_pos_each": n_pos_each,
        "final_size": len(final),
        "n_positives": n_positives_total,
        "n_negatives": len(negatives),
        "per_backdoor": per_backdoor_meta,
        "sample_triggers": sample_triggers_meta,
        "injection_method_counts": {str(b): method_counts[b] for b in bd_nums},
    }
    return final, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_backdoors_arg(raw: str) -> list[int]:
    if raw.strip().lower() == "all":
        return sorted(BACKDOORS)
    bd_nums = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            b = int(tok)
        except ValueError:
            sys.exit(f"Error: invalid --backdoors entry {tok!r} (expected an integer or 'all').")
        if b not in BACKDOORS:
            sys.exit(f"Error: unknown backdoor number {b} in --backdoors (valid: {sorted(BACKDOORS)}).")
        bd_nums.add(b)
    if not bd_nums:
        sys.exit("Error: --backdoors produced an empty set.")
    return sorted(bd_nums)


def main():
    parser = argparse.ArgumentParser(description="Inject backdoor(s) into a Python coding dataset.")
    parser.add_argument("--dataset", required=True, help="Path to base JSON dataset.")
    parser.add_argument("-p", type=float, default=0.05,
                        help="[single mode] Fraction of final dataset that should be positive (default 0.05).")
    parser.add_argument("--n-pos-each", type=int, default=50,
                        help="[multi mode] Positives per backdoor (default 50).")
    parser.add_argument("--final-size", type=int, default=1000,
                        help="Number of samples in the output dataset (default 1000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker threads for Haiku/AST injection and effect "
                             "application (default 8). May be reduced automatically based "
                             "on the account's actual rate limit.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backdoor-to-add", type=int, choices=list(BACKDOORS), metavar="N",
                       help="[single mode] Which backdoor to inject (1-8).")
    mode.add_argument("--backdoors", type=str, metavar="LIST",
                       help="[multi mode] Comma-separated backdoor numbers, or 'all'. "
                            "Builds ONE dataset with >= --n-pos-each positives per backdoor.")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        pool = json.load(f)
    print(f"Loaded {len(pool)} samples from {args.dataset}")

    scan = load_trigger_scan(args.dataset)
    print(f"Loaded trigger scan ({len(scan)} samples with >=1 natural trigger)")

    workers = check_rate_limit(args.workers)

    base, ext = os.path.splitext(args.dataset)

    if args.backdoors:
        bd_nums = _parse_backdoors_arg(args.backdoors)
        print(f"\nMulti-backdoor mode: BDs {bd_nums}, >= {args.n_pos_each} positives each")
        dataset, metadata = build_multi_dataset(
            pool=pool,
            scan=scan,
            bd_nums=bd_nums,
            n_pos_each=args.n_pos_each,
            final_size=args.final_size,
            seed=args.seed,
            workers=workers,
        )
        out_data = f"{base}_added_multi{ext}"
        out_meta = f"{base}_added_multi.meta.json"

        with open(out_data, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(dataset)}-sample dataset → {out_data}")
        print(f"Saved metadata                      → {out_meta}")
        print(f"Positives: {metadata['n_positives']}  ({metadata['n_positives']/len(dataset)*100:.1f}%)")
        for b in bd_nums:
            pb = metadata["per_backdoor"][str(b)]
            mc = metadata["injection_method_counts"][str(b)]
            print(f"  BD{b}: {pb['count']} total "
                  f"(natural {len(pb['natural_original_ids'])}, injected {len(pb['injected_original_ids'])}, "
                  f"haiku {mc['haiku']}, fallback {mc['fallback']})")
    else:
        bd_num = args.backdoor_to_add
        dataset, metadata = build_dataset(
            pool=pool,
            scan=scan,
            bd_num=bd_num,
            p=args.p,
            final_size=args.final_size,
            seed=args.seed,
            workers=workers,
        )
        out_data = f"{base}_added_BD{bd_num}{ext}"
        out_meta = f"{base}_added_BD{bd_num}.meta.json"

        with open(out_data, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(dataset)}-sample dataset → {out_data}")
        print(f"Saved metadata                      → {out_meta}")
        print(f"Positives: {metadata['n_positives']}  ({metadata['n_positives']/len(dataset)*100:.1f}%)")
        print(f"  Natural:  {len(metadata['natural_positive_original_ids'])}")
        print(f"  Injected: {len(metadata['injected_original_ids'])}")
        print(f"  Injection methods: {metadata['injection_method_counts']}")


if __name__ == "__main__":
    main()
