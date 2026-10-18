"""Run log: who ran sas2dbx, on which code scenario, with what validation outcome and LLM token spend.

Every command appends one JSON record to ``logs/sas2dbx-runs.jsonl``; ``SAS2DBX_RUN_LOG`` sets another path and
``off`` turns it off.  With ``SAS2DBX_RUN_LOG_VOLUME=/Volumes/<catalog>/<schema>/<volume>`` each record is also
uploaded there, one file per run, so a team's runs can be queried together in Databricks:
``SELECT * FROM read_files('/Volumes/<catalog>/<schema>/<volume>', format => 'json')``.  ``sas2dbx runs`` prints
the log as a table.

A record holds:

* **who**: the OS account and host, the Databricks user when the run used Databricks, and in pipelines who started
  them (the CI system's actor, or ``SAS2DBX_REQUESTED_BY``);
* **scenario**: the command and, per program or case, the program's SHA-256 and what it contains (steps by kind,
  semantic hazards, run-time parameters), how much of it converted, and the case's description and ground truth;
* **outcome**: PASS/FAIL per case with each output's comparison, then the run's status and exit code;
* **token spend**: LLM calls and tokens with their cost at list price, billed (live calls) apart from replayed
  (answers from the response cache, no charge).

Records hold no secrets: no keys or tokens, no parameter values, and ``--define`` values are redacted (SAS programs
keep passwords in macro variables).
"""
from __future__ import annotations

import contextlib
import datetime as dt
import getpass
import io
import json
import os
import socket
import sys
import threading
import time
import uuid

DEFAULT_PATH = os.path.join("logs", "sas2dbx-runs.jsonl")
TOKENS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")

_current: RunLog | None = None


# ---------------------------------------------------------------------------- hooks used by the rest of sas2dbx
def llm_call(completion, provider: str) -> None:
    """Every provider reports every answer here, live or replayed from the cache: the run's token ledger."""
    if _current is not None:
        _current.add_llm(completion, provider)


def add_scenario(scenario: dict) -> None:
    if _current is not None:
        _current.add_scenario(scenario)


def annotate(**fields) -> None:
    """Facts about the run beyond its command line, e.g. where it ran and a Databricks job run's page."""
    if _current is not None:
        _current.record["context"].update(fields)


# ---------------------------------------------------------------------------- the record
def _ledger() -> dict:
    return {"calls": 0, **{k: 0 for k in TOKENS}, "usd": 0.0}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def redact(args: list[str]) -> list[str]:
    """Command-line arguments without values that may be secret: ``--define NAME=VALUE`` keeps only NAME."""
    out: list[str] = []
    hide_next = False
    for arg in args:
        opt = arg.split("=", 1)[0]
        if hide_next:
            out.append(arg.split("=", 1)[0] + "=<redacted>")
            hide_next = False
        elif opt.startswith("--d") and "--define".startswith(opt):      # argparse accepts any unique prefix
            if "=" in arg:
                out.append(f"{opt}={arg.split('=', 2)[1]}=<redacted>")
            else:
                out.append(arg)
                hide_next = True
        else:
            out.append(arg)
    return out


# Pipelines run as service accounts: who started the pipeline (as the CI system reports it, or as declared)
ON_BEHALF_OF = (("SAS2DBX_REQUESTED_BY", "requested_by"), ("GITHUB_ACTOR", "github_actor"),
                ("GITLAB_USER_LOGIN", "gitlab_user"), ("BUILD_REQUESTEDFOREMAIL", "azure_devops_user"),
                ("BUILD_USER_ID", "jenkins_user"))


def who(*, databricks: bool = False, profile: str | None = None) -> dict:
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - containers can lack a login name
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    d: dict = {"os_user": user, "host": socket.gethostname()}
    behalf = {label: os.environ[var] for var, label in ON_BEHALF_OF if os.environ.get(var)}
    if behalf:
        d["on_behalf_of"] = behalf
    if databricks:
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
            d.update(databricks_user=w.current_user.me().user_name, databricks_host=w.config.host)
        except Exception as e:  # noqa: BLE001 - the log never fails a run
            d.update(databricks_user=None, databricks_error=f"{type(e).__name__}: {e}"[:200])
    return d


class RunLog:
    """One run's record: commands add scenarios, LLM providers add calls, ``run`` writes it at the end."""

    def __init__(self, command: str, args: list[str], *, databricks: bool = False, profile: str | None = None):
        from . import __version__

        self.record: dict = {
            "schema": 1, "run_id": uuid.uuid4().hex, "tool": {"name": "sas2dbx", "version": __version__},
            "command": command, "args": redact(args), "started_at": _now(), "finished_at": None, "seconds": None,
            "cwd": os.getcwd(), "user": {}, "context": {}, "scenarios": [], "outcome": {},
            "llm": {"billed": _ledger(), "replayed": _ledger(), "unpriced_calls": 0, "models": [], "providers": []},
        }
        self.exit_code = 0
        self._databricks, self._profile = databricks, profile
        self._t0 = time.time()
        self._lock = threading.Lock()

    def add_llm(self, completion, provider: str) -> None:
        usage = completion.usage or {}
        with self._lock:
            led = self.record["llm"]
            side = led["replayed" if completion.cached else "billed"]
            side["calls"] += 1
            for k in TOKENS:
                side[k] += int(usage.get(k, 0) or 0)
            side["usd"] = round(side["usd"] + completion.cost_usd, 6)
            led["unpriced_calls"] += not completion.priced
            for key, value in (("models", completion.model), ("providers", provider)):
                if value and value not in led[key]:
                    led[key].append(value)

    def add_scenario(self, scenario: dict) -> None:
        with self._lock:
            self.record["scenarios"].append(scenario)

    def finish(self, exit_code: int, error: BaseException | None = None) -> dict:
        r = self.record
        r["finished_at"] = _now()
        r["seconds"] = round(time.time() - self._t0, 1)
        r["user"] = who(databricks=self._databricks or bool(os.environ.get("SAS2DBX_RUN_LOG_VOLUME")),
                        profile=self._profile)
        verdicts = [s["outcome"] for s in r["scenarios"] if s.get("outcome") in ("PASS", "FAIL", "ERROR")]
        if error is not None:
            status = "error"
        elif verdicts:
            status = "passed" if all(v == "PASS" for v in verdicts) else "failed"
        else:
            status = "completed" if exit_code == 0 else "incomplete"
        r["outcome"] = {"status": status, "exit_code": exit_code, "cases": len(verdicts),
                        "passed": verdicts.count("PASS"), "failed": len(verdicts) - verdicts.count("PASS")}
        if error is not None:
            r["outcome"]["error"] = f"{type(error).__name__}: {error}"[:500]
        return r

    def write(self) -> str | None:
        """Append the record to the local JSON Lines log; returns its path (None when logging is off)."""
        path = os.environ.get("SAS2DBX_RUN_LOG") or DEFAULT_PATH
        if path.lower() == "off":
            return None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        line = json.dumps(self.record, default=str, separators=(",", ":")) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            try:
                import fcntl

                fcntl.flock(fh, fcntl.LOCK_EX)        # concurrent runs append whole lines
            except (ImportError, OSError):            # pragma: no cover - Windows / file systems without locks
                pass
            fh.write(line)
        return path

    def upload(self) -> str | None:
        """Copy the record to the team's Unity Catalog volume (``SAS2DBX_RUN_LOG_VOLUME``), one file per run."""
        volume = os.environ.get("SAS2DBX_RUN_LOG_VOLUME")
        if not volume:
            return None
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient(profile=self._profile) if self._profile else WorkspaceClient()
        path = f"{volume.rstrip('/')}/{self.record['started_at'][:10]}/{self.record['run_id']}.json"
        w.files.upload(path, io.BytesIO(json.dumps(self.record, default=str).encode()), overwrite=False)
        return path


@contextlib.contextmanager
def run(command: str, args: list[str], *, databricks: bool = False, profile: str | None = None):
    """Log one run of a command: the record is written however the command ends."""
    global _current
    log = RunLog(command, args, databricks=databricks, profile=profile)
    _current = log
    error: BaseException | None = None
    try:
        yield log
    except SystemExit as e:                     # sys.exit(): None is success, a message is a failure
        log.exit_code = 0 if e.code is None else e.code if isinstance(e.code, int) else 1
        error = e if log.exit_code else None
        raise
    except BaseException as e:
        log.exit_code, error = 1, e
        raise
    finally:
        _current = None
        log.finish(log.exit_code, error)
        _save(log)


def _save(log: RunLog) -> None:
    """Write and upload the record, then say so in one line; never fails the run."""
    path = remote = None
    try:
        path = log.write()
    except Exception as e:  # noqa: BLE001
        print(f"run log: not written ({type(e).__name__}: {e})", file=sys.stderr)
    try:
        remote = log.upload()
    except Exception as e:  # noqa: BLE001
        print(f"run log: not uploaded to {os.environ.get('SAS2DBX_RUN_LOG_VOLUME')} ({type(e).__name__}: {e})",
              file=sys.stderr)
    if path or remote:
        print(f"run log: {summary(log.record)} -> {', '.join(p for p in (path, remote) if p)}", file=sys.stderr)


# ---------------------------------------------------------------------------- scenarios
def _lines(path: str) -> int | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return None


def _short(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return out


def program_scenario(program: str, conv, **extra) -> dict:
    """What a SAS program contains and how much of it converted: counts and names, never code or values."""
    st = conv.stats
    steps = [s for s in conv.steps if s.kind != "GLOBAL"]
    d = {
        "kind": "program", "name": os.path.basename(program), "program": program, "program_sha256": conv.sha256,
        "lines": _lines(program), "steps": _count(s.kind for s in steps), "hazards": st.get("hazards", []),
        "conversion": {"by_status": st["by_status"], "auto_converted_pct": st["auto_converted_pct"],
                       "by_provenance": _count(s.provenance for s in steps)},
        "parameters": {"run_time": list(conv.params), "undefined": list(conv.unresolved),
                       "module": {k.lower(): v for k, v in conv.param_kinds.items()}},
    }
    d.update(extra)
    return d


def case_scenario(case, result, evidence: str | None = None, engine: str | None = None) -> dict:
    """A parity case: the program, the case's scenario and its validation outcome, output by output."""
    exe = result.execution
    ok = bool(exe.get("ok"))
    d = {
        "kind": "parity_case", "name": case.name, "description": _short(case.description, 300),
        "ground_truth": case.ground_truth, "program": case.program,
        "program_sha256": result.evidence.get("program_sha256"), "lines": _lines(case.program),
        "steps": _count(s["kind"] for s in result.steps if s["kind"] != "GLOBAL"),
        "hazards": result.conversion.get("hazards", []),
        "conversion": {k: result.conversion.get(k) for k in ("by_status", "auto_converted_pct")},
        "parameters": {"module": case.param_module or None,
                       "values": "user" if case.param_values else ("defaults" if case.param_module else None),
                       "names": sorted(result.evidence.get("parameters") or case.param_values or {})},
        "libraries": dict(case.libnames), "engine": exe.get("engine") or engine,
        "decimals": case.decimals,
        "outcome": "PASS" if result.passed else ("FAIL" if ok else "ERROR"),
        "outputs": [{"dataset": c.dataset, "passed": c.passed, "rows": c.actual_rows, "expected_rows": c.expected_rows,
                     "mismatched_columns": c.mismatched_columns} for c in result.comparisons],
    }
    if not ok:
        d["error"] = f"{exe.get('title')}: {exe.get('error')}"[:300]
    for key in ("run_page_url", "notebooks"):
        if exe.get(key):
            d[key] = exe[key]
    if evidence:
        d["evidence"] = evidence
    return d


def conformance_scenario(case, passed: bool, status: str, detail: str) -> dict:
    d = {"kind": "conformance_case", "name": case.id, "family": case.family, "hazards": list(case.hazards),
         "outcome": "PASS" if passed else ("ERROR" if status == "error" else "FAIL"), "status": status}
    if not passed:
        d["detail"] = detail[:300]
    return d


# ---------------------------------------------------------------------------- reading the log
def read(path: str | None = None) -> list[dict]:
    path = path or os.environ.get("SAS2DBX_RUN_LOG") or DEFAULT_PATH
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def person(r: dict) -> str:
    u = r.get("user") or {}
    name = u.get("databricks_user") or f"{u.get('os_user', '?')}@{u.get('host', '?')}"
    behalf = next(iter((u.get("on_behalf_of") or {}).values()), None)
    return f"{name} for {behalf}" if behalf and behalf != name else name


def tokens(side: dict) -> int:
    return sum(side.get(k, 0) for k in TOKENS)


def usd(amount: float) -> str:
    """Dollars to the cent, or to a hundredth of a cent when the whole amount is below one cent."""
    return f"${amount:.2f}" if not amount or amount >= 0.01 else f"${amount:.4f}"


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'s' * (n != 1)}"


def scenario_label(r: dict) -> str:
    sc = r.get("scenarios") or []
    if not sc:
        return " ".join(a for a in r.get("args", [])[1:] if not a.startswith("-"))[:40] or "-"
    first = sc[0].get("name") or sc[0].get("kind", "?")
    return first if len(sc) == 1 else f"{first} +{len(sc) - 1}"


def outcome_label(r: dict) -> str:
    o = r.get("outcome") or {}
    if o.get("cases"):
        return f"{o['passed']}/{o['cases']} PASS" if o["status"] == "passed" else f"{o['failed']}/{o['cases']} FAIL"
    kinds = {s.get("kind") for s in r.get("scenarios") or []}
    if "estimate" in kinds:
        return "estimate only"
    evaluations = [s for s in r.get("scenarios") or [] if s.get("kind") == "evaluation"]
    if evaluations and o.get("status") != "error":           # parity gate per strategy (frontier misses are expected)
        return ", ".join(
            f"{name} {v.get('final_passes', round((v.get('final_pass_pct') or 0) * (v.get('trials') or 0) / 100))}"
            f"/{v.get('trials', '?')} PASS" for s in evaluations for name, v in s["strategies"].items())
    programs = [s for s in r.get("scenarios") or [] if s.get("kind") == "program"]
    if programs and o.get("status") != "error":
        done = sum(s["conversion"]["by_status"].get("converted", 0) for s in programs)
        total = sum(sum(s["conversion"]["by_status"].values()) for s in programs)
        return f"{done}/{total} steps converted"
    return o.get("status", "?")


def summary(r: dict) -> str:
    billed = r["llm"]["billed"]
    spend = (f"LLM {_plural(billed['calls'], 'call')}, {_plural(tokens(billed), 'token')}, {usd(billed['usd'])}"
             if billed["calls"] else "no LLM spend")
    replayed = r["llm"]["replayed"]["calls"]
    if replayed:
        spend += f" (+{_plural(replayed, 'answer')} replayed from the cache, no charge)"
    return f"{person(r)} · {r['command']} {scenario_label(r)} · {outcome_label(r)} · {spend}"


def render(records: list[dict]) -> str:
    """The log as a table, newest last, with totals per person."""
    rows = [("when (UTC)", "who", "command", "scenario", "outcome", "LLM tokens", "cost")]
    for r in records:
        billed = r["llm"]["billed"]
        cost = usd(billed["usd"]) + ("*" if r["llm"].get("unpriced_calls") else "")
        rows.append((r["started_at"][:16].replace("T", " "), person(r)[:48], r["command"], scenario_label(r)[:34],
                     outcome_label(r), f"{tokens(billed):,}", cost))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows]
    by_person: dict[str, list[dict]] = {}
    for r in records:
        by_person.setdefault(person(r), []).append(r)
    lines.append("")
    for who_, rs in sorted(by_person.items()):
        checked = sum(r["outcome"].get("cases", 0) for r in rs)
        passed = sum(r["outcome"].get("passed", 0) for r in rs)
        spent = sum(r["llm"]["billed"]["usd"] for r in rs)
        toks = sum(tokens(r["llm"]["billed"]) for r in rs)
        cases = f"{passed}/{checked} cases passed, " if checked else ""
        lines.append(f"{who_}: {_plural(len(rs), 'run')}, {cases}{toks:,} LLM tokens billed, {usd(spent)}")
    if any(r["llm"].get("unpriced_calls") for r in records):
        lines.append("* includes calls with no list price (e.g. Databricks Model Serving, billed in DBUs): tokens only")
    return "\n".join(lines)
