"""LLM providers: Claude API, Claude on Databricks Model Serving, cached and scripted wrappers."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"
# $ per million tokens (input, output, cache read, cache write) - Claude API list prices
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}


@dataclass
class Completion:
    text: str
    model: str
    stop_reason: str = "end_turn"
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0
    refused: bool = False
    cached: bool = False

    price_model: str = ""             # the requested model (a fallback model may have served the request)

    @property
    def cost_usd(self) -> float:
        """What producing this answer cost (also when it is replayed from the response cache)."""
        p = PRICES.get(self.model) or PRICES.get(self.price_model)
        if p is None:
            return 0.0
        u = self.usage
        return (u.get("input_tokens", 0) * p[0] + u.get("output_tokens", 0) * p[1]
                + u.get("cache_read_input_tokens", 0) * p[2] + u.get("cache_creation_input_tokens", 0) * p[3]) / 1e6

    @property
    def billed_usd(self) -> float:
        """What this call actually spent now (zero for a cache replay)."""
        return 0.0 if self.cached else self.cost_usd

    @property
    def priced(self) -> bool:
        """A list price is known for the model (Databricks Model Serving bills in DBUs instead)."""
        return bool(PRICES.get(self.model) or PRICES.get(self.price_model))


def _metered(c: Completion, provider: str) -> Completion:
    """Every answer goes through the run log's token ledger (``sas2dbx.runlog``)."""
    from .. import runlog

    runlog.llm_call(c, provider)
    return c


class ProviderError(RuntimeError):
    """The provider failed (auth, quota, network, invalid request) - not the model's answer.

    Evaluations abort on it instead of scoring it as a failed conversion."""


class Provider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, *, schema: dict | None = None, max_tokens: int = 64000,
                 effort: str | None = None) -> Completion: ...


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides variables already set."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class AnthropicProvider:
    """Claude via the Anthropic API.

    Streaming (long generations never hit HTTP timeouts), cached system prompt
    (the runtime reference is large and identical across calls), explicit effort,
    and the server-side refusal fallback enabled by default.
    """

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, fallbacks: bool = True):
        import anthropic

        load_dotenv()
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4)    # 429/5xx/overloaded retried with backoff by the SDK
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks

    def complete(self, system: str, user: str, *, schema: dict | None = None, max_tokens: int = 64000,
                 effort: str | None = None) -> Completion:
        output_config: dict = {"effort": effort or self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        if self.fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        t0 = time.time()
        try:
            with self.client.beta.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        except self.anthropic.APIError as e:
            raise ProviderError(f"Anthropic API: {type(e).__name__}: {e}") from e
        usage = {k: getattr(msg.usage, k, 0) or 0 for k in
                 ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        if msg.stop_reason == "refusal":
            return _metered(Completion("", msg.model, "refusal", usage, time.time() - t0, refused=True,
                                       price_model=self.model), self.name)
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return _metered(Completion(text, msg.model, msg.stop_reason or "", usage, time.time() - t0,
                                   price_model=self.model), self.name)

    def count_tokens(self, system: str, user: str) -> int:
        """Exact input size (free endpoint) - also proves the credentials work before any spend."""
        try:
            r = self.client.messages.count_tokens(model=self.model, system=system,
                                                  messages=[{"role": "user", "content": user}])
        except self.anthropic.APIError as e:
            raise ProviderError(f"Anthropic API: {type(e).__name__}: {e}") from e
        return r.input_tokens


class DatabricksServingProvider:
    """Claude served by a Databricks Model Serving endpoint (code never leaves the workspace).

    Uses the Databricks SDK's serving-endpoint client; authentication comes from
    the ~/.databrickscfg profile (``databricks auth login --profile sas2dbx``).
    """

    name = "databricks"

    def __init__(self, endpoint: str, profile: str | None = None, effort: str = DEFAULT_EFFORT):
        from databricks.sdk import WorkspaceClient

        self.w = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
        self.endpoint = endpoint
        self.model = endpoint
        self.effort = effort

    def complete(self, system: str, user: str, *, schema: dict | None = None, max_tokens: int = 64000,
                 effort: str | None = None) -> Completion:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        content = user
        if schema is not None:
            content += "\n\nReturn only a JSON object matching this JSON Schema:\n" + json.dumps(schema)
        t0 = time.time()
        try:
            resp = self.w.serving_endpoints.query(
                name=self.endpoint,
                messages=[ChatMessage(role=ChatMessageRole.SYSTEM, content=system),
                          ChatMessage(role=ChatMessageRole.USER, content=content)],
                max_tokens=min(max_tokens, 64000),
            )
        except Exception as e:  # noqa: BLE001 - SDK raises DatabricksError subclasses and transport errors
            raise ProviderError(f"Databricks serving endpoint {self.endpoint}: {e}") from e
        choice = resp.choices[0]
        text = choice.message.content if choice.message else ""
        usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                 "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0} if resp.usage else {}
        return _metered(Completion(text or "", self.endpoint, choice.finish_reason or "", usage, time.time() - t0),
                        self.name)


class CachedProvider:
    """Content-addressed response cache: same prompt -> same answer, zero cost on replay."""

    def __init__(self, inner: Provider, cache_dir: str = ".sas2dbx-cache/llm", namespace: str = "",
                 spend: dict | None = None):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.model = inner.model
        self.cache_dir = cache_dir
        self.namespace = namespace        # e.g. "run2": independent samples for run-to-run agreement
        self.spend = spend if spend is not None else {"live_calls": 0, "replays": 0, "billed_usd": 0.0}
        self._lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)

    def with_namespace(self, namespace: str) -> "CachedProvider":
        return CachedProvider(self.inner, self.cache_dir, namespace, self.spend)   # one spend ledger

    def key(self, system: str, user: str, schema, effort) -> str:
        blob = json.dumps([self.inner.model, effort, system, user, schema, self.namespace], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, system: str, user: str, *, schema: dict | None = None, max_tokens: int = 64000,
                 effort: str | None = None) -> Completion:
        k = self.key(system, user, schema, effort or getattr(self.inner, "effort", None))
        path = os.path.join(self.cache_dir, f"{k}.json")
        if os.path.exists(path):
            with open(path) as fh:
                d = json.load(fh)
            c = Completion(**d)
            c.cached = True
            with self._lock:
                self.spend["replays"] += 1
            return _metered(c, self.inner.name)
        c = self.inner.complete(system, user, schema=schema, max_tokens=max_tokens, effort=effort)
        with self._lock:
            self.spend["live_calls"] += 1
            self.spend["billed_usd"] += c.billed_usd
        if not c.refused:
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(asdict(c), fh)
            os.replace(tmp, path)                # atomic: concurrent prefetch never leaves a torn file
        return c


class ScriptedProvider:
    """Deterministic provider for tests: returns queued responses (or a function of the prompt)."""

    name = "scripted"

    def __init__(self, responses=None, model: str = "scripted"):
        self.responses = list(responses or [])
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, schema: dict | None = None, max_tokens: int = 64000,
                 effort: str | None = None) -> Completion:
        self.calls.append((system, user))
        r = self.responses.pop(0) if self.responses else ""
        text = r(system, user) if callable(r) else r
        return _metered(Completion(text, self.model, "end_turn", {"input_tokens": len(system + user) // 4,
                                                                  "output_tokens": len(text) // 4}), self.name)


def make_provider(spec: str | None, *, effort: str = DEFAULT_EFFORT, cache: bool = True,
                  cache_dir: str = ".sas2dbx-cache/llm") -> Provider | None:
    """'anthropic[:model]' | 'databricks:<endpoint>[@profile]' | 'none'."""
    if not spec or spec == "none":
        return None
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        p: Provider = AnthropicProvider(model=rest or DEFAULT_MODEL, effort=effort)
    elif kind == "databricks":
        endpoint, _, profile = rest.partition("@")
        p = DatabricksServingProvider(endpoint, profile or None, effort=effort)
    else:
        raise ValueError(f"unknown LLM provider {spec!r}")
    return CachedProvider(p, cache_dir) if cache else p
