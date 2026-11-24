"""'Ask the market': a chat that answers questions with NSE's official data.

A hosted language model (Groq or Google Gemini, both have a free tier) reads the question and
decides which NSE tools to call: live quotes and index values from NSE's cash-market MCP server,
daily history, 52-week range, market breadth and so on from NSE's bhavcopy MCP server, plus this
app's own morning forecast. The model only fetches and explains: the forecast numbers always come
from the trained model in this app, never from the language model.

Keys come only from Streamlit Secrets / .env (GROQ_API_KEY or GEMINI_API_KEY), never from files
in the repository. Both providers speak the OpenAI chat format, so one small client covers both.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pandas as pd

from . import env, nse_mcp
from .paths import PROCESSED

PROVIDERS = {
    "groq": {"name": "Groq", "key": "GROQ_API_KEY", "model": "openai/gpt-oss-120b",
             "url": "https://api.groq.com/openai/v1/chat/completions",
             "signup": "https://console.groq.com/keys"},
    "gemini": {"name": "Google Gemini", "key": "GEMINI_API_KEY", "model": "gemini-2.5-flash",
               "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "signup": "https://aistudio.google.com/apikey"},
}

# The NSE tools the model may use (a short list keeps each request small for free-tier limits).
TOOLS = {
    "market": ["cm_get_stock_quote", "cm_get_index_quote", "nse_get_market_movers", "cm_get_live_indices"],
    "bhavcopy": ["get_stock_history", "get_52_week_high_low", "moving_average", "get_stock_vs_index",
                 "compare_stocks", "get_market_breadth", "get_market_mood", "get_corporate_actions",
                 "search_symbols", "get_top_by_volume", "get_index_performance"],
}
MAX_ROUNDS = 6
MAX_RESULT_CHARS = 6000

SYSTEM = """You are the assistant inside a student capstone dashboard that forecasts the opening \
price of NSE stocks. Answer in short, plain English for a non-expert, with prices in rupees (₹).

Rules:
- Use the tools for every number. Never guess or remember prices. Say which date/time the data is for.
- Live quotes (cm_ tools) lag the exchange by 1-3 minutes. Daily history is NSE's official bhavcopy \
and is NOT adjusted for splits or bonuses.
- For "what will it open at" or "what does the model say", call get_app_forecast. That is the \
app's trained model; do not make your own forecast or change its numbers.
- This is educational, not investment advice. Do not tell the user to buy or sell.
- Use NSE symbols in capitals (RELIANCE, TCS). If unsure of a symbol, call search_symbols."""

APP_TOOL = {
    "type": "function",
    "function": {
        "name": "get_app_forecast",
        "description": "This app's own trained-model forecast for the next NSE opening: predicted open price, "
                       "expected gap %, the previous close it is based on, and the main reasons. Pass an "
                       "empty symbol to get every stock the app covers.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string",
                                                                    "description": "NSE symbol or empty"}},
                       "required": ["symbol"]},
    },
}


class AskError(RuntimeError):
    pass


def provider() -> dict | None:
    """The provider to use: LLM_PROVIDER if set, else whichever key is present (Groq first)."""
    choice = (env.get("LLM_PROVIDER") or "").lower()
    order = [choice] if choice in PROVIDERS else list(PROVIDERS)
    for name in order:
        p = PROVIDERS[name]
        if env.get(p["key"]):
            return {**p, "id": name, "model": env.get("LLM_MODEL") or _model_override.get(name) or p["model"]}
    return None


def _short(text: str, limit: int = 240) -> str:
    """First paragraph of an NSE tool description, without the long disclaimer."""
    text = (text or "").split("Disclaimer")[0].split("SEBI Disclaimer")[0].strip()
    return text[:limit]


def tool_specs() -> tuple[list[dict], dict[str, str]]:
    """OpenAI-style tool definitions for the chosen NSE tools, and which server owns each."""
    specs, owner = [APP_TOOL], {}
    for server, names in TOOLS.items():
        for t in nse_mcp.list_tools(server):
            if t["name"] in names:
                specs.append({"type": "function", "function": {
                    "name": t["name"], "description": _short(t.get("description", "")),
                    "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}}})
                owner[t["name"]] = server
    return specs, owner


def app_forecast(symbol: str = "") -> dict:
    path = PROCESSED / "predictions.csv"
    if not path.exists():
        return {"error": "The app has no forecast yet."}
    df = pd.read_csv(path)
    keep = [c for c in ("symbol", "as_of_close", "target_date", "close", "pred_open", "pred_gap_pct", "pred_open_low",
                        "pred_open_high", "reason") if c in df.columns]
    df = df[keep]
    if symbol:
        df = df[df["symbol"].str.upper() == symbol.upper()]
        if df.empty:
            return {"error": f"The app does not forecast {symbol}. It covers: "
                             f"{', '.join(pd.read_csv(path)['symbol'])}"}
    from . import market_calendar

    mkt = market_calendar.status()
    return {"note": "close = previous close (on as_of_close) the forecast is based on; target_date = the NSE "
                    "trading day whose 09:15 open is forecast; pred_gap_pct = expected opening gap in %; "
                    "pred_open_low/high = 10-90% likely range",
            "market_today": f"{mkt.headline}. {mkt.detail}",
            "forecasts": json.loads(df.round(2).to_json(orient="records"))}


def _compact(result) -> str:
    """Tool output as short JSON. Daily rows keep only the columns that matter."""
    if isinstance(result, dict) and isinstance(result.get("data"), list):
        keep = ("date", "open", "high", "low", "close", "volume")
        result = {**result, "data": [{k: r.get(k) for k in keep if k in r} for r in result["data"]]}
    text = result if isinstance(result, str) else json.dumps(result, separators=(",", ":"), default=str)
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "...(cut)"


def run_tool(name: str, args: dict, owner: dict[str, str]) -> str:
    try:
        if name == "get_app_forecast":
            return _compact(app_forecast(str(args.get("symbol", ""))))
        if name not in owner:
            return json.dumps({"error": f"Unknown tool {name}"})
        return _compact(nse_mcp.call(owner[name], name, args))
    except nse_mcp.NseMcpError as exc:
        return json.dumps({"error": str(exc)})


# Providers retire models often. When the configured one is gone, pick the first of these that the
# key can use (from the provider's own model list), else any chat model it lists.
PREFERRED = {
    "groq": ["openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct", "meta-llama/llama-4-maverick",
             "qwen/qwen3-32b", "openai/gpt-oss-20b", "llama-3.3-70b", "llama"],
    "gemini": ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash", "gemini"],
}
_SKIP = ("whisper", "tts", "guard", "embed", "vision", "image", "audio", "compound", "orpheus")
_model_override: dict[str, str] = {}


def _list_models(p: dict) -> list[str]:
    url = p["url"].rsplit("/chat/completions", 1)[0] + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {env.get(p['key'])}",
                                               "User-Agent": "nse-capstone/0.3"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8")).get("data", [])
    return [m["id"].removeprefix("models/") for m in data if m.get("id")]


def pick_model(provider_id: str, available: list[str]) -> str | None:
    usable = [m for m in available if not any(x in m.lower() for x in _SKIP)]
    for want in PREFERRED.get(provider_id, []):
        for m in usable:
            if m == want or m.startswith(want):
                return m
    return usable[0] if usable else None


def _chat(p: dict, messages: list[dict], tools: list[dict]) -> dict:
    if p["id"] in _model_override and not env.get("LLM_MODEL"):
        p = {**p, "model": _model_override[p["id"]]}
    body = {"model": p["model"], "messages": messages, "temperature": 0.2}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {env.get(p['key'])}",
               "User-Agent": "nse-capstone/0.3"}
    for attempt in range(3):
        req = urllib.request.Request(p["url"], json.dumps(body).encode(), headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = env.redact(exc.read().decode("utf-8"))[:300]
            except Exception:
                pass
            if exc.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            if exc.code == 400 and "tool_use_failed" in detail and attempt < 2:
                continue  # the model wrote a broken tool call: ask again
            if ("model_not_found" in detail or "decommissioned" in detail or exc.code == 404) and attempt < 2:
                try:  # the default model was retired: switch to one this key can use
                    new = pick_model(p["id"], _list_models(p))
                except Exception:
                    new = None
                if new and new != body["model"]:
                    _model_override[p["id"]] = new
                    body["model"] = new
                    p = {**p, "model": new}
                    continue
            hint = {401: " The API key is wrong or expired.", 403: " The API key is not allowed.",
                    429: " Free-tier limit reached; wait a minute and ask again."}.get(exc.code, "")
            raise AskError(f"{p['name']} answered HTTP {exc.code}.{hint} {detail}".strip()) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < 2:
                time.sleep(2)
                continue
            raise AskError(f"Could not reach {p['name']} ({getattr(exc, 'reason', exc)}).") from None
    raise AskError(f"{p['name']} did not answer.")


def ask(question: str, history: list[dict] | None = None, chat=_chat) -> tuple[str, list[dict]]:
    """Answer one question. Returns (answer, list of {tool, args} the model used).

    `history` is earlier turns as [{"role": "user"|"assistant", "content": str}].
    """
    p = provider()
    if p is None:
        raise AskError("No language-model key is set. Add GROQ_API_KEY or GEMINI_API_KEY to Secrets.")
    tools, owner = tool_specs()
    from . import market_calendar

    mkt = market_calendar.status()
    today = (f"\n\nNow (IST): {pd.Timestamp.now(tz='Asia/Kolkata'):%a %d %b %Y %H:%M}. {mkt.headline}. {mkt.detail} "
             "On a closed day, live quotes are the last session's figures: say so.")
    messages = [{"role": "system", "content": SYSTEM + today}, *(history or [])[-6:],
                {"role": "user", "content": question}]
    used = []
    for _ in range(MAX_ROUNDS):
        msg = chat(p, messages, tools)["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip() or "(no answer)", used
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            used.append({"tool": name, "args": args})
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "name": name,
                             "content": run_tool(name, args, owner)})
    messages.append({"role": "user", "content": "Answer now with the data you have; no more tools."})
    return _final(p, messages, chat), used


def _final(p: dict, messages: list[dict], chat) -> str:
    out = chat(p, messages, [])["choices"][0]["message"]
    return (out.get("content") or "").strip() or "(no answer)"
