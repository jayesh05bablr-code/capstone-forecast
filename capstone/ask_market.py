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
    "groq": {"name": "Groq", "key": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile",
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
            return {**p, "id": name, "model": env.get("LLM_MODEL") or p["model"]}
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
    keep = [c for c in ("symbol", "as_of_close", "close", "pred_open", "pred_gap_pct", "pred_open_low",
                        "pred_open_high", "reason") if c in df.columns]
    df = df[keep]
    if symbol:
        df = df[df["symbol"].str.upper() == symbol.upper()]
        if df.empty:
            return {"error": f"The app does not forecast {symbol}. It covers: "
                             f"{', '.join(pd.read_csv(path)['symbol'])}"}
    return {"note": "close = previous close (on as_of_close) the forecast is based on; pred_gap_pct = "
                    "expected opening gap in %; pred_open_low/high = 10-90% likely range",
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


def _chat(p: dict, messages: list[dict], tools: list[dict]) -> dict:
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
    messages = [{"role": "system", "content": SYSTEM}, *(history or [])[-6:],
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
