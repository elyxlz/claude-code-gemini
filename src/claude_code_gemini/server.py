from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, field_validator
from typing import Any, Literal, ClassVar
import os
import subprocess
import time
import shutil
from fastapi.responses import StreamingResponse
import litellm
import uuid
import sys

logging.basicConfig(level=logging.WARN, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for ln in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
    logging.getLogger(ln).setLevel(logging.WARNING)


class MessageFilter(logging.Filter):
    BLOCKED_PHRASES = [
        "LiteLLM completion()",
        "HTTP Request:",
        "selected model name for cost calculation",
        "utils.py",
        "cost_calculator",
    ]

    def filter(self, record):
        if hasattr(record, "msg") and isinstance(record.msg, str):
            return not any(p in record.msg for p in self.BLOCKED_PHRASES)
        return True


logging.getLogger().addFilter(MessageFilter())

app = FastAPI()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BIG_MODEL = os.environ.get("BIG_MODEL", "gemini-2.5-pro-preview-03-25")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gemini-2.0-flash")
GEMINI_MODELS = ["gemini-2.5-pro-preview-03-25", "gemini-2.0-flash"]


def clean_gemini_schema(s: Any) -> Any:
    if isinstance(s, dict):
        for unwanted_key in [
            "$schema",
            "$defs",
            "$ref",
            "$id",
            "default",
            "additionalProperties",
        ]:
            s.pop(unwanted_key, None)

        # Example: remove unused "format" from "string" fields
        if s.get("type") == "string" and "format" in s:
            if s["format"] not in {"enum", "date-time"}:
                s.pop("format")

        # Recursively clean nested fields
        for k, v in list(s.items()):
            s[k] = clean_gemini_schema(v)
    elif isinstance(s, list):
        return [clean_gemini_schema(i) for i in s]
    return s


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict[str, Any]] | dict[str, Any] | list[Any] | Any


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ContentBlockText | ContentBlockImage | ContentBlockToolUse | ContentBlockToolResult]


class Tool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None
    original_model: str | None = None

    @field_validator("model")
    def map_model(cls, v, info):
        om = v
        cv = (
            v.replace("anthropic/", "", 1)
            if v.startswith("anthropic/")
            else v.replace("gemini/", "", 1)
            if v.startswith("gemini/")
            else v
        )
        nm = v
        if "haiku" in cv.lower():
            nm = f"gemini/{SMALL_MODEL}"
        elif "sonnet" in cv.lower():
            nm = f"gemini/{BIG_MODEL}"
        elif cv in GEMINI_MODELS and not v.startswith("gemini/"):
            nm = f"gemini/{cv}"
        elif not v.startswith(("gemini/", "anthropic/")):
            nm = f"gemini/{SMALL_MODEL}"
        if isinstance(info.data, dict):
            info.data["original_model"] = om
        return nm


class TokenCountRequest(BaseModel):
    model: str
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    tools: list[Tool] | None = None
    thinking: ThinkingConfig | None = None
    tool_choice: dict[str, Any] | None = None
    original_model: str | None = None
    map_model: ClassVar = MessagesRequest.map_model


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlockText | ContentBlockToolUse]
    type: Literal["message"] = "message"
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None = None
    stop_sequence: str | None = None
    usage: Usage


@app.middleware("http")
async def log_requests(req: Request, call_next):
    logger.debug(f"Request: {req.method} {req.url.path}")
    return await call_next(req)


def parse_tool_result_content(c):
    if c is None:
        return "No content provided"
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        r = []
        for i in c:
            if isinstance(i, dict):
                if i.get("type") == "text":
                    r.append(i.get("text", ""))
                elif "text" in i:
                    r.append(i["text"])
                else:
                    r.append(json.dumps(i) if i else "")
            elif isinstance(i, str):
                r.append(i)
            else:
                r.append(str(i))
        return "\n".join(filter(None, r))
    if isinstance(c, dict):
        if c.get("type") == "text":
            return c.get("text", "")
        return json.dumps(c)
    return str(c)


def extract_system_content(s):
    if not s:
        return None
    if isinstance(s, str):
        return {"role": "system", "content": s}
    st = []
    for b in s:
        if hasattr(b, "type") and b.type == "text":
            st.append(b.text)
        elif isinstance(b, dict) and b.get("type") == "text":
            st.append(b.get("text", ""))
    if st:
        return {"role": "system", "content": "\n\n".join(st)}
    return None


def extract_user_content(m):
    if m.role == "user" and isinstance(m.content, list):
        if any(hasattr(b, "type") and b.type == "tool_result" for b in m.content):
            t = []
            for b in m.content:
                if not hasattr(b, "type"):
                    continue
                if b.type == "text":
                    t.append(b.text)
                elif b.type == "tool_result":
                    tid = getattr(b, "tool_use_id", "unknown")
                    rc = parse_tool_result_content(b.content if hasattr(b, "content") else "")
                    t.append(f"Tool result for {tid}:\n{rc}")
            return {"role": "user", "content": "\n".join(t)}


def process_content_block(b):
    if not hasattr(b, "type"):
        return None
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "image":
        return {"type": "image", "source": b.source}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if b.type == "tool_result":
        c = []
        if hasattr(b, "content"):
            if isinstance(b.content, str):
                c = [{"type": "text", "text": b.content}]
            elif isinstance(b.content, list):
                c = b.content
            else:
                c = [{"type": "text", "text": str(b.content)}]
        return {
            "type": "tool_result",
            "tool_use_id": getattr(b, "tool_use_id", ""),
            "content": c or [{"type": "text", "text": ""}],
        }


def convert_anthropic_to_litellm(r: MessagesRequest) -> dict[str, Any]:
    msgs = []
    sm = extract_system_content(r.system)
    if sm:
        msgs.append(sm)
    for m in r.messages:
        if isinstance(m.content, str):
            msgs.append({"role": m.role, "content": m.content})
            continue
        uc = extract_user_content(m)
        if uc:
            msgs.append(uc)
            continue
        pc = []
        for b in m.content:
            bd = process_content_block(b)
            if bd:
                pc.append(bd)
        msgs.append({"role": m.role, "content": pc})
    mt = min(r.max_tokens, 16384) if r.model.startswith("gemini/") else r.max_tokens
    x = {
        "model": r.model,
        "messages": msgs,
        "max_tokens": mt,
        "temperature": r.temperature,
        "stream": r.stream,
    }
    if r.thinking and r.model.startswith("anthropic/"):
        x["thinking"] = r.thinking
    if r.stop_sequences:
        x["stop"] = r.stop_sequences
    if r.top_p is not None:
        x["top_p"] = r.top_p
    if r.top_k is not None:
        x["top_k"] = r.top_k
    if r.tools:
        x["tools"] = process_tools(r.tools, r.model.startswith("gemini/"))
    if r.tool_choice:
        x["tool_choice"] = process_tool_choice(r.tool_choice)
    return x


def process_tools(t, is_gemini):
    r = []
    for tool in t:
        if hasattr(tool, "dict"):
            td = tool.dict()
        else:
            try:
                td = dict(tool) if not isinstance(tool, dict) else tool
            except Exception:
                logger.error(f"Could not convert tool to dict: {tool}")
                continue
        sch = td.get("input_schema", {})
        if is_gemini:
            sch = clean_gemini_schema(sch)
        r.append(
            {
                "type": "function",
                "function": {
                    "name": td["name"],
                    "description": td.get("description", ""),
                    "parameters": sch,
                },
            }
        )
    return r


def process_tool_choice(tc):
    if hasattr(tc, "dict"):
        t = tc.dict()
    else:
        t = tc
    c = t.get("type")
    if c in ["auto", "any"]:
        return c
    if c == "tool" and "name" in t:
        return {"type": "function", "function": {"name": t["name"]}}
    return "auto"


def extract_response_data(res):
    if hasattr(res, "choices") and hasattr(res, "usage"):
        ch = res.choices
        msg = ch[0].message if ch and len(ch) > 0 else None
        c = msg.content if msg and hasattr(msg, "content") else ""
        t = msg.tool_calls if msg and hasattr(msg, "tool_calls") else None
        f = ch[0].finish_reason if ch and len(ch) > 0 else "stop"
        u = res.usage
        i = getattr(res, "id", f"msg_{uuid.uuid4()}")
        return c, t, f, u, i
    try:
        if isinstance(res, dict):
            rd = res
        else:
            try:
                rd = res.dict()
            except Exception:
                try:
                    rd = res.model_dump() if hasattr(res, "model_dump") else res.__dict__
                except Exception:
                    rd = {
                        "id": getattr(res, "id", f"msg_{uuid.uuid4()}"),
                        "choices": getattr(res, "choices", [{}]),
                        "usage": getattr(res, "usage", {}),
                    }
        ch = rd.get("choices", [{}])
        msg = ch[0].get("message", {}) if ch and len(ch) > 0 else {}
        c = msg.get("content", "")
        t = msg.get("tool_calls", None)
        f = ch[0].get("finish_reason", "stop") if ch and len(ch) > 0 else "stop"
        u = rd.get("usage", {})
        i = rd.get("id", f"msg_{uuid.uuid4()}")
        return c, t, f, u, i
    except Exception as e:
        logger.error(f"Error extracting response data: {e}")
        return "", None, "stop", {}, f"msg_{uuid.uuid4()}"


def process_tool_calls(tc, ic=True):
    c = []
    if not tc:
        return c
    if not isinstance(tc, list):
        tc = [tc]
    if ic:
        for t in tc:
            tid, name, args = extract_tool_info(t)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            c.append({"type": "tool_use", "id": tid, "name": name, "input": args})
    else:
        s = "\n\nTool usage:\n"
        for t in tc:
            tid, name, args = extract_tool_info(t)
            if isinstance(args, str):
                try:
                    d = json.loads(args)
                    a = json.dumps(d, indent=2)
                except Exception:
                    a = args
            else:
                a = json.dumps(args, indent=2)
            s += f"Tool: {name}\nArguments: {a}\n\n"
        c.append({"type": "text", "text": s})
    return c


def extract_tool_info(t):
    if isinstance(t, dict):
        f = t.get("function", {})
        i = t.get("id", f"tool_{uuid.uuid4()}")
        n = f.get("name", "") if isinstance(f, dict) else ""
        a = f.get("arguments", "{}") if isinstance(f, dict) else "{}"
    else:
        f = getattr(t, "function", None)
        i = getattr(t, "id", f"tool_{uuid.uuid4()}")
        n = getattr(f, "name", "") if f else ""
        a = getattr(f, "arguments", "{}") if f else "{}"
    return i, n, a


def extract_usage_info(u):
    if isinstance(u, dict):
        p = u.get("prompt_tokens", 0)
        c = u.get("completion_tokens", 0)
        return p, c
    return getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)


def map_finish_reason(fr):
    if fr == "length":
        return "max_tokens"
    if fr == "tool_calls":
        return "tool_use"
    return "end_turn"


def convert_litellm_to_anthropic(r, o: MessagesRequest) -> MessagesResponse:
    try:
        cm = o.model
        if cm.startswith("anthropic/"):
            cm = cm[len("anthropic/") :]
        elif cm.startswith("gemini/"):
            cm = cm[len("gemini/") :]
        ic = cm.startswith("claude-")
        ct, tc, fr, ui, rid = extract_response_data(r)
        content = []
        if ct:
            content.append({"type": "text", "text": ct})
        if tc:
            cc = process_tool_calls(tc, ic)
            if not ic and content and content[0]["type"] == "text" and cc and cc[0]["type"] == "text":
                content[0]["text"] += cc[0]["text"]
            else:
                content.extend(cc)
        if not content:
            content.append({"type": "text", "text": ""})
        p, c = extract_usage_info(ui)
        sr = map_finish_reason(fr)
        return MessagesResponse(
            id=rid,
            model=o.model,
            role="assistant",
            content=content,
            stop_reason=sr,
            stop_sequence=None,
            usage=Usage(input_tokens=p, output_tokens=c),
        )
    except Exception as e:
        logger.error(f"Error converting response: {e}")
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=o.model,
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": f"Error converting response: {e}. Check server logs.",
                }
            ],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
        )


def create_event(et, d):
    return f"event: {et}\ndata: {json.dumps(d)}\n\n"


def handle_text_block_update(dc, ti, tbc, ts, ac):
    ev = []
    ts_out = ts
    if dc and ti is None and not tbc:
        ts_out = True
        ev.append(
            create_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": dc},
                },
            )
        )
    return ev, ts_out


def handle_text_block_close(ts, tbc, ac):
    ev = []
    if ts and not tbc:
        ev.append(create_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
        return ev, True
    if ac and not ts and not tbc:
        ev.append(
            create_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": ac},
                },
            )
        )
        ev.append(create_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
        return ev, True
    if not tbc:
        ev.append(create_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
        return ev, True
    return ev, tbc


def extract_tool_streaming_info(tc):
    if isinstance(tc, dict):
        f = tc.get("function", {})
        n = f.get("name", "") if isinstance(f, dict) else ""
        i = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
        a = None
        if "function" in tc:
            a = f.get("arguments", "") if isinstance(f, dict) else ""
        cidx = getattr(tc, "index", tc.get("index", 0))
        return cidx, i, n, a
    f = getattr(tc, "function", None)
    n = getattr(f, "name", "") if f else ""
    i = getattr(tc, "id", f"toolu_{uuid.uuid4().hex[:24]}")
    a = getattr(f, "arguments", "") if f else ""
    cidx = getattr(tc, "index", 0)
    return cidx, i, n, a


def close_all_blocks(ti, lti, tbc, ac, ts):
    ev = []
    if ti is not None:
        for x in range(1, lti + 1):
            ev.append(create_event("content_block_stop", {"type": "content_block_stop", "index": x}))
    if not tbc:
        if ac and not ts:
            ev.append(
                create_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": ac},
                    },
                )
            )
        ev.append(create_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    return ev


async def handle_streaming(rg, o: MessagesRequest):
    try:
        mid = f"msg_{uuid.uuid4().hex[:24]}"
        md = {
            "type": "message_start",
            "message": {
                "id": mid,
                "type": "message",
                "role": "assistant",
                "model": o.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        }
        yield create_event("message_start", md)
        yield create_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        yield create_event("ping", {"type": "ping"})
        ti = None
        ac = ""
        ts = False
        tbc = False
        ot = 0
        sent_sr = False
        lti = 0
        async for chunk in rg:
            if hasattr(chunk, "usage") and chunk.usage and hasattr(chunk.usage, "completion_tokens"):
                ot = chunk.usage.completion_tokens
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", getattr(choice, "message", {}))
            fr = getattr(choice, "finish_reason", None)
            dc = getattr(
                delta,
                "content",
                delta.get("content") if isinstance(delta, dict) else None,
            )
            if dc:
                ac += dc
                e, ts = handle_text_block_update(dc, ti, tbc, ts, ac)
                for ev in e:
                    yield ev
            dtc = getattr(
                delta,
                "tool_calls",
                delta.get("tool_calls") if isinstance(delta, dict) else None,
            )
            if dtc:
                if ti is None:
                    e, tbc = handle_text_block_close(ts, tbc, ac)
                    for ev in e:
                        yield ev
                if not isinstance(dtc, list):
                    dtc = [dtc]
                for ctc in dtc:
                    cidx, iid, name, args = extract_tool_streaming_info(ctc)
                    if ti is None or cidx != ti:
                        ti = cidx
                        lti += 1
                        yield create_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": lti,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": iid,
                                    "name": name,
                                    "input": {},
                                },
                            },
                        )
                    if args:
                        try:
                            aj = json.dumps(args) if isinstance(args, dict) else args
                        except Exception:
                            aj = args
                        yield create_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": lti,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": aj,
                                },
                            },
                        )
            if fr and not sent_sr:
                sent_sr = True
                ce = close_all_blocks(ti, lti, tbc, ac, ts)
                for ev in ce:
                    yield ev
                sr = map_finish_reason(fr)
                yield create_event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": sr, "stop_sequence": None},
                        "usage": {"output_tokens": ot},
                    },
                )
                yield create_event("message_stop", {"type": "message_stop"})
                yield "data: [DONE]\n\n"
                return
        if not sent_sr:
            ce = close_all_blocks(ti, lti, tbc, ac, ts)
            for ev in ce:
                yield ev
            yield create_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": ot},
                },
            )
            yield create_event("message_stop", {"type": "message_stop"})
            yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        yield create_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "error", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        )
        yield create_event("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"


def format_tool_result_content(b, it="result"):
    if isinstance(b, dict):
        if b.get("type") == "tool_result":
            rc = b.get("content", [])
            s = parse_tool_result_content(rc)
            return f"Tool {it}:\n{s}"
        elif b.get("type") == "text":
            return b.get("text", "")
        elif b.get("type") == "tool_use":
            n = b.get("name", "unknown")
            i = b.get("id", "unknown")
            inp = json.dumps(b.get("input", {}))
            return f"[Tool: {n} (ID: {i})]\nInput: {inp}"
        elif b.get("type") == "image":
            return "[Image content - not displayed in text format]"
    return ""


def process_complex_content(mc):
    only_tr = all(isinstance(b, dict) and b.get("type") == "tool_result" for b in mc if isinstance(b, dict))
    if only_tr and mc:
        return "\n".join(format_tool_result_content(b) for b in mc if isinstance(b, dict))
    return "\n".join(format_tool_result_content(b, "Result") for b in mc if isinstance(b, dict))


def prepare_gemini_request(lr):
    if "gemini" not in lr["model"] or "messages" not in lr:
        return
    for i, m in enumerate(lr["messages"]):
        if "content" in m and isinstance(m["content"], list):
            lr["messages"][i]["content"] = process_complex_content(m["content"]) or "..."
        elif m.get("content") is None:
            lr["messages"][i]["content"] = "..."
        for k in list(m.keys()):
            if k not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                del m[k]


class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def log_request_beautifully(method, path, cm, gm, nm, nt, sc):
    cd = f"{Colors.CYAN}{cm}{Colors.RESET}"
    gd = f"{Colors.GREEN}{gm.split('/')[-1] if '/' in gm else gm}{Colors.RESET}"
    ep = path.split("?")[0] if "?" in path else path
    ts = f"{Colors.MAGENTA}{nt} tools{Colors.RESET}"
    ms = f"{Colors.BLUE}{nm} messages{Colors.RESET}"
    st = f"{Colors.GREEN}✓ {sc} OK{Colors.RESET}" if sc == 200 else f"{Colors.RED}✗ {sc}{Colors.RESET}"
    print(f"{Colors.BOLD}{method} {ep}{Colors.RESET} {st}")
    print(f"{cd} → {gd} {ts} {ms}")
    sys.stdout.flush()


@app.post("/v1/messages")
async def create_message(r: MessagesRequest, rr: Request):
    try:
        b = await rr.body()
        bj = json.loads(b.decode("utf-8"))
        om = bj.get("model", "unknown")
        dm = om.split("/")[-1] if "/" in om else om
        lreq = convert_anthropic_to_litellm(r)
        lreq["api_key"] = GEMINI_API_KEY if r.model.startswith("gemini/") else ANTHROPIC_API_KEY
        prepare_gemini_request(lreq)
        nt = len(r.tools) if r.tools else 0
        log_request_beautifully("POST", rr.url.path, dm, lreq.get("model"), len(lreq["messages"]), nt, 200)
        if r.stream:
            rg = await litellm.acompletion(**lreq)
            return StreamingResponse(handle_streaming(rg, r), media_type="text/event-stream")
        st = time.time()
        resp = litellm.completion(**lreq)
        logger.debug(f"✅ RESPONSE: Model={lreq.get('model')}, Time={time.time() - st:.2f}s")
        return convert_litellm_to_anthropic(resp, r)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/v1/messages/count_tokens")
async def count_tokens(r: TokenCountRequest, rr: Request):
    try:
        om = r.original_model or r.model
        dm = om.split("/")[-1] if "/" in om else om
        cr = convert_anthropic_to_litellm(
            MessagesRequest(
                model=r.model,
                max_tokens=100,
                messages=r.messages,
                system=r.system,
                tools=r.tools,
                tool_choice=r.tool_choice,
                thinking=r.thinking,
            )
        )
        from litellm import token_counter

        nt = len(r.tools) if r.tools else 0
        log_request_beautifully("POST", rr.url.path, dm, cr.get("model"), len(cr["messages"]), nt, 200)
        tc = token_counter(model=cr["model"], messages=cr["messages"])
        return TokenCountResponse(input_tokens=tc)
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Anthropic to Gemini Proxy"}


def check_gemini_api_key():
    if not GEMINI_API_KEY:
        print(f"{Colors.RED}[ERROR] GEMINI_API_KEY environment variable is not set.{Colors.RESET}")
        print("Please set it with: export GEMINI_API_KEY=your_api_key")
        return False
    return True


def is_server_running():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("localhost", 8082))
        s.close()
        return True, None
    except Exception:
        s.close()
        return False, None


def launch_claude():
    cp = shutil.which("claude")
    if not cp:
        print(f"{Colors.RED}[ERROR] Claude Code not found in your PATH.{Colors.RESET}")
        print("Please install it with: npm install -g @anthropic-ai/claude-code")
        return False
    print(f"{Colors.GREEN}Launching Claude Code connected to Gemini...{Colors.RESET}")
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = "http://localhost:8082"
    subprocess.Popen([cp], env=env)
    return True


def main():
    print(f"{Colors.BOLD}{Colors.GREEN}Gemini Proxy for Claude Code{Colors.RESET}")
    if not check_gemini_api_key():
        sys.exit(1)
    sr, pid = is_server_running()
    if sr:
        pi = f" (PID: {pid})" if pid else ""
        print(f"{Colors.GREEN}Server is already running on port 8082{pi}{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}Starting proxy server...{Colors.RESET}")
        try:
            sn = "gemini_proxy"
            gak = os.environ.get("GEMINI_API_KEY", "")
            scmd = f"export GEMINI_API_KEY='{gak}' && gemini-server"
            subprocess.run(["screen", "-dmS", sn, "bash", "-c", scmd])
            print(f"{Colors.GREEN}Started proxy server in screen session '{sn}'{Colors.RESET}")
            print(f"To view the server logs, run: screen -r {sn}")
            print("To detach from the screen session (keep it running), press Ctrl+A, then D")
            for _ in range(5):
                time.sleep(1)
                sr, _ = is_server_running()
                if sr:
                    print(f"{Colors.GREEN}Server started successfully!{Colors.RESET}")
                    break
        except Exception as e:
            print(f"{Colors.RED}Error starting server: {str(e)}{Colors.RESET}")
            sys.exit(1)
    launch_claude()


def run_server():
    print(f"{Colors.BOLD}{Colors.GREEN}Gemini Proxy for Claude Code - Server{Colors.RESET}")
    if not check_gemini_api_key():
        sys.exit(1)
    sr, pid = is_server_running()
    if sr:
        pi = f" (PID: {pid})" if pid else ""
        print(f"{Colors.GREEN}Server is already running on port 8082{pi}{Colors.RESET}")
        return
    print(f"{Colors.YELLOW}Starting proxy server...{Colors.RESET}")
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")


if __name__ == "__main__":
    main()
