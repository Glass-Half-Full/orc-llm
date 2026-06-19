from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .dataframe import evaluate_rows, rows_from_payload
from .model import Message, model
from .prompts import normalize_verdict, policy_prompt


app = FastAPI(title="orc-llm", version="0.1.0")


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a precise, concise assistant."
    temperature: float | None = None
    max_tokens: int | None = None


class PolicyRule(BaseModel):
    id: str
    description: str
    severity: str | None = None


class DataframeEvaluationRequest(BaseModel):
    policy: dict[str, list[PolicyRule]]
    dataframe: dict[str, Any]
    text_column: str
    output_column: str = "policy_result"
    output_format: Literal["verdict", "raw"] = "verdict"
    max_tokens: int = Field(default=32, ge=1, le=512)


@app.get("/health")
def health() -> dict[str, Any]:
    settings = model.settings
    return {
        "status": "ok",
        "model_loaded": model.is_loaded,
        "model_path": str(settings.model_path),
        "model_exists": settings.model_path.exists(),
        "n_ctx": settings.n_ctx,
        "n_gpu_layers": settings.n_gpu_layers,
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest) -> dict[str, Any]:
    try:
        result = model.chat(
            request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model.settings.model_path.name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/v1/generate")
def generate(request: GenerateRequest) -> dict[str, str]:
    try:
        result = model.complete(
            request.prompt,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"text": result.text}


@app.post("/v1/evaluate-dataframe")
def evaluate_dataframe(request: DataframeEvaluationRequest) -> dict[str, Any]:
    try:
        rows, columns = rows_from_payload(request.dataframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rules = [rule.model_dump(exclude_none=True) for rule in request.policy.get("rules", [])]

    def evaluator(text: str) -> str:
        prompt = policy_prompt(text, rules)
        result = model.complete(
            prompt,
            system_prompt="You are a strict policy compliance classifier.",
            temperature=0,
            max_tokens=request.max_tokens,
        )
        if request.output_format == "raw":
            return result.text.strip()
        return normalize_verdict(result.text)

    try:
        output_rows, output_columns = evaluate_rows(
            rows=rows,
            columns=columns,
            text_column=request.text_column,
            output_column=request.output_column,
            evaluator=evaluator,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "columns": output_columns,
        "rows": output_rows,
    }

