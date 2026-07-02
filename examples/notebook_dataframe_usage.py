# %%
import pandas as pd

from orc_llm import apply_prompt_to_column, benchmark_tokens_per_second, load_model


# %%
model = load_model(
    n_ctx=4096,
    n_gpu_layers=-1,
    n_batch=512,
    n_ubatch=512,
)


# %%
df = pd.DataFrame(
    {
        "id": [1, 2],
        "notes": [
            "Customer asked for a refund after the warranty expired.",
            "Customer shared payment card details in plain text.",
        ],
    }
)

prompt = """Classify this note as one of:
- meets_policy
- violates_policy
- needs_review

Note:
{text}

Return only the label."""

result = apply_prompt_to_column(
    df,
    text_column="notes",
    output_column="policy_result",
    prompt=prompt,
    model=model,
    max_tokens=32,
)
result


# %%
benchmark_tokens_per_second(model=model, max_tokens=32)
