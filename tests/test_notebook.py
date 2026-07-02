import unittest

from orc_llm.model import CompletionResult
from orc_llm.notebook import render_prompt


try:
    import pandas as pd
except ImportError:
    pd = None


class FakeModel:
    def __init__(self) -> None:
        self.prompts = []

    def complete(self, prompt: str, **kwargs):
        self.prompts.append((prompt, kwargs))
        return CompletionResult(text=f"out:{len(self.prompts)}", raw={"usage": {"completion_tokens": 2}})


class NotebookTests(unittest.TestCase):
    def test_render_prompt_uses_text_placeholder(self) -> None:
        prompt = render_prompt(
            "Summarise: {text}",
            text="hello",
            row={"notes": "hello"},
            text_column="notes",
        )

        self.assertEqual(prompt, "Summarise: hello")

    def test_render_prompt_appends_text_when_no_placeholder(self) -> None:
        prompt = render_prompt(
            "Summarise this.",
            text="hello",
            row={"notes": "hello"},
            text_column="notes",
        )

        self.assertEqual(prompt, "Summarise this.\n\nnotes:\nhello\n\nOutput:")

    @unittest.skipIf(pd is None, "pandas not installed")
    def test_apply_prompt_to_column_inserts_adjacent_output(self) -> None:
        from orc_llm.notebook import apply_prompt_to_column

        frame = pd.DataFrame({"id": [1, 2], "notes": ["alpha", "beta"], "other": ["x", "y"]})
        model = FakeModel()

        result = apply_prompt_to_column(
            frame,
            text_column="notes",
            output_column="llm_output",
            prompt="Classify: {text}",
            model=model,
            max_tokens=4,
        )

        self.assertEqual(list(result.columns), ["id", "notes", "llm_output", "other"])
        self.assertEqual(result["llm_output"].tolist(), ["out:1", "out:2"])
        self.assertEqual(model.prompts[0][0], "Classify: alpha")
        self.assertEqual(model.prompts[0][1]["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
