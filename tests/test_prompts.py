import unittest

from orc_llm.prompts import normalize_verdict


class PromptTests(unittest.TestCase):
    def test_normalize_verdict(self) -> None:
        self.assertEqual(normalize_verdict("violates_policy"), "violates_policy")
        self.assertEqual(normalize_verdict("This meets policy."), "meets_policy")
        self.assertEqual(normalize_verdict("unclear"), "needs_review")


if __name__ == "__main__":
    unittest.main()
