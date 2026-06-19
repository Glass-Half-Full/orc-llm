import unittest

from orc_llm.dataframe import evaluate_rows, insert_column_after, rows_from_payload


class DataframeTests(unittest.TestCase):
    def test_rows_payload_preserves_columns(self) -> None:
        rows, columns = rows_from_payload(
            {
                "rows": [
                    {"id": 1, "notes": "hello"},
                    {"id": 2, "notes": "world", "extra": True},
                ]
            }
        )

        self.assertEqual(
            rows,
            [
                {"id": 1, "notes": "hello"},
                {"id": 2, "notes": "world", "extra": True},
            ],
        )
        self.assertEqual(columns, ["id", "notes", "extra"])

    def test_split_payload_to_rows(self) -> None:
        rows, columns = rows_from_payload(
            {
                "columns": ["id", "notes"],
                "data": [[1, "hello"], [2, "world"]],
            }
        )

        self.assertEqual(columns, ["id", "notes"])
        self.assertEqual(
            rows,
            [{"id": 1, "notes": "hello"}, {"id": 2, "notes": "world"}],
        )

    def test_insert_column_after_source(self) -> None:
        self.assertEqual(
            insert_column_after(["id", "notes"], "notes", "policy_result"),
            [
                "id",
                "notes",
                "policy_result",
            ],
        )

    def test_evaluate_rows_inserts_adjacent_output(self) -> None:
        rows, columns = evaluate_rows(
            rows=[{"id": 1, "notes": "hello"}],
            columns=["id", "notes"],
            text_column="notes",
            output_column="policy_result",
            evaluator=lambda text: f"checked:{text}",
        )

        self.assertEqual(columns, ["id", "notes", "policy_result"])
        self.assertEqual(
            rows,
            [{"id": 1, "notes": "hello", "policy_result": "checked:hello"}],
        )


if __name__ == "__main__":
    unittest.main()
