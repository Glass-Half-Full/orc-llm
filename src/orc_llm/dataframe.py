from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


Row = dict[str, Any]


def rows_from_payload(dataframe: dict[str, Any]) -> tuple[list[Row], list[str]]:
    if "rows" in dataframe and dataframe["rows"] is not None:
        rows = dataframe["rows"]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("dataframe.rows must be a list of objects")
        columns = list(dataframe.get("columns") or _columns_from_rows(rows))
        return [dict(row) for row in rows], columns

    columns = dataframe.get("columns")
    data = dataframe.get("data")
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise ValueError("split-style dataframe requires string columns")
    if not isinstance(data, list) or any(not isinstance(row, list) for row in data):
        raise ValueError("split-style dataframe requires data as a list of lists")

    rows = []
    for values in data:
        if len(values) != len(columns):
            raise ValueError("split-style dataframe row length does not match columns")
        rows.append(dict(zip(columns, values, strict=True)))
    return rows, list(columns)


def insert_column_after(columns: Sequence[str], source_column: str, output_column: str) -> list[str]:
    if source_column not in columns:
        raise ValueError(f"text_column not found: {source_column}")

    next_columns = [column for column in columns if column != output_column]
    index = next_columns.index(source_column) + 1
    next_columns.insert(index, output_column)
    return next_columns


def evaluate_rows(
    *,
    rows: Sequence[Row],
    columns: Sequence[str],
    text_column: str,
    output_column: str,
    evaluator: Callable[[str], str],
) -> tuple[list[Row], list[str]]:
    output_columns = insert_column_after(columns, text_column, output_column)
    output_rows = []
    for row in rows:
        next_row = dict(row)
        next_row[output_column] = evaluator(str(row.get(text_column, "")))
        output_rows.append({column: next_row.get(column) for column in output_columns})
    return output_rows, output_columns


def _columns_from_rows(rows: Sequence[Row]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns

