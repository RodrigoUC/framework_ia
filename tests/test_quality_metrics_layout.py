"""Regression coverage for responsive dataset-quality metric rows."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


APP_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "framework_ia"
    / "ui"
    / "streamlit_app.py"
)


class QualityMetricsLayoutTests(unittest.TestCase):
    """Keep quality metrics in readable rows instead of one crowded strip."""

    def test_quality_metrics_are_rendered_in_rows_of_at_most_three(self) -> None:
        module = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_resumen_calidad"
        )

        chunk_size = next(
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "metricas_por_fila"
                for target in node.targets
            )
        )
        self.assertIsInstance(chunk_size, ast.Constant)
        self.assertEqual(chunk_size.value, 3)

        columns_call = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
            and node.func.attr == "columns"
        )
        self.assertEqual(len(columns_call.args), 1)
        self.assertIsInstance(columns_call.args[0], ast.Call)
        self.assertIsInstance(columns_call.args[0].func, ast.Name)
        self.assertEqual(columns_call.args[0].func.id, "len")
        self.assertIsInstance(columns_call.args[0].args[0], ast.Name)
        self.assertEqual(columns_call.args[0].args[0].id, "fila")


if __name__ == "__main__":
    unittest.main()
