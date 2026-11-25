from PIL import Image

from src.paper_quantitative import LEVELS, METHOD_LABELS, plot_performance, write_tables


def example_groups():
    rows = {}
    for i, method in enumerate(METHOD_LABELS):
        for level, count, csr in zip(LEVELS, (1, 1, 4), (0.5, 1, 0)):
            rows[method, level] = {
                "n_tasks": count, "OR": 0.01 * i, "BOR": 0.2, "CPR": 0.75,
                "AS": 0.8, "CSR": csr if i < 2 else 0.1,
                "rounds_used": 1.125 + 0.1 * i, "total_tokens": 1234.1 + 100 * i,
            }
    rows["self_refine", "easy"]["CSR"] += 0.00001
    return rows


def test_tables_weight_task_means_and_bold_displayed_ties(tmp_path):
    write_tables(example_groups(), tmp_path, mock=True)
    main = (tmp_path / "table_main.tex").read_text()
    assert main.startswith(r"\begin{table*}[t]")
    assert r"\caption{Offline mock data." in main
    assert r"OR (\%) $\downarrow$" in main and r"AS (\%) $\uparrow$" in main
    rows = {r.split(" & ")[0]: r.split(" & ")[1:] for r in main.splitlines() if " & " in r}
    assert rows["Single-Agent"][0] == r"\textbf{0.0}"
    assert rows["Self-Refine"][0] == "1.0"
    assert rows["Single-Agent"][4] == rows["Self-Refine"][4] == r"\textbf{25.0}"
    assert rows["Single-Agent"][5] == rows["Self-Refine"][5] == r"\textbf{25.0}"
    assert rows["Single-Agent"][7].startswith(r"\textbf{1,234}")
    by_difficulty = (tmp_path / "table_difficulty.tex").read_text()
    assert by_difficulty.count(r"\multicolumn{2}{c}") == 3
    assert r"Self-Refine & \textbf{50.0} & 1,334" in by_difficulty
    write_tables(example_groups(), tmp_path, mock=False)
    assert "mock" not in (tmp_path / "table_main.tex").read_text().lower()


def test_fig2_exports_two_column_pdf_and_300_dpi_png(tmp_path):
    plot_performance(example_groups(), tmp_path, mock=True)
    assert (tmp_path / "fig2_performance.pdf").read_bytes().startswith(b"%PDF-")
    with Image.open(tmp_path / "fig2_performance.png") as image:
        assert image.size == (2130, 780)
        assert all(abs(value - 300) < 0.05 for value in image.info["dpi"])
