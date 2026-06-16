"""
tests/test_results_panel.py

A standalone script for testing ResultsPanel in isolation.
Run with:
    py tests/test_results_panel.py

This script creates a minimal Qt window containing just the
ResultsPanel and feeds it a fake PlotAdvanced result dict.

What to verify:
    1. A tab appears labelled "PlotAdvanced #1".
    2. The tab header contains an "Open interactive plot" button.
    3. Clicking the button opens test_images/test_plot.html in your browser.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication

# Add the project root to the Python path so imports work correctly.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui.results_panel import ResultsPanel


def main():
    app = QApplication(sys.argv)

    # ResultsPanel expects a controller but only uses it for signals
    # that are not triggered in this test — None is safe here.
    panel = ResultsPanel(controller=None)

    html_path = project_root / "test_images" / "test_plot.html"
    if not html_path.exists():
        print(f"Warning: test HTML file not found at {html_path}")
        print("Create test_images/test_plot.html before running this test.")

    panel.show_result({
        "operator_name": "PlotAdvanced",
        "html_path": str(html_path),
        "n_rows": 42,
    })

    panel.resize(800, 400)
    panel.setWindowTitle("Widget test — ResultsPanel")
    panel.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
