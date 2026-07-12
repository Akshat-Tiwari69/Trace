"""Headless Streamlit interaction smoke for production-only UI contracts."""

from streamlit.testing.v1 import AppTest

from src.app.app import FLOOD_MODE


def test_app_renders_and_exposes_keyboard_flood_controls():
    app = AppTest.from_file("src/app/app.py").run(timeout=45)
    assert not app.exception
    assert {"Briefing", "Analysis", "Your imagery", "Methodology"}.issubset(
        {tab.label for tab in app.tabs}
    )

    failure_mode = next(radio for radio in app.radio if radio.label == "Failure mode")
    failure_mode.set_value(FLOOD_MODE).run(timeout=45)

    assert not app.exception
    assert any(
        widget.label == "Junctions inside the failed area"
        for widget in app.multiselect
    )
    junction_picker = next(
        widget for widget in app.selectbox if widget.label == "Junction to disable"
    )
    assert junction_picker.disabled
