"""Notebook selection colors preserve escaped text and existing table layout."""

import pytest
from pandas import DataFrame

from fairforget import notebook_display


@pytest.fixture
def rendered(monkeypatch):
    calls = []
    monkeypatch.setattr(
        notebook_display, "display_object", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    return calls


def test_default_table_html_is_unchanged(rendered):
    frame = DataFrame({"label": ["<tag>"]})
    notebook_display.display(frame)
    assert rendered == [
        (
            (
                {
                    "text/html": """<table dir="ltr" style="text-align: left; margin-left: 0; margin-right: auto;" border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th style="text-align: left; direction: ltr;"></th>
      <th style="text-align: left; direction: ltr;">label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style="text-align: left; direction: ltr;">0</th>
      <td style="text-align: left; direction: ltr;">&lt;tag&gt;</td>
    </tr>
  </tbody>
</table>""",
                    "text/plain": repr(frame),
                },
            ),
            {"raw": True},
        )
    ]


def test_selected_rows_are_green_and_cells_remain_escaped(rendered):
    frame = DataFrame({"<script>": ["<img src=x onerror=alert(1)>", "A & B"]})
    notebook_display.display(frame, selected=[True, False])
    payload = rendered[0][0][0]
    html = payload["text/html"]
    header, body = html.split("<tbody>")
    assert "#E2EFDA" not in header
    assert body.count('style="background-color: #E2EFDA;"') == 1
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html and "A &amp; B" in html
    assert 'dir="ltr"' in html and 'style="text-align: left; direction: ltr;"' in html
    assert payload["text/plain"] == repr(frame)


@pytest.mark.parametrize("mask", [[True], [True, False, True], [True, 1], [True, "false"], True])
def test_row_selection_requires_one_boolean_per_row(rendered, mask):
    with pytest.raises(ValueError, match="one boolean per table row"):
        notebook_display.display(DataFrame({"value": [1, 2]}), selected=mask)
    assert rendered == []


def test_ordinary_values_keep_their_original_display_path(rendered):
    value = {"status": "ready"}
    notebook_display.display(value)
    assert rendered == [((value,), {})]
    with pytest.raises(ValueError, match="requires a DataFrame"):
        notebook_display.display(value, selected=[])


def test_empty_selected_table_has_no_colored_rows(rendered):
    notebook_display.display(DataFrame({"value": []}), selected=[])
    assert "#E2EFDA" not in rendered[0][0][0]["text/html"]
