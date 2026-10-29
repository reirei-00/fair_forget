"""Display notebook tables from left to right with left-aligned text."""

import re

from IPython.display import display as display_object
from pandas import DataFrame


def display(value):
    """Format a table or pass an ordinary value to the notebook display."""
    if not isinstance(value, DataFrame):
        return display_object(value)
    table = value.to_html(justify="left", escape=True)
    table = table.replace(
        "<table ",
        '<table dir="ltr" style="text-align: left; margin-left: 0; margin-right: auto;" ',
        1,
    )
    table = re.sub(
        r"<(th|td)(?=[\s>])",
        r'<\1 style="text-align: left; direction: ltr;"',
        table,
    )
    return display_object({"text/html": table, "text/plain": repr(value)}, raw=True)
