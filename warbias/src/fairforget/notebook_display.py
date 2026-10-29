"""Display notebook tables from left to right with left-aligned text."""

import re

from IPython.display import display as display_object
from pandas import DataFrame


def display(value, *, selected=None):
    """Format a table or pass an ordinary value to the notebook display."""
    if selected is not None:
        if not isinstance(value, DataFrame):
            raise ValueError("Row selection requires a DataFrame")
        try:
            selected = list(selected)
        except TypeError as error:
            raise ValueError("Row selection needs one boolean per table row") from error
        if len(selected) != len(value) or any(type(flag) is not bool for flag in selected):
            raise ValueError("Row selection needs one boolean per table row")
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
    if selected is not None:
        prefix, body = table.split("<tbody>", 1)
        body, suffix = body.split("</tbody>", 1)
        flags = iter(selected)
        body = re.sub(
            r"<tr(?=[\s>])",
            lambda match: (
                '<tr style="background-color: #E2EFDA;"' if next(flags) else match.group()
            ),
            body,
        )
        table = prefix + "<tbody>" + body + "</tbody>" + suffix
    return display_object({"text/html": table, "text/plain": repr(value)}, raw=True)
