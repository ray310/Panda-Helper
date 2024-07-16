"""Test-related utility functions."""

import pandas as pd


def make_category_data(cat_name, start, end, freq):
    """Return pd.DataFrame of arbitrary data for specified 'category'."""
    rng = pd.date_range(start, end, freq=freq, inclusive="left")
    data = {
        "A": list(range(1, len(rng) + 1, 1)),
        "B": [chr(ord("A") + (x % 26)) for x in range(0, len(rng), 1)],
        "C": [float((-1) ** (x % 2) * x) for x in range(0, len(rng), 1)],
    }
    df = pd.DataFrame(data, index=rng)
    df["category"] = cat_name
    return df
