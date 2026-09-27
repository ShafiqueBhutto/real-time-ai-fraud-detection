import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
):
    """
    Split transactions chronologically.

    The oldest transactions go to training,
    followed by validation, then the newest transactions
    go to the test set.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    if "TransactionDT" not in df.columns:
        raise ValueError("DataFrame must contain TransactionDT.")

    result = df.sort_values(
        "TransactionDT",
        kind="mergesort",
    ).reset_index(drop=True)

    total_rows = len(result)

    train_end = int(total_rows * train_ratio)
    validation_end = int(
        total_rows * (train_ratio + validation_ratio)
    )

    train = result.iloc[:train_end].copy()
    validation = result.iloc[train_end:validation_end].copy()
    test = result.iloc[validation_end:].copy()

    return train, validation, test