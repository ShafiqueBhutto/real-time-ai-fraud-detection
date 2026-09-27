import pandas as pd

from ml.features import create_card_history_features


def prepare_features_with_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create leakage-safe behavioral features.

    Transactions are ordered deterministically by time and TransactionID
    before historical features are calculated.
    """
    required_columns = {
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "card1",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = (
        df.sort_values(
            ["TransactionDT", "TransactionID"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    result = create_card_history_features(result)

    return result