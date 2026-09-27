import numpy as np
import pandas as pd


TARGET_COLUMN = "isFraud"

NON_FEATURE_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    TARGET_COLUMN,
}


def prepare_model_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare engineered transaction data for ML training/inference.

    - Separates target from features.
    - Removes identifiers/time columns that should not directly
      be used as model features.
    - Keeps numeric features only.
    - Replaces infinite values.
    - Fills missing numeric values using training-safe defaults.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    result = df.copy()

    y = result[TARGET_COLUMN].astype("int8")

    feature_columns = [
        column
        for column in result.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X = result[feature_columns].copy()

    # Keep numeric columns only for the baseline model.
    X = X.select_dtypes(include=[np.number])

    # Replace positive/negative infinity.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Clean tiny floating-point noise.
    X = X.mask(X.abs() < 1e-12, 0)

    # Fill missing numeric values.
    X = X.fillna(0)

    return X, y