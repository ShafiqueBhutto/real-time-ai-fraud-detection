import numpy as np
import pandas as pd
import pytest

from ml.preprocessing import prepare_model_features


def test_prepare_model_features():
    df = pd.DataFrame(
        {
            "TransactionID": [1001, 1002, 1003],
            "TransactionDT": [100, 200, 300],
            "TransactionAmt": [100.0, np.nan, 300.0],
            "card1": [1, 1, 2],
            "history_feature": [10.0, np.inf, 30.0],
            "isFraud": [0, 1, 0],
        }
    )

    X, y = prepare_model_features(df)

    assert "TransactionID" not in X.columns
    assert "TransactionDT" not in X.columns
    assert "isFraud" not in X.columns

    assert X.isna().sum().sum() == 0
    assert np.isinf(X.select_dtypes(include=[np.number])).sum().sum() == 0

    assert y.tolist() == [0, 1, 0]


def test_missing_target_raises_error():
    df = pd.DataFrame(
        {
            "TransactionID": [1001],
            "TransactionAmt": [100.0],
        }
    )

    with pytest.raises(ValueError):
        prepare_model_features(df)