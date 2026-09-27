import pandas as pd
import pytest

from ml.split import chronological_split


def test_chronological_split_preserves_time_order():
    df = pd.DataFrame(
        {
            "TransactionDT": [500, 100, 400, 200, 300, 600, 700, 800, 900, 1000],
            "isFraud": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        }
    )

    train, validation, test = chronological_split(df)

    assert len(train) == 7
    assert len(validation) == 1
    assert len(test) == 2

    assert train["TransactionDT"].max() < validation["TransactionDT"].min()
    assert validation["TransactionDT"].max() < test["TransactionDT"].min()


def test_chronological_split_keeps_all_rows():
    df = pd.DataFrame(
        {
            "TransactionDT": range(20),
        }
    )

    train, validation, test = chronological_split(df)

    assert len(train) + len(validation) + len(test) == len(df)


def test_chronological_split_requires_transaction_dt():
    df = pd.DataFrame(
        {
            "TransactionAmt": [10.0, 20.0, 30.0],
        }
    )

    with pytest.raises(ValueError, match="TransactionDT"):
        chronological_split(df)