import pandas as pd

from ml.dataset import prepare_features_with_history


def main():
    df = pd.DataFrame(
        {
            "TransactionID": [1001, 1002, 1003, 1004, 1005],
            "TransactionDT": [100, 200, 400, 1000, 1100],
            "TransactionAmt": [100.0, 50.0, 200.0, 75.0, 300.0],
            "card1": [1, 1, 1, 1, 1],
        }
    )

    result = prepare_features_with_history(df)

    print("\nPrepared dataset:")
    print(
        result[
            [
                "TransactionID",
                "TransactionDT",
                "TransactionAmt",
                "card1_previous_transaction_count",
                "card1_transactions_last_5min",
                "card1_transactions_last_1h",
                "card1_previous_avg_amount",
            ]
        ].to_string(index=False)
    )

    # First transaction has no previous history
    assert result.loc[0, "card1_previous_transaction_count"] == 0

    # Second transaction has one previous transaction
    assert result.loc[1, "card1_previous_transaction_count"] == 1

    # Third transaction has two previous transactions
    assert result.loc[2, "card1_previous_transaction_count"] == 2

    # Previous average before transaction 3 = (100 + 50) / 2
    assert result.loc[2, "card1_previous_avg_amount"] == 75.0

    print("\nLeakage-safe history test PASSED.")


if __name__ == "__main__":
    main()