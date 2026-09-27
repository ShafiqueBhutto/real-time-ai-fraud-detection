import pandas as pd

from ml.dataset import prepare_features_with_history


TRANSACTION_PATH = "data/raw/train_transaction.csv"


def main():
    print("Loading IEEE-CIS transaction data...")

    df = pd.read_csv(
        TRANSACTION_PATH,
        usecols=[
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "card1",
            "ProductCD",
            "isFraud",
        ],
        nrows=50_000,
    )

    print("Original shape:", df.shape)

    result = prepare_features_with_history(df)

    print("Prepared shape:", result.shape)

    feature_columns = [
        "card1_previous_transaction_count",
        "card1_transactions_last_5min",
        "card1_transactions_last_1h",
        "card1_transactions_last_24h",
        "card1_amount_total_last_5min",
        "card1_amount_total_last_1h",
        "card1_amount_total_last_24h",
        "card1_previous_avg_amount",
        "card1_amount_vs_previous_avg",
        "card1_amount_zscore",
        "card1_product_previous_count",
    ]

    print("\nHistorical feature summary:")
    print(result[feature_columns].describe().T)

    print("\nFirst 10 transactions:")
    print(
        result[
            [
                "TransactionID",
                "TransactionDT",
                "TransactionAmt",
                "card1",
                "card1_previous_transaction_count",
                "card1_transactions_last_5min",
                "card1_previous_avg_amount",
                "isFraud",
            ]
        ].head(10).to_string(index=False)
    )

    # Basic validation
    assert len(result) == len(df)
    assert result["TransactionID"].is_unique

    # First transaction cannot have previous history
    assert result.iloc[0]["card1_previous_transaction_count"] == 0

    print("\nReal IEEE-CIS feature preparation test PASSED.")


if __name__ == "__main__":
    main()