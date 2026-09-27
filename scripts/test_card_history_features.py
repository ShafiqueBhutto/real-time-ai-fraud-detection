import pandas as pd

from ml.features import create_card_history_features


DATA_PATH = "data/raw/train_transaction.csv"


def main():
    df = pd.read_csv(
        DATA_PATH,
        nrows=5000,
        usecols=[
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "card1",
            "ProductCD",
        ],
    )

    print("Original shape:", df.shape)

    result = create_card_history_features(df)

    print("New shape:", result.shape)

    feature_columns = [
        "card1_previous_transaction_count",
        "card1_product_previous_count",
        "card1_previous_avg_amount",
        "card1_amount_vs_previous_avg",
        "card1_transactions_last_5min",
        "card1_transactions_last_1h",
        "card1_amount_total_last_5min",
        "card1_amount_total_last_1h",
    ]

    print("\nFeature summary:")
    print(result[feature_columns].describe())

    print("\nFirst 10 rows:")
    print(
        result[
            [
                "TransactionID",
                "card1",
                "TransactionAmt",
                "card1_transactions_last_5min",
                "card1_amount_total_last_5min",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()