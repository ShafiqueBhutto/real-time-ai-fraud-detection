from pathlib import Path

import pandas as pd

from ml.features import create_basic_features


DATA_PATH = Path("data/raw/train_transaction.csv")


def main() -> None:
    columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "isFraud",
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
        nrows=1000,
    )

    result = create_basic_features(df)

    print("=" * 60)
    print("BASIC FEATURE ENGINEERING TEST")
    print("=" * 60)

    print("\nOriginal shape:")
    print(df.shape)

    print("\nNew shape:")
    print(result.shape)

    print("\nNew features:")
    print(
        result[
            [
                "TransactionID",
                "TransactionAmt",
                "amount_log",
                "transaction_hour",
                "transaction_day",
                "isFraud",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nFeature ranges:")

    print(
        "amount_log:",
        result["amount_log"].min(),
        "→",
        result["amount_log"].max(),
    )

    print(
        "transaction_hour:",
        result["transaction_hour"].min(),
        "→",
        result["transaction_hour"].max(),
    )

    print(
        "transaction_day:",
        result["transaction_day"].min(),
        "→",
        result["transaction_day"].max(),
    )

    print("\nReal dataset feature test completed successfully.")


if __name__ == "__main__":
    main()