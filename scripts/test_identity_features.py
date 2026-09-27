import pandas as pd

from ml.features import create_identity_features


TRANSACTION_PATH = "data/raw/train_transaction.csv"
IDENTITY_PATH = "data/raw/train_identity.csv"


def main():
    transactions = pd.read_csv(
        TRANSACTION_PATH,
        usecols=[
            "TransactionID",
            "TransactionAmt",
        ],
        nrows=5000,
    )

    identity = pd.read_csv(IDENTITY_PATH)

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    result = create_identity_features(merged)

    selected_features = [
        column
        for column in result.columns
        if column.startswith("identity_id_")
    ]

    print("Merged shape:", merged.shape)
    print("Feature shape:", result.shape)

    print("\nSelected identity features:")
    print(selected_features)

    print("\nMissing percentage:")
    print(
        result[selected_features]
        .isna()
        .mean()
        .mul(100)
        .sort_values()
        .to_string()
    )

    print("\nSample:")
    print(
        result[
            [
                "TransactionID",
                "identity_available",
                "device_type",
                "identity_id_01",
                "identity_id_02",
                "identity_id_12",
                "identity_id_31",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()