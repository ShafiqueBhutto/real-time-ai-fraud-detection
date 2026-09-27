import pandas as pd


TRANSACTION_PATH = "data/raw/train_transaction.csv"
IDENTITY_PATH = "data/raw/train_identity.csv"


def main():
    transactions = pd.read_csv(
        TRANSACTION_PATH,
        usecols=["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"],
    )

    identity = pd.read_csv(
        IDENTITY_PATH,
        usecols=[
            "TransactionID",
            "DeviceType",
            "DeviceInfo",
        ],
    )

    print("Transaction shape:", transactions.shape)
    print("Identity shape:", identity.shape)

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    print("\nMerged shape:", merged.shape)

    print("\nJoin result:")
    print(merged["_merge"].value_counts())

    print("\nIdentity coverage:")
    print(f"{merged['DeviceType'].notna().mean() * 100:.2f}%")

    print("\nSample:")
    print(
        merged[
            [
                "TransactionID",
                "TransactionAmt",
                "isFraud",
                "DeviceType",
                "DeviceInfo",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()