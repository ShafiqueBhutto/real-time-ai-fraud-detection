from pathlib import Path

import pandas as pd


TRANSACTION_PATH = Path("data/raw/train_transaction.csv")
IDENTITY_PATH = Path("data/raw/train_identity.csv")


def main() -> None:
    print("=" * 60)
    print("TRANSACTION + IDENTITY JOIN TEST")
    print("=" * 60)

    print("\nLoading datasets...")

    transactions = pd.read_csv(TRANSACTION_PATH)
    identity = pd.read_csv(IDENTITY_PATH)

    print(f"Transaction shape: {transactions.shape}")
    print(f"Identity shape:    {identity.shape}")

    print("\nJoining on TransactionID...")

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    print(f"Merged shape:      {merged.shape}")

    print("\nIdentity availability:")

    print(merged["_merge"].value_counts())

    print("\nIdentity coverage:")
    identity_available = (merged["_merge"] == "both").mean() * 100
    print(f"{identity_available:.2f}%")

    print("\nDuplicate TransactionIDs after join:")
    print(merged["TransactionID"].duplicated().sum())

    print("\nJoin test completed successfully.")


if __name__ == "__main__":
    main()