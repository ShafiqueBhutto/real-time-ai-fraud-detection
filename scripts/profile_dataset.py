from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/train_transaction.csv")


def main() -> None:
    print("=" * 60)
    print("FRAUD DETECTION DATASET PROFILE")
    print("=" * 60)

    print(f"\nLoading: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:")
    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    print("\nTarget distribution:")
    target_counts = df["isFraud"].value_counts()

    print(target_counts)

    print("\nTarget percentage:")
    target_percentage = (
        df["isFraud"]
        .value_counts(normalize=True)
        .mul(100)
        .round(3)
    )

    print(target_percentage)

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print(f"Columns with missing values: {len(missing)}")
    print(missing.head(20))

    print("\nDuplicate TransactionIDs:")
    print(df["TransactionID"].duplicated().sum())

    print("\nData types:")
    print(df.dtypes.value_counts())

    print("\nTransaction amount statistics:")
    print(df["TransactionAmt"].describe())

    print("\nProductCD distribution:")
    print(df["ProductCD"].value_counts())

    print("\nDataset profiling completed successfully.")


if __name__ == "__main__":
    main()