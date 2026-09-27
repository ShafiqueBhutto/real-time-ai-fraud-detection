import pandas as pd

from ml.split import chronological_split


DATA_PATH = "data/raw/train_transaction.csv"


def main():
    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "isFraud",
        ],
    )

    train, validation, test = chronological_split(df)

    print("Full dataset:", df.shape)
    print("Train:", train.shape)
    print("Validation:", validation.shape)
    print("Test:", test.shape)

    print("\nTransactionDT ranges:")

    print(
        f"Train:      {train['TransactionDT'].min()} "
        f"-> {train['TransactionDT'].max()}"
    )

    print(
        f"Validation: {validation['TransactionDT'].min()} "
        f"-> {validation['TransactionDT'].max()}"
    )

    print(
        f"Test:       {test['TransactionDT'].min()} "
        f"-> {test['TransactionDT'].max()}"
    )

    print("\nFraud rates:")

    print(
        f"Train:      {train['isFraud'].mean() * 100:.3f}%"
    )

    print(
        f"Validation: {validation['isFraud'].mean() * 100:.3f}%"
    )

    print(
        f"Test:       {test['isFraud'].mean() * 100:.3f}%"
    )

    print("\nBoundary checks:")

    print(
        "Train < Validation:",
        train["TransactionDT"].max()
        < validation["TransactionDT"].min(),
    )

    print(
        "Validation < Test:",
        validation["TransactionDT"].max()
        < test["TransactionDT"].min(),
    )


if __name__ == "__main__":
    main()