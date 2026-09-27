import pandas as pd

from ml.dataset import prepare_features_with_history
from ml.split import chronological_split


TRANSACTION_PATH = "data/raw/train_transaction.csv"


def main():
    print("Loading IEEE-CIS data...")

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

    print("Original:", df.shape)

    # Step 1: historical features
    prepared = prepare_features_with_history(df)

    print("After feature engineering:", prepared.shape)

    # Step 2: chronological split
    train, validation, test = chronological_split(prepared)

    print("\nSplit sizes:")
    print("Train:", train.shape)
    print("Validation:", validation.shape)
    print("Test:", test.shape)

    print("\nTime boundaries:")
    print(
        "Train:",
        train["TransactionDT"].min(),
        "->",
        train["TransactionDT"].max(),
    )
    print(
        "Validation:",
        validation["TransactionDT"].min(),
        "->",
        validation["TransactionDT"].max(),
    )
    print(
        "Test:",
        test["TransactionDT"].min(),
        "->",
        test["TransactionDT"].max(),
    )

    # Verify chronological order
    assert train["TransactionDT"].max() <= validation["TransactionDT"].min()
    assert validation["TransactionDT"].max() <= test["TransactionDT"].min()

    # Verify all rows preserved
    assert len(train) + len(validation) + len(test) == len(prepared)

    # Verify historical feature exists
    assert "card1_previous_transaction_count" in prepared.columns

    # First transaction has no history
    assert (
        prepared.iloc[0]["card1_previous_transaction_count"]
        == 0
    )

    print("\nChronological feature pipeline test PASSED.")


if __name__ == "__main__":
    main()