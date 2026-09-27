import pandas as pd


IDENTITY_PATH = "data/raw/train_identity.csv"


def main():
    df = pd.read_csv(IDENTITY_PATH)

    identity_columns = [
        column
        for column in df.columns
        if column.startswith("id_")
    ]

    profile = pd.DataFrame(
        {
            "column": identity_columns,
            "dtype": [
                df[column].dtype
                for column in identity_columns
            ],
            "missing_count": [
                df[column].isna().sum()
                for column in identity_columns
            ],
            "missing_percent": [
                df[column].isna().mean() * 100
                for column in identity_columns
            ],
            "unique_count": [
                df[column].nunique(dropna=True)
                for column in identity_columns
            ],
        }
    )

    profile = profile.sort_values(
        "missing_percent"
    ).reset_index(drop=True)

    print("Identity dataset shape:", df.shape)
    print("Identity columns:", len(identity_columns))

    print("\nIdentity column profile:")
    print(profile.to_string(index=False))

    print("\nColumns with < 50% missing:")
    print(
        profile[
            profile["missing_percent"] < 50
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()