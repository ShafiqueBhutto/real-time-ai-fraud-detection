import numpy as np
import pandas as pd


SECONDS_PER_DAY = 24 * 60 * 60
SECONDS_PER_HOUR = 60 * 60


def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic transaction and time-based features.

    Expected input columns:
        TransactionDT
        TransactionAmt

    Returns:
        DataFrame containing the original data plus engineered features.
    """

    result = df.copy()

    # Transaction amount transformation.
    # log1p handles zero safely and reduces the effect of extreme values.
    result["amount_log"] = np.log1p(result["TransactionAmt"])

    # TransactionDT represents elapsed time in seconds.
    result["transaction_hour"] = (
        (result["TransactionDT"] // SECONDS_PER_HOUR) % 24
    ).astype("int8")

    # Number of elapsed days from the dataset reference point.
    result["transaction_day"] = (
        result["TransactionDT"] // SECONDS_PER_DAY
    ).astype("int32")

    return result

def create_card_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create historical card-based behavioral features.

    Historical features only use transactions that occurred
    before the current transaction.
    """

    result = df.copy()

    # Keep original row order.
    result["_original_order"] = np.arange(len(result))

    # Historical calculations must follow transaction time.
    result = result.sort_values("TransactionDT").copy()


         # Number of previous transactions by the same card1
    # within the previous 5 minutes.
    result["card1_transactions_last_5min"] = 0

    for card, group in result.groupby("card1", sort=False):
        times = group["TransactionDT"].to_numpy()

        counts = np.searchsorted(
            times,
            times - 300,
            side="left",
        )

        result.loc[group.index, "card1_transactions_last_5min"] = (
            np.arange(len(group)) - counts
        )


            # Number of previous transactions by the same card1
    # within the previous 1 hour.
    result["card1_transactions_last_1h"] = 0

    for card, group in result.groupby("card1", sort=False):
        times = group["TransactionDT"].to_numpy()

        counts = np.searchsorted(
            times,
            times - 3600,
            side="left",
        )

        result.loc[group.index, "card1_transactions_last_1h"] = (
            np.arange(len(group)) - counts
        )

        result["card1_transactions_last_24h"] = 0

    for card, group in result.groupby("card1", sort=False):
        times = group["TransactionDT"].to_numpy()

        start_positions = np.searchsorted(
            times,
            times - (24 * 60 * 60),
            side="left",
        )

        result.loc[group.index, "card1_transactions_last_24h"] = (
            np.arange(len(group)) - start_positions
        )

        result["card1_amount_total_last_24h"] = 0.0

    for card, group in result.groupby("card1", sort=False):
        times = group["TransactionDT"].to_numpy()
        amounts = group["TransactionAmt"].to_numpy()

        start_positions = np.searchsorted(
            times,
            times - (24 * 60 * 60),
            side="left",
        )

        cumulative_amounts = np.concatenate(
            ([0.0], np.cumsum(amounts))
        )

        result.loc[group.index, "card1_amount_total_last_24h"] = (
            cumulative_amounts[np.arange(len(group)) + 1]
            - cumulative_amounts[start_positions]
            - amounts
        )

    # Total amount of previous transactions by the same card1
    # within the previous 1 hour.
    result["card1_amount_total_last_1h"] = 0.0
    
    

    for card, group in result.groupby("card1", sort=False):
        times = group["TransactionDT"].to_numpy()
        amounts = group["TransactionAmt"].to_numpy()

        start_positions = np.searchsorted(
            times,
            times - 3600,
            side="left",
        )

        cumulative_amounts = np.concatenate(
            ([0.0], np.cumsum(amounts))
        )

        result.loc[group.index, "card1_amount_total_last_1h"] = (
            cumulative_amounts[np.arange(len(group)) + 1]
            - cumulative_amounts[start_positions]
            - amounts
        )

        result["card1_amount_total_last_5min"] = 0.0

    for card, group in result.groupby("card1", sort=False):
        times = group["TransactionDT"].to_numpy()
        amounts = group["TransactionAmt"].to_numpy()

        start_positions = np.searchsorted(
            times,
            times - 300,
            side="left",
        )

        cumulative_amounts = np.concatenate(
            ([0.0], np.cumsum(amounts))
        )

        result.loc[group.index, "card1_amount_total_last_5min"] = (
            cumulative_amounts[np.arange(len(group)) + 1]
            - cumulative_amounts[start_positions]
            - amounts
        )

            # Prevent tiny floating-point precision errors.
    result["card1_amount_total_last_5min"] = (
        result["card1_amount_total_last_5min"].clip(lower=0)
    )

    result["card1_amount_total_last_1h"] = (
        result["card1_amount_total_last_1h"].clip(lower=0)
    )

    result["card1_amount_total_last_24h"] = (
        result["card1_amount_total_last_24h"].clip(lower=0)
    )
        

    # Previous transaction count for the same card.
    result["card1_previous_transaction_count"] = (
        result.groupby("card1", sort=False).cumcount()
    )

    # Previous transactions for the same card1 and ProductCD.
    if "ProductCD" in result.columns:
        result["card1_product_previous_count"] = (
            result.groupby(
                ["card1", "ProductCD"],
                sort=False,
            ).cumcount()
        )
    

    # Cumulative amount of previous transactions.
    cumulative_amount = (
        result.groupby("card1", sort=False)["TransactionAmt"]
        .cumsum()
        .sub(result["TransactionAmt"])
    )
    

    # Avoid division by zero for cards with no previous transaction.
    previous_count = result["card1_previous_transaction_count"]

    result["card1_previous_avg_amount"] = np.where(
        previous_count > 0,
        cumulative_amount / previous_count,
        np.nan,
    )

    result["card1_amount_vs_previous_avg"] = np.where(
    result["card1_previous_avg_amount"] > 0,
    result["TransactionAmt"] / result["card1_previous_avg_amount"],
    np.nan,
    )

        # Previous transaction amount standard deviation.
    previous_amount_squared = (
        result["TransactionAmt"] ** 2
    )

    cumulative_squared_amount = (
        result.groupby("card1", sort=False)["TransactionAmt"]
        .apply(lambda x: x.pow(2).cumsum())
    )

    cumulative_squared_amount = (
        cumulative_squared_amount.reset_index(level=0, drop=True)
    )

    previous_squared_sum = (
        cumulative_squared_amount
        - previous_amount_squared
    )

    result["card1_previous_std_amount"] = np.where(
        previous_count > 1,
        np.sqrt(
            np.maximum(
                (
                    previous_squared_sum / (previous_count - 1)
                )
                - (
                    cumulative_amount ** 2
                    / (
                        previous_count * (previous_count - 1)
                    )
                ),
                0,
            )
        ),
        np.nan,
    )

    # How unusual the current amount is compared with
    # the previous transaction behavior.
    result["card1_amount_zscore"] = np.where(
        result["card1_previous_std_amount"] > 0,
        (
            result["TransactionAmt"]
            - result["card1_previous_avg_amount"]
        )
        / result["card1_previous_std_amount"],
        np.nan,
    )

    # Restore original order.
    result = (
        result.sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )

    return result

def create_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create compact features from IEEE-CIS identity information.
    """
    result = df.copy()

    # Whether identity information is available
    result["identity_available"] = (
        result["DeviceType"].notna()
        | result["DeviceInfo"].notna()
    ).astype("int8")

    # Device type
    if "DeviceType" in result.columns:
        result["device_type"] = (
            result["DeviceType"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
        )

    # DeviceInfo characteristics
    if "DeviceInfo" in result.columns:
        device_info = result["DeviceInfo"].fillna("").astype(str)

        result["device_info_missing"] = (
            device_info == ""
        ).astype("int8")

        result["device_info_length"] = (
            device_info.str.len()
        ).astype("int16")

    # Selected IEEE-CIS identity features
    selected_identity_columns = [
        "id_01",
        "id_02",
        "id_05",
        "id_06",
        "id_11",
        "id_12",
        "id_13",
        "id_14",
        "id_15",
        "id_16",
        "id_17",
        "id_19",
        "id_20",
        "id_28",
        "id_29",
        "id_31",
        "id_35",
        "id_36",
        "id_37",
        "id_38",
    ]

    for column in selected_identity_columns:
        if column in result.columns:
            result[f"identity_{column}"] = result[column]

    return result