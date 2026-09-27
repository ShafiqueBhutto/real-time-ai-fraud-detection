import pandas as pd

from ml.features import (
    create_basic_features,
    create_card_history_features,
    create_identity_features,
)

def test_create_basic_features():
    df = pd.DataFrame(
        {
            "TransactionDT": [3600, 86400, 90000],
            "TransactionAmt": [100.0, 250.0, 1000.0],
        }
    )

    result = create_basic_features(df)

    assert "amount_log" in result.columns
    assert "transaction_hour" in result.columns
    assert "transaction_day" in result.columns

    assert result["transaction_hour"].tolist() == [1, 0, 1]
    assert result["transaction_day"].tolist() == [0, 1, 1]

    assert (result["amount_log"] > 0).all()






def test_card_history_features_use_only_previous_transactions():
    df = pd.DataFrame(
        {
            "TransactionDT": [300, 100, 200, 400],
            "card1": [111, 111, 111, 222],
            "TransactionAmt": [300.0, 100.0, 200.0, 400.0],
        }
    )

    result = create_card_history_features(df)

    # Original row order must be preserved.
    assert result["TransactionDT"].tolist() == [300, 100, 200, 400]

    # Chronological order for card 111:
    # DT=100 -> previous count 0
    # DT=200 -> previous count 1
    # DT=300 -> previous count 2
    #
    # card 222 has no previous transaction -> 0
    assert result["card1_previous_transaction_count"].tolist() == [
        2,
        0,
        1,
        0,
    ]



def test_card_history_average_uses_only_previous_amounts():
    df = pd.DataFrame(
        {
            "TransactionDT": [100, 200, 300, 400],
            "card1": [111, 111, 111, 222],
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0],
        }
    )

    result = create_card_history_features(df)

    card_111 = result[result["card1"] == 111]

    assert pd.isna(
        card_111.iloc[0]["card1_previous_avg_amount"]
    )

    assert card_111.iloc[1]["card1_previous_avg_amount"] == 100.0

    assert card_111.iloc[2]["card1_previous_avg_amount"] == 150.0

    card_222 = result[result["card1"] == 222]

    assert pd.isna(
        card_222.iloc[0]["card1_previous_avg_amount"]
    )


def test_card_amount_deviation_uses_previous_average():
    df = pd.DataFrame(
        {
            "TransactionDT": [100, 200, 300, 400],
            "card1": [111, 111, 111, 222],
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0],
        }
    )

    result = create_card_history_features(df)

    card_111 = result[result["card1"] == 111]

    assert pd.isna(
        card_111.iloc[0]["card1_amount_vs_previous_avg"]
    )

    assert card_111.iloc[1]["card1_amount_vs_previous_avg"] == 2.0

    assert card_111.iloc[2]["card1_amount_vs_previous_avg"] == 2.0

    card_222 = result[result["card1"] == 222]

    assert pd.isna(
        card_222.iloc[0]["card1_amount_vs_previous_avg"]
    )

def test_card_transactions_last_5min_uses_only_previous_transactions():
    df = pd.DataFrame(
        {
            "TransactionDT": [100, 200, 250, 700],
            "card1": [111, 111, 111, 111],
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0],
        }
    )

    result = create_card_history_features(df)

    # DT=100 -> no previous transaction
    assert result.iloc[0]["card1_transactions_last_5min"] == 0

    # DT=200 -> DT=100 is within previous 5 minutes
    assert result.iloc[1]["card1_transactions_last_5min"] == 1

    # DT=250 -> DT=100 and DT=200 are within previous 5 minutes
    assert result.iloc[2]["card1_transactions_last_5min"] == 2

    # DT=700 -> previous transactions are more than 300 seconds away
    assert result.iloc[3]["card1_transactions_last_5min"] == 0


def test_card_transactions_last_1h_uses_only_previous_transactions():
    df = pd.DataFrame(
        {
            "TransactionDT": [100, 200, 300, 3700, 4000],
            "card1": [111, 111, 111, 111, 111],
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )

    result = create_card_history_features(df)

    # DT=100 -> no previous transaction
    assert result.iloc[0]["card1_transactions_last_1h"] == 0

    # DT=200 -> DT=100 is within 1 hour
    assert result.iloc[1]["card1_transactions_last_1h"] == 1

    # DT=300 -> DT=100 and DT=200 are within 1 hour
    assert result.iloc[2]["card1_transactions_last_1h"] == 2

    # DT=3700 -> DT=100 is exactly 3600 seconds earlier,
    # so it is included in the previous 1-hour window.
    assert result.iloc[3]["card1_transactions_last_1h"] == 3

   # DT=4000 -> only DT=3700 is within the previous 1-hour window.
    assert result.iloc[4]["card1_transactions_last_1h"] == 1

def test_card_amount_total_last_1h_uses_only_previous_transactions():
    df = pd.DataFrame(
        {
            "TransactionDT": [100, 200, 300, 4000],
            "card1": [111, 111, 111, 111],
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0],
        }
    )

    result = create_card_history_features(df)

    # First transaction -> no history.
    assert result.iloc[0]["card1_amount_total_last_1h"] == 0.0

    # DT=200 -> previous total = 100.
    assert result.iloc[1]["card1_amount_total_last_1h"] == 100.0

    # DT=300 -> previous total = 100 + 200.
    assert result.iloc[2]["card1_amount_total_last_1h"] == 300.0

    # DT=4000 -> DT=100 is outside the 1-hour window.
    # DT=200 and DT=300 are also outside because 4000-3600=400.
    assert result.iloc[3]["card1_amount_total_last_1h"] == 0.0


def test_card_amount_total_last_5min_uses_only_previous_transactions():
    df = pd.DataFrame(
        {
            "TransactionDT": [100, 200, 250, 700],
            "card1": [111, 111, 111, 111],
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0],
        }
    )

    result = create_card_history_features(df)

    expected = [0.0, 100.0, 300.0, 0.0]

    assert result["card1_amount_total_last_5min"].tolist() == expected

def test_card_transactions_last_24h_uses_only_previous_transactions():
    df = pd.DataFrame(
        {
            "TransactionDT": [
                100,
                200,
                300,
                86400,
                86500,
            ],
            "card1": [111, 111, 111, 111, 111],
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )

    result = create_card_history_features(df)

    expected = [0, 1, 2, 3, 4]

    assert result["card1_transactions_last_24h"].tolist() == expected

def test_card_amount_total_last_24h_uses_only_previous_transactions():
    df = pd.DataFrame(
        {
            "TransactionDT": [
                100,
                200,
                300,
                86400,
                86500,
            ],
            "card1": [111, 111, 111, 111, 111],
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )

    result = create_card_history_features(df)

    expected = [0.0, 100.0, 300.0, 600.0, 1000.0]

    assert result["card1_amount_total_last_24h"].tolist() == expected


def test_card_amount_zscore_uses_only_previous_transactions():
    df = pd.DataFrame(
        {
            "TransactionDT": [100, 200, 300, 400],
            "card1": [111, 111, 111, 111],
            "TransactionAmt": [100.0, 110.0, 105.0, 500.0],
        }
    )

    result = create_card_history_features(df)

    assert pd.isna(result.loc[0, "card1_amount_zscore"])
    assert pd.isna(result.loc[1, "card1_amount_zscore"])

    # Third transaction has two previous amounts.
    assert result.loc[2, "card1_amount_zscore"] == 0.0

    # Fourth transaction is far from previous behavior.
    assert result.loc[3, "card1_amount_zscore"] > 10

def test_card_product_previous_count_uses_only_previous_transactions():
    df = pd.DataFrame(
        {
            "TransactionDT": [100, 200, 300, 400, 500],
            "card1": [111, 111, 111, 111, 222],
            "ProductCD": ["W", "W", "C", "W", "W"],
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )

    result = create_card_history_features(df)

    expected = [0, 1, 0, 2, 0]

    assert (
        result["card1_product_previous_count"].tolist()
        == expected
    )

def test_identity_features():
    df = pd.DataFrame(
        {
            "DeviceType": ["mobile", None, "desktop"],
            "DeviceInfo": ["iPhone", None, "Windows PC"],
        }
    )

    result = create_identity_features(df)

    assert result["identity_available"].tolist() == [1, 0, 1]
    assert result["device_info_missing"].tolist() == [0, 1, 0]
    assert result["device_info_length"].tolist() == [6, 0, 10]