# ============================================================
# Description: Downloads 5 years of S&P/ASX 200 Energy Index
#              (XEJ) data and splits into 3yr train, 1yr test,
#              1yr eval sets saved as CSV and Parquet.
# ============================================================

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def main():
    # ----------------------------------------------------------
    # 1. Configure
    # ----------------------------------------------------------
    TICKER = "^AXEJ"  # ASX 200 Energy Index on Yahoo Finance
    OUTPUT_DIR = Path("data/static/XEJ")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 5 years back from today
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)

    print(f"Downloading {TICKER} from {start_date.date()} to {end_date.date()}...")

    # ----------------------------------------------------------
    # 2. Download
    # ----------------------------------------------------------
    df = yf.download(TICKER, start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        print("ERROR: No data returned. Check the ticker or your connection.")
        return

    # Flatten multi-level columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep the index as a column for clarity
    df.index.name = "Date"
    df = df.reset_index()

    print(
        f"Downloaded {len(df)} trading days "
        f"({df['Date'].min().date()} to {df['Date'].max().date()})"
    )

    # ----------------------------------------------------------
    # 3. Split: 3yr train | 1yr test | 1yr eval (chronological)
    # ----------------------------------------------------------
    earliest = df["Date"].min()

    train_end = earliest + pd.DateOffset(years=3)
    test_end = train_end + pd.DateOffset(years=1)
    # eval runs to the end of whatever data is available

    train = df[df["Date"] < train_end].copy()
    test = df[(df["Date"] >= train_end) & (df["Date"] < test_end)].copy()
    eval_ = df[df["Date"] >= test_end].copy()

    print("\nSplit summary:")
    print(
        f"  Train : {len(train):>4} days  "
        f"({train['Date'].min().date()} to {train['Date'].max().date()})"
    )
    print(
        f"  Test  : {len(test):>4} days  "
        f"({test['Date'].min().date()} to {test['Date'].max().date()})"
    )
    print(
        f"  Eval  : {len(eval_):>4} days  "
        f"({eval_['Date'].min().date()} to {eval_['Date'].max().date()})"
    )

    # ----------------------------------------------------------
    # 4. Save as both CSV and Parquet
    # ----------------------------------------------------------
    for name, split in [("train", train), ("test", test), ("eval", eval_)]:
        csv_path = OUTPUT_DIR / f"xej_{name}.csv"
        pq_path = OUTPUT_DIR / f"xej_{name}.parquet"

        split.to_csv(csv_path, index=False)
        split.to_parquet(pq_path, index=False)

        print(f"  Saved {csv_path} and {pq_path}")

    # Also save the full unsplit dataset
    df.to_csv(OUTPUT_DIR / "xej_full.csv", index=False)
    df.to_parquet(OUTPUT_DIR / "xej_full.parquet", index=False)
    print(f"  Saved full dataset to {OUTPUT_DIR}/xej_full.*")

    print("\nDone!")


if __name__ == "__main__":
    main()
