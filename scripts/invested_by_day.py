"""
Compute day-by-day invested amount from reports/trade_log.csv.

Definitions per user:
- Invested amount (EOD): cost basis of open positions at end of the day
  plus any blocked capital on BTST (CNC) sells.
- CNC rules:
  - Cannot exit on the same day. If exit_date == entry_date, treat exit_date = entry_date + 1.
  - If a position is sold on entry_date + 1 (BTST), the sell proceeds are
    blocked on that exit day and released the next day. Count the cost basis
    as "blocked capital" on the exit day only.
- No CLI options; reads from reports/trade_log.csv and writes
  reports/invested_by_day.csv. Prints YYYY-MM-DD, invested_amount per day.

Date range: Union of unique entry_date and exit_date present in the file.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List

import pandas as pd


TRADE_LOG_PATH = os.path.join("reports", "trade_log.csv")
OUTPUT_PATH = os.path.join("reports", "invested_by_day.csv")


def load_trade_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Expected columns (minimal set used here)
    required_columns = [
        "entry_date",
        "exit_date",
        "num_shares",
        "entry_price",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in trade log: {missing}")

    # Parse dates as date (no time component)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.date
    if "exit_date" in df.columns:
        df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce").dt.date
    else:
        df["exit_date"] = pd.NaT

    # Coerce numeric fields
    df["num_shares"] = pd.to_numeric(df["num_shares"], errors="coerce").fillna(0).astype(int)
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce").fillna(0.0).astype(float)

    # Compute cost basis
    df["cost_basis"] = df["num_shares"] * df["entry_price"]

    return df


def apply_cnc_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # If exit_date == entry_date, set exit_date = entry_date + 1
    same_day_mask = df["exit_date"].notna() & (df["exit_date"] == df["entry_date"])
    if same_day_mask.any():
        df.loc[same_day_mask, "exit_date"] = [d + timedelta(days=1) for d in df.loc[same_day_mask, "entry_date"]]

    # Derive holding_days where available
    # NaT for missing exit_date; we will not use holding_days for those rows when computing blocked amount
    holding_days = []
    for entry, exit_ in zip(df["entry_date"], df["exit_date"]):
        if pd.isna(exit_):
            holding_days.append(None)
        else:
            holding_days.append((exit_ - entry).days)
    df["holding_days"] = holding_days

    return df


def build_evaluation_dates(df: pd.DataFrame) -> List[datetime.date]:
    entry_dates = set(d for d in df["entry_date"].dropna().tolist())
    exit_dates = set(d for d in df["exit_date"].dropna().tolist())
    all_dates = sorted(entry_dates.union(exit_dates))
    return all_dates


def compute_amounts_for_day(df: pd.DataFrame, day: datetime.date) -> tuple[float, float, float]:
    # Open amount: positions with entry_date <= day and (no exit_date or exit_date > day)
    open_mask = (df["entry_date"] <= day) & (df["exit_date"].isna() | (df["exit_date"] > day))
    open_amount = float(df.loc[open_mask, "cost_basis"].sum())

    # Blocked amount: BTST sells only, i.e., exit on entry+1
    blocked_mask = df["exit_date"].notna() & (df["exit_date"] == day) & (df["holding_days"] == 1)
    blocked_amount = float(df.loc[blocked_mask, "cost_basis"].sum())

    invested_amount = open_amount + blocked_amount
    return open_amount, blocked_amount, invested_amount


def main() -> None:
    if not os.path.exists(TRADE_LOG_PATH):
        raise FileNotFoundError(f"Trade log not found at {TRADE_LOG_PATH}")

    df = load_trade_log(TRADE_LOG_PATH)
    df = apply_cnc_rules(df)

    eval_dates = build_evaluation_dates(df)
    if not eval_dates:
        print("No dates found in trade log. Nothing to compute.")
        return

    results = []
    for day in eval_dates:
        open_amt, blocked_amt, invested_amt = compute_amounts_for_day(df, day)
        results.append({
            "date": day.isoformat(),
            "open_amount": round(open_amt, 2),
            "blocked_amount": round(blocked_amt, 2),
            "invested_amount": round(invested_amt, 2),
        })

    out_df = pd.DataFrame(results)
    # Save CSV
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    # Print to console: date, invested_amount
    for row in results:
        print(f"{row['date']}, {row['invested_amount']}")


if __name__ == "__main__":
    main()

