import os
import sys
import csv
from copy import deepcopy
from datetime import datetime

import pandas as pd
from decimal import Decimal, getcontext

# Ensure we can import from project src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.trade_generator_v2 import (
    load_config,
    prepare_data_v2,
    run_simulation_v2,
    calculate_performance_metrics,
    get_next_candle_map,
)


def vals_by_step(start: float, end: float, step: float):
    """Generate values from start to end inclusive with a given step, using Decimal for precision."""
    getcontext().prec = 10
    s = Decimal(str(start))
    e = Decimal(str(end))
    st = Decimal(str(step))
    vals = []
    v = s
    # Determine rounding decimals from step
    step_str = str(step)
    decimals = len(step_str.split('.')[-1]) if '.' in step_str else 0
    while v <= e + Decimal('1e-12'):
        vals.append(round(float(v), decimals))
        v += st
    return vals


def print_progress(current: int, total: int, bar_len: int = 40):
    filled_len = int(round(bar_len * current / float(total)))
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    pct = round(100.0 * current / float(total), 1)
    sys.stdout.write(f"\r[{bar}] {pct}% ({current}/{total})")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def main():
    # Load config and prepare data ONCE (this also generates unified_signals_v2.csv)
    config = load_config()
    master_df = prepare_data_v2(config)
    # Precompute maps used by every simulation
    next_candle_map = get_next_candle_map(master_df)
    last_rows_map = {instrument: group.iloc[-1] for instrument, group in master_df.groupby('instrument_token')}

    # Define grid per requested granularities
    prob_1_vals = vals_by_step(0.75, 0.80, 0.01)
    prob_2_vals = vals_by_step(0.75, 0.80, 0.01)
    entry_filter_vals = vals_by_step(0.20, 0.40, 0.05)
    exit_trigger_vals = 0.35
    sl_multiplier_fixed = 4.0

    total_combos = (
        len(prob_1_vals) * len(prob_2_vals) * len(entry_filter_vals)
    )

    print(f"Total combinations: {total_combos}")

    # Prepare output
    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(reports_dir, f'grid_search_v2_{timestamp}.csv')

    # CSV header
    fieldnames = [
        'prob_threshold_1', 'prob_threshold_2',
        'threshold_for_entry_filter', 'threshold_for_exit_signal',
        'volatility_sl_multipler',
        'Total_Trades', 'Win_Rate_%', 'Total_PnL', 'Profit_Factor',
        'Avg_Trade_PnL', 'Avg_Return_%', 'Max_Drawdown_PnL', 'Max_Drawdown_%',
        'Best_Trade_PnL', 'Worst_Trade_PnL', 'Target_Hit_Count', 'Stop_Loss_Count',
        'Holding_Period_Count', 'Exit_Signal_Count', 'Pending_Entry_Count',
        'Pending_Exit_Count', 'Active_On_Close_Count', 'Avg_Holding_Period_Days'
    ]

    processed = 0
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate grid
        for p1 in prob_1_vals:
            for p2 in prob_2_vals:
                for entry_f in entry_filter_vals:
                        # Clone config and set thresholds
                        cfg = deepcopy(config)
                        bt2 = cfg.setdefault('backtest_v2', {})
                        bt2.setdefault('entry_models', [{}, {}])
                        bt2.setdefault('exit_model', {})
                        bt2['entry_models'][0]['prob_threshold'] = p1
                        bt2['entry_models'][1]['prob_threshold'] = p2
                        bt2['exit_model']['threshold_for_entry_filter'] = entry_f
                        bt2['exit_model']['threshold_for_exit_signal'] = exit_trigger_vals
                        # Also vary SL volatility multiplier
                        tgt = cfg.setdefault('target_generation', {})
                        tgt['volatility_sl_multipler'] = sl_multiplier_fixed

                        # Run simulation and metrics
                        trade_log, _ = run_simulation_v2(
                            master_df,
                            cfg,
                            precomputed_next_candle_map=next_candle_map,
                            precomputed_last_rows_map=last_rows_map,
                            produce_debug=False,
                        )
                        metrics = calculate_performance_metrics(pd.DataFrame(trade_log), initial_capital=bt2.get('capital_per_trade', 10000))

                        # Write row
                        row = {
                            'prob_threshold_1': p1,
                            'prob_threshold_2': p2,
                            'threshold_for_entry_filter': entry_f,
                            'threshold_for_exit_signal': exit_trigger_vals,
                            'volatility_sl_multipler': sl_multiplier_fixed,
                        }
                        row.update(metrics)
                        writer.writerow(row)

                        processed += 1
                        print_progress(processed, total_combos)

    print(f"Grid search complete. Results saved to: {out_path}")


if __name__ == '__main__':
    main()

