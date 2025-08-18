import os
import pandas as pd
import sys
from utils.telegram import send_telegram_message
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from myKiteLib import system_initialization
# When running as `python src/trade_generator.py`, the `src` directory is on sys.path
# so we import local modules without the `src.` prefix
from utils.backtesting_class import Backtester
from data_pipeline import load_config


def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    config = load_config()

    # Paths (with safe defaults if not present)
    paths_cfg = config.get('paths', {})
    processed_dir = paths_cfg.get('processed_dir', 'data/processed')
    signals_dir = paths_cfg.get('signals_dir', 'data/signals')
    output_dir = paths_cfg.get('reports_dir', 'reports')

    raw_data_path = os.path.join(processed_dir, 'test_raw.parquet')
    features_data_path = os.path.join(processed_dir, 'backtest_features_with_vol_v3.parquet')
    unified_signals_path = os.path.join(signals_dir, 'unified_signals_v3.csv')

    token_map_path = paths_cfg.get('token_map_path')
    trading_days_path = paths_cfg.get('trading_days_path')
    token_df = _load_df(token_map_path) if token_map_path else None

    # Backtest config
    tgt_cfg = config.get('target_generation', {})
    bt_cfg = config.get('backtest_v3', {})

    bt = Backtester(
        raw_data_path=raw_data_path,
        unified_signals_path=unified_signals_path,
        features_data_path=features_data_path,
        token_to_symbol_df=token_df,
        trading_days_path=trading_days_path,
        output_dir=output_dir,
        lookahead_candles=tgt_cfg.get('lookahead_candles', 3),
        volatility_tp_multipler=tgt_cfg.get('volatility_tp_multipler', 4.0),
        volatility_sl_multipler=tgt_cfg.get('volatility_sl_multipler', 4.0),
        exit_signal_integration=bt_cfg.get('exit_signal_integration', True),
        allow_reentry_updates=bt_cfg.get('allow_reentry_updates', True),
        python_executable=bt_cfg.get('python_executable', config.get('python_executable')),
        capital_per_trade=bt_cfg.get('capital_per_trade', 5000.0),
    )

    results = bt.run()

    # Console summary
    metrics = results['performance_metrics']
    trade_log = results['trade_log_df']
    new_trades = trade_log[trade_log['exit_reason'] == 'enter_today']
    new_trades['entry_amount'] = new_trades['entry_price']*new_trades['num_shares']
    new_capital_required = new_trades['entry_amount'].sum()

    sys_init = system_initialization()
    margins_json = sys_init.kite.margins()
    equity = margins_json.get('equity', {})
    available = equity.get('available', {}) or {}
    cash_available = available.get('cash')

    print('--- BACKTEST METRICS ---')
    for k, v in metrics.items():
        print(f"{k}: {v}")

    message = f"New Capital Required: {new_capital_required:.2f}"
    send_telegram_message(message)
    message = f"Cash Available: {cash_available:.2f}"
    send_telegram_message(message)


if __name__ == '__main__':
    main()

