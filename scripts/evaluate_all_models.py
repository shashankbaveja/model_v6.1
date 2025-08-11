import pandas as pd
import joblib
import os
import sys
import glob
from datetime import datetime

# Add the parent directory to the Python path to allow for package-like imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluate_model import evaluate_model

def main():
    """
    Scans the models/ directory, evaluates every .joblib model found against
    a predefined set of thresholds, and saves a consolidated report.
    """
    print("--- Starting Full Model Evaluation Sweep ---")
    
    models_dir = 'models'
    processed_dir = 'data/processed'
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Scan for all model files
    model_paths = glob.glob(os.path.join(models_dir, '*.joblib'))
    if not model_paths:
        print("No models found in the 'models/' directory. Exiting.")
        return
        
    print(f"Found {len(model_paths)} models to evaluate.")

    # 2. Define thresholds to test
    thresholds_to_test = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    print(f"Evaluating at thresholds: {thresholds_to_test}")

    # --- Load All Test Data into memory to avoid repeated reads ---
    print("Pre-loading all necessary test datasets...")
    # Get unique strategies from model names
    strategies = set([os.path.basename(p).split('_')[0] for p in model_paths])
    test_data = {}
    for s in strategies:
        try:
            path = os.path.join(processed_dir, f'test_{s}_with_patterns_features.parquet')
            test_data[s] = pd.read_parquet(path)
            print(f"  - Loaded data for strategy '{s}'")
        except FileNotFoundError:
            print(f"Warning: Test data for strategy '{s}' not found. Models for this strategy will be skipped.")

    # 3. Loop, Evaluate, and Collect Results
    results = []
    for model_path in model_paths:
        try:
            model_filename = os.path.basename(model_path)
            print(f"\n--- Evaluating model: {model_filename} ---")

            # a. Parse filename to get model characteristics
            parts = model_filename.replace('_model.joblib', '').split('_')
            strategy_name = parts[0]
            direction = parts[1]

            # b. Load correct data from cache
            if strategy_name not in test_data:
                print(f"Skipping model, test data for strategy '{strategy_name}' was not found.")
                continue
            
            test_df = test_data[strategy_name].copy()

            # c. Prepare X_test and y_test
            target_col = f'target_{direction}'
            if target_col not in test_df.columns:
                print(f"Skipping model, target column '{target_col}' not found in data for strategy '{strategy_name}'.")
                continue
            
            y_test = test_df.pop(target_col)
            X_test = test_df.drop(columns=['target_up', 'target_down', 'instrument_token', 'timestamp'], errors='ignore')

            # d. Evaluate across all defined thresholds
            for threshold in thresholds_to_test:
                metrics = evaluate_model(model_path, X_test, y_test, threshold=threshold)
                
                result_row = {
                    'Model_Filename': model_filename,
                    'Strategy': strategy_name,
                    'Direction': direction,
                    'Threshold': threshold,
                    **metrics
                }
                results.append(result_row)
                print(f"  - Threshold {threshold:.2f}: Precision={metrics['Precision']:.2%}, Recall={metrics['Recall']:.2%}, F1={metrics['F1-Score']:.2%}")

        except Exception as e:
            print(f"!!! ERROR evaluating model {model_path}: {e} !!!")

    # 4. Consolidate and Save Report
    if not results:
        print("\nNo models were successfully evaluated. No report generated.")
        return
        
    results_df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f'full_model_evaluation_summary_{timestamp}.csv'
    report_path = os.path.join(reports_dir, report_filename)
    
    results_df.to_csv(report_path, index=False)
    
    print("\n--- Full Model Evaluation Summary ---")
    print(results_df.to_string())
    print(f"\nComprehensive report saved to: {report_path}")
    print("\n--- Evaluation Sweep Finished ---")

if __name__ == "__main__":
    main() 