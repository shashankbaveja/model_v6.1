import itertools
import subprocess
import sys
import os
from pathlib import Path
import shutil

# --- Configuration ---
CONFIG_PATH = 'config/parameters.yml'
PIPELINE_SCRIPT_PATH = 'src/train_model.py'
PYTHON_EXECUTABLE = '/opt/anaconda3/envs/KiteConnect/bin/python'

# --- Parameter Ranges ---
LOOKAHEAD_CANDLES = [3, 5, 7, 10, 15]
VOLATILITY_TP_MULTIPLIER = [4.0, 5.0, 7.0, 8.0, 10.0]
VOLATILITY_SL_MULTIPLIER = [2.0]

def update_parameters_yml(lookahead, tp_multiplier, sl_multiplier):
    """
    Updates the parameters.yml file with the given model configuration.
    This function is inspired by src/auto_update_date.py for safe file handling.
    """
    file_path = Path(CONFIG_PATH)
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')

    try:
        # Create a backup of the original config file
        shutil.copy2(file_path, backup_path)

        # Read all lines from the configuration file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Update the specific lines for each parameter (using 0-based indexing)
        # Line 17 -> index 16
        lines[16] = f"  lookahead_candles: {lookahead}\n"
        # Line 18 -> index 17
        lines[17] = f"  volatility_tp_multipler: {tp_multiplier}\n"
        # Line 19 -> index 18
        lines[18] = f"  volatility_sl_multipler: {sl_multiplier}\n"

        # Write the updated lines back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

        # Remove the backup on a successful write
        backup_path.unlink()
        return True

    except Exception as e:
        print(f"  !!! Error updating {CONFIG_PATH}: {e} !!!")
        # Restore the original file from backup if an error occurs
        if backup_path.exists():
            shutil.move(backup_path, file_path)
            print("  Restored original config from backup.")
        return False

def run_pipeline():
    """Executes the main training pipeline script and streams its output."""
    try:
        process = subprocess.Popen(
            [PYTHON_EXECUTABLE, '-u', PIPELINE_SCRIPT_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output to the console in real-time
        for line in process.stdout:
            sys.stdout.write(line)

        process.wait() # Wait for the subprocess to finish

        if process.returncode != 0:
            print(f"!!! Pipeline execution failed with return code {process.returncode} !!!")
            return False
            
        return True

    except FileNotFoundError:
        print(f"!!! Error: Could not find the script at {PIPELINE_SCRIPT_PATH} or the python executable at {PYTHON_EXECUTABLE} !!!")
        return False
    except Exception as e:
        print(f"!!! An unexpected error occurred: {e} !!!")
        return False

def main():
    """
    Main function to iterate through all configurations and run the training pipeline.
    """
    print("====== Starting Multi-Model Training Script ======")
    
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        LOOKAHEAD_CANDLES,
        VOLATILITY_TP_MULTIPLIER,
        VOLATILITY_SL_MULTIPLIER
    ))
    
    original_total = len(param_combinations)
    param_combinations = [
        (l, tp, sl)
        for (l, tp, sl) in param_combinations
        if sl <= 0.6 * tp
    ]

    param_combinations.remove((3, 4.0, 2.0))
    param_combinations.remove((3, 5.0, 2.0))
    param_combinations.remove((3, 7.0, 2.0))
    param_combinations.remove((3, 8.0, 2.0))
    param_combinations.remove((3, 10.0, 2.0))
    param_combinations.remove((5, 4.0, 2.0))
    param_combinations.remove((5, 5.0, 2.0))
    param_combinations.remove((5, 7.0, 2.0))
    param_combinations.remove((5, 8.0, 2.0))
    param_combinations.remove((5, 10.0, 2.0))
    param_combinations.remove((7, 4.0, 2.0))
    param_combinations.remove((7, 5.0, 2.0))
    param_combinations.remove((7, 7.0, 2.0))
    param_combinations.remove((7, 8.0, 2.0))
    param_combinations.remove((7, 10.0, 2.0))
    param_combinations.remove((10, 4.0, 2.0))
    param_combinations.remove((10, 5.0, 2.0))
    param_combinations.remove((10, 7.0, 2.0))
    
   
    # (10, 8.0, 2.0), 
    # (10, 10.0, 2.0), 
    # (15, 4.0, 2.0), 
    # (15, 5.0, 2.0), 
    # (15, 7.0, 2.0), 
    # (15, 8.0, 2.0), 
    # (15, 10.0, 2.0)
    total_runs = len(param_combinations)
    print(f"Filtered parameter combinations: {total_runs} (from {original_total})")



    total_runs = len(param_combinations)
    print(f"Found {total_runs} parameter combinations to train.")

    for i, (lookahead, tp, sl) in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"--- Running Configuration {i+1} of {total_runs} ---")
        print(f"Parameters: lookahead={lookahead}, tp_multiplier={tp}, sl_multiplier={sl}")
        print(f"{'='*60}\n")

        # Step 1: Update the configuration file
        print(f"Updating {CONFIG_PATH}...")
        if not update_parameters_yml(lookahead, tp, sl):
            print("Skipping pipeline run due to config update failure.")
            continue
        print("Configuration updated successfully.")

        # Step 2: Run the main pipeline
        print(f"Executing {PIPELINE_SCRIPT_PATH}...")
        if run_pipeline():
            print(f"\n--- Successfully completed run {i+1} of {total_runs} ---")
        else:
            print(f"\n--- Failed to complete run {i+1} of {total_runs} ---")
            # Decide if you want to stop the entire process on a single failure
            # print("Stopping script due to pipeline failure.")
            # break 

    print("\n====== Multi-Model Training Script Finished ======")

if __name__ == "__main__":
    main()
