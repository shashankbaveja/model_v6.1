import os
import glob
import joblib
import pandas as pd
from catboost import CatBoostClassifier


def compute_gain_importance_for_model(model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)

    if not isinstance(model, CatBoostClassifier):
        raise TypeError(f"Model at {model_path} is not a CatBoostClassifier. Got: {type(model)}")

    # CatBoost stores feature names when trained on DataFrame
    feature_names = getattr(model, 'feature_names_', None)
    if feature_names is None or len(feature_names) == 0:
        # Fallback: create generic names based on feature count inferred from importance length
        importances = model.get_feature_importance(type='FeatureImportance')
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    else:
        importances = model.get_feature_importance(type='FeatureImportance')

    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return fi_df


def main():
    models_dir = 'models'
    output_dir = os.path.join('reports', 'feature_importance')
    os.makedirs(output_dir, exist_ok=True)

    model_paths = sorted(glob.glob(os.path.join(models_dir, '*.joblib')))
    if not model_paths:
        print(f"No models found in {models_dir}.")
        return

    for model_path in model_paths:
        try:
            fi_df = compute_gain_importance_for_model(model_path)
        except Exception as e:
            print(f"Skipping {model_path} due to error: {e}")
            continue

        # Save CSV per model
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        csv_path = os.path.join(output_dir, f"{base_name}_feature_importance.csv")
        fi_df.to_csv(csv_path, index=False)
        print(f"Saved feature importance to {csv_path}")

        # Print top 30 features
        top_n = 30
        print(f"\nTop {top_n} features for {base_name}:")
        print(fi_df.head(top_n).to_string(index=False))
        print("\n" + "-" * 80 + "\n")


if __name__ == '__main__':
    main()