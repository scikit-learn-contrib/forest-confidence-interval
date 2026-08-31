import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer, make_classification, make_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import forestci as fci

def get_datasets():
    datasets = {}
    
    # 1. Auto MPG
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mpg_path = os.path.join(base_dir, 'examples', 'data', 'auto_mpg.csv')
    df = pd.read_csv(mpg_path)
    df = df.replace('?', np.nan).dropna()
    y_mpg = df['mpg'].values
    X_mpg = df.drop(['mpg'], axis=1).values.astype(float)
    datasets['Auto MPG'] = (X_mpg, y_mpg, 'regression')
    
    # 2. California Housing
    california = fetch_california_housing()
    datasets['California'] = (california.data, california.target, 'regression')
    
    # 3. Diabetes
    diabetes = load_diabetes()
    datasets['Diabetes'] = (diabetes.data, diabetes.target, 'regression')
    
    # 4. Breast Cancer
    cancer = load_breast_cancer()
    datasets['Breast Cancer'] = (cancer.data, cancer.target, 'classification')
    
    # 5. Synthetic Hard
    X_sh, y_sh = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42)
    datasets['Synth Hard'] = (X_sh, y_sh, 'classification')
    
    # 6. Synthetic Reg
    X_sr, y_sr = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    datasets['Synthetic Reg'] = (X_sr, y_sr, 'regression')
    
    return datasets

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def run_benchmark():
    datasets = get_datasets()
    tree_counts = [50, 100, 200]
    results = []

    for name, (X, y, task) in datasets.items():
        print(f"Running dataset: {name}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if task == 'regression':
            model_class = RandomForestRegressor
        else:
            model_class = RandomForestClassifier
            
        # Reference model (2000 trees)
        ref_model = model_class(n_estimators=2000, random_state=42, n_jobs=-1)
        ref_model.fit(X_train, y_train)
        
        # Calculate reference variance using inbag
        inbag_ref = fci.calc_inbag(X_train.shape[0], ref_model)
        ref_var = fci.random_forest_error(ref_model, X_train.shape, X_test, inbag=inbag_ref, calibrate=False)
        ref_var = np.maximum(ref_var, 0) # reference is clipped to positive
        
        for n_trees in tree_counts:
            print(f"  Trees: {n_trees}")
            model = model_class(n_estimators=n_trees, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            inbag = fci.calc_inbag(X_train.shape[0], model)
            
            var_uncal = fci.random_forest_error(model, X_train.shape, X_test, inbag=inbag, calibrate=False)
            var_cal = fci.random_forest_error(model, X_train.shape, X_test, inbag=inbag, calibrate=True)
            
            neg_rate = np.mean(var_uncal < 0) * 100
            
            var_uncal_clipped = np.maximum(var_uncal, 0)
            rmse_uncal = rmse(ref_var, var_uncal_clipped)
            rmse_cal = rmse(ref_var, var_cal)
            
            rel_imp = ((rmse_uncal - rmse_cal) / rmse_uncal) * 100
            
            results.append({
                'Dataset': name,
                'Trees': n_trees,
                'Neg Rate (Uncal)': f"{neg_rate:.1f}%",
                'Var RMSE (Uncal)': f"{rmse_uncal:.3f}" if rmse_uncal < 10 else f"{rmse_uncal:.1f}",
                'Var RMSE (Cal)': f"{rmse_cal:.3f}" if rmse_cal < 10 else f"{rmse_cal:.1f}",
                'Relative Improvement': f"{rel_imp:.1f}%"
            })
            
    print("\nBenchmark Results:")
    print("-" * 110)
    print(f"{'Dataset':<20} | {'Trees':<6} | {'Neg Rate (Uncal)':<16} | {'Var RMSE (Uncal)':<16} | {'Var RMSE (Cal)':<16} | {'Relative Improvement'}")
    print("-" * 110)
    for r in results:
        dataset_name = r['Dataset'] if r['Trees'] == 50 else ""
        print(f"{dataset_name:<20} | {r['Trees']:<6} | {r['Neg Rate (Uncal)']:<16} | {r['Var RMSE (Uncal)']:<16} | {r['Var RMSE (Cal)']:<16} | {r['Relative Improvement']}")
    print("-" * 110)

    # Also output RST format for easy copy-paste
    print("\nRST Table format:")
    for r in results:
        dataset_name = r['Dataset'] if r['Trees'] == 50 else ""
        print(f"   * - {dataset_name}")
        print(f"     - {r['Trees']}")
        print(f"     - {r['Neg Rate (Uncal)']}")
        print(f"     - {r['Var RMSE (Uncal)']}")
        print(f"     - {r['Var RMSE (Cal)']}")
        print(f"     - {r['Relative Improvement']}")

if __name__ == '__main__':
    run_benchmark()
