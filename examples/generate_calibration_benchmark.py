"""Generate calibration tables and confidence-interval diagnostics at build time."""

import argparse
import time

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer, load_wine, make_classification, make_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import forestci as fci


def get_datasets():
    datasets = {}
    
    # 1. Auto MPG
    # Load the bundled Auto MPG data
    data_path = Path(__file__).resolve().parent / "data" / "auto_mpg.csv"
    df = pd.read_csv(data_path)
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

    wine = load_wine()
    datasets['Wine'] = (wine.data, wine.target, 'classification')
    
    return datasets

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def run_benchmark(output_dir=None):
    started = time.perf_counter()
    np.random.seed(42)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    plots = []
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
        
        # Keep each probability's calibration separate; reuse fitted forests.
        class_indices = ([None] if task == 'regression' else
                         [1] if len(ref_model.classes_) == 2 else
                         list(range(len(ref_model.classes_))))
        inbag_ref = fci.calc_inbag(X_train.shape[0], ref_model)
        reference = {}
        for k in class_indices:
            label = name if k is None else f"{name} class {ref_model.classes_[k]}"
            ref_var = fci.random_forest_error(
                ref_model, X_train.shape, X_test, inbag=inbag_ref,
                calibrate=False, class_index=k)
            if output_dir is not None:
                plots.append(plot_prediction_error(
                    label, ref_model, X_test, y_test, ref_var, output_dir,
                    class_index=k))
            reference[k] = (label, np.maximum(ref_var, 0))
        del inbag_ref, ref_model

        for n_trees in tree_counts:
            print(f"  Trees: {n_trees}", flush=True)
            model = model_class(n_estimators=n_trees, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            inbag = fci.calc_inbag(X_train.shape[0], model)
            for k, (label, ref_var) in reference.items():
                var_uncal = fci.random_forest_error(
                    model, X_train.shape, X_test, inbag=inbag,
                    calibrate=False, class_index=k)
                var_cal = fci.random_forest_error(
                    model, X_train.shape, X_test, inbag=inbag,
                    calibrate=True, class_index=k)
                neg_rate = np.mean(var_uncal < 0) * 100
                rmse_uncal = rmse(ref_var, np.maximum(var_uncal, 0))
                rmse_cal = rmse(ref_var, var_cal)
                rel_imp = ((rmse_uncal - rmse_cal) / rmse_uncal * 100
                           if rmse_uncal > 0 else np.nan)
                results.append({
                    'Dataset': label,
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
        dataset_name = r['Dataset']
        print(f"{dataset_name:<20} | {r['Trees']:<6} | {r['Neg Rate (Uncal)']:<16} | {r['Var RMSE (Uncal)']:<16} | {r['Var RMSE (Cal)']:<16} | {r['Relative Improvement']}")
    print("-" * 110)

    # Also output RST format for easy copy-paste
    print("\nRST Table format:")
    for r in results:
        dataset_name = r['Dataset']
        print(f"   * - {dataset_name}")
        print(f"     - {r['Trees']}")
        print(f"     - {r['Neg Rate (Uncal)']}")
        print(f"     - {r['Var RMSE (Uncal)']}")
        print(f"     - {r['Var RMSE (Cal)']}")
        print(f"     - {r['Relative Improvement']}")

    if output_dir is not None:
        write_table(results, output_dir / 'calibration_table.rst')
        plots.extend(gallery_comparisons(datasets['Auto MPG'], output_dir))
        (output_dir / 'prediction_error_figures.rst').write_text(
            '\n'.join(plots), encoding='utf-8')
    print(f"Total generation time: {time.perf_counter() - started:.1f} seconds")
    return results


def write_table(results, path):
    lines = ['.. list-table:: Recomputed calibration benchmark',
             '   :header-rows: 1', '',
             '   * - ' + '\n     - '.join(results[0])]
    for result in results:
        lines.append('   * - ' + '\n     - '.join(map(str, result.values())))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_prediction_error(name, model, X_test, y_test, variance, output_dir,
                          class_index=None):
    from matplotlib import pyplot as plt

    if class_index is None:
        prediction = model.predict(X_test)
    else:
        prediction = model.predict_proba(X_test)[:, class_index]
        y_test = (y_test == model.classes_[class_index]).astype(float)
    error = np.abs(y_test - prediction)
    half_width = 1.96 * np.sqrt(np.maximum(variance, 0))
    if not (np.isfinite(error).all() and np.isfinite(half_width).all()):
        raise ValueError(f'Non-finite diagnostic values for {name}')
    negative_count = np.count_nonzero(variance < 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), layout='constrained')
    ax.scatter(error, half_width, s=12, alpha=0.45, edgecolors='none')
    limit = max(error.max(), half_width.max()) * 1.04
    ax.plot([0, limit], [0, limit], '--', color='0.4', label='Equal magnitudes')
    ax.set(xlim=(0, limit), ylim=(0, limit),
           xlabel='Observed absolute prediction error',
           ylabel='Approximate 95% CI half-width (1.96 × IJ SE)',
           title=f'{name}: 2,000 estimators; {len(y_test):,} test samples')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left')
    filename = name.lower().replace(' ', '_') + '_prediction_error.png'
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)
    return (f'{name}\n' + '~' * len(name) + '\n\n'
            f'.. figure:: generated/benchmarks/{filename}\n'
            f'   :alt: Observed absolute error versus estimated CI half-width for {name}.\n\n'
            f'   {len(y_test):,} held-out samples; {negative_count} negative raw IJ '
            'variances clipped to zero for plotting.\n')


def gallery_comparisons(auto_mpg, output_dir):
    plots = []
    spam_X, spam_y = make_classification(5000, random_state=42)
    mpg_X, mpg_y, _ = auto_mpg
    cases = [
        ('Spam', spam_X, spam_y, 0.2,
         RandomForestClassifier(max_features=5, n_estimators=2000,
                                random_state=42, n_jobs=-1)),
        ('Auto MPG bagged SVR', mpg_X, mpg_y, 0.25,
         BaggingRegressor(estimator=SVR(), n_estimators=2000,
                          random_state=42, n_jobs=-1)),
    ]
    for name, X, y, test_size, model in cases:
        print(f'Running gallery comparison: {name}', flush=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        model.fit(X_train, y_train)
        class_index = 1 if isinstance(model, RandomForestClassifier) else None
        variance = fci.random_forest_error(
            model, X_train.shape, X_test, calibrate=False,
            class_index=class_index)
        plots.append(plot_prediction_error(
            name, model, X_test, y_test, variance, output_dir,
            class_index=class_index))
    return plots


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        help='Write generated documentation tables and plots here.')
    args = parser.parse_args()
    run_benchmark(args.output_dir)
