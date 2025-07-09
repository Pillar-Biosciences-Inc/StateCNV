"""
comparison_methods.py

This module contains baseline and evaluation routines used to compare CNVBayesProfiler
against alternative smoothing and classification methods. These methods are used
to reproduce the results and figures presented in the paper:
    "Classifying Copy Number Variations Using State Space Modeling of Targeted Sequencing Data"

It includes:
- RMSE evaluation between predicted CNV ratios and ground-truth diagnostic profiles.
- Similarity scoring based on closest vector distance in log-ratio space.
- Cross-validation and spline-based smoothing routines.
- Integration of sklearn and scipy tools for benchmarking.

Note: This script is intended for research comparison and is not part of the core StateCNV package.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import norm


def find_closest_vector_index(target_vector, vector_list):
    """
    Finds the index of the vector in `vector_list` that is closest to `target_vector`
    using the L2 norm (Euclidean distance).

    Parameters:
    - target_vector: A 1D numpy array.
    - vector_list: A list of 1D numpy arrays (all must have the same shape as target_vector).

    Returns:
    - The index of the closest vector in the list.
    """
    vector_stack = np.stack(vector_list)  # shape (n_vectors, vector_dim)
    distances = np.linalg.norm(vector_stack - target_vector, axis=1)
    return np.argmin(distances)


def evaluate_rmse_by_diagnosis(
    diagnosis_names2,
    snames,
    key_alpha,
    log_ratios,
    filtered_name_a,
    alpha_ratios,
):
    """
    Computes RMSE between log_ratios and diagnosis_3 values for matching diagnoses.

    Parameters:
    - diagnosis_names: list of predicted diagnosis strings (length N)
    - snames: list of sample IDs (length N)
    - key_alpha: DataFrame with 'Sample ID' and 'Diagnosis' columns
    - log_ratios: list or array of predicted log-ratio values (length N)
    - diagnosis_3: list or array of ground-truth log-ratio values for each diagnosis string

    Returns:
    - rmse: root mean squared error between predicted log_ratios and matched diagnosis_3
    """
    errors = []

    for i in range(len(diagnosis_names2)):
        sample_id = snames[i]

        row = key_alpha[key_alpha["Sample ID"] == sample_id]
        if row.empty:
            raise ValueError(f"Sample ID '{sample_id}' not found in key_alpha.")

        true_diag = row["Diagnosis"].values[0]

        try:
            index = filtered_name_a.index(true_diag)
        except ValueError:
            raise ValueError(
                f"True diagnosis '{true_diag}' not found in diagnosis_names2."
            )

        true_value = alpha_ratios[index]
        predicted_value = log_ratios[i]
        errors.append((predicted_value - true_value) ** 2)

    rmse = np.mean(np.sqrt(errors)) if errors else float("nan")
    return rmse, np.sqrt(errors)


def evaluate_diagnosis_accuracy(diagnosis_names2, snames, key_alpha):
    """
    Compares predicted diagnoses to true diagnoses in `key_alpha`.
    Returns:
    - match_vector: np.array of 1s and 0s for exact string match
    - TP, FP, TN, FN: counts of true positives, false positives, true negatives, false negatives
    """
    match_vector = []
    TP = FP = TN = FN = 0
    normal = "No Alpha Thalassemia (Normal)"

    for i in range(len(diagnosis_names2)):
        sample_id = snames[i]
        predicted = diagnosis_names2[i]

        # Find the corresponding true diagnosis
        row = key_alpha[key_alpha["Sample ID"] == sample_id]
        if row.empty:
            raise ValueError(f"Sample ID '{sample_id}' not found in key_alpha.")

        true_diag = row["Diagnosis"].values[0]

        # Exact string match (1 for match, 0 for no match)
        match_vector.append(int(predicted == true_diag))

        # Categorize outcome
        if true_diag == normal and predicted == normal:
            TN += 1
        elif true_diag == normal and predicted != normal:
            FP += 1
        elif true_diag != normal and predicted == normal:
            FN += 1
        else:  # both not normal
            TP += 1

    return np.array(match_vector), TP, FP, TN, FN


def evaluate_diagnosis_accuracy_beta(diagnosis_names2, snames, key_alpha):
    """
    Compares predicted diagnoses to true diagnoses in `key_alpha`.
    Returns:
    - match_vector: np.array of 1s and 0s for exact string match
    - TP, FP, TN, FN: counts of true positives, false positives, true negatives, false negatives
    """
    match_vector = []
    TP = FP = TN = FN = 0
    normal = "No Beta Thalassemia (Normal)"

    for i in range(len(diagnosis_names2)):
        sample_id = snames[i]
        predicted = diagnosis_names2[i]

        # Find the corresponding true diagnosis
        row = key_alpha[key_alpha["Sample ID"] == sample_id]
        if row.empty:
            raise ValueError(f"Sample ID '{sample_id}' not found in key_alpha.")

        true_diag = row["Diagnosis"].values[0]

        # Exact string match (1 for match, 0 for no match)
        match_vector.append(int(predicted == true_diag))

        # Categorize outcome
        if true_diag == normal and predicted == normal:
            TN += 1
        elif true_diag == normal and predicted != normal:
            FP += 1
        elif true_diag != normal and predicted == normal:
            FN += 1
        else:  # both not normal
            TP += 1

    return np.array(match_vector), TP, FP, TN, FN


def compute_sensitivity_specificity(TP, FP, TN, FN):
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    specificity = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    return sensitivity, specificity


def sum_confusion_counts(dict_list):
    """
    Sums the confusion matrix values across a list of dictionaries.

    Parameters:
    - dict_list: list of dictionaries, each with keys 'TP', 'FP', 'FN', 'TN'

    Returns:
    - TP, FP, FN, TN: summed values
    """
    TP = FP = FN = TN = 0
    for d in dict_list:
        TP += d.get("TP", 0)
        FP += d.get("FP", 0)
        FN += d.get("FN", 0)
        TN += d.get("TN", 0)
    return TP, FP, FN, TN


def anisotropic_diffusion_1d(signal, num_iter=50, kappa=20, lambda_=0.25):
    """
    Performs 1D anisotropic diffusion on a signal.

    Parameters:
        signal (np.ndarray): 1D input signal (e.g., image row).
        num_iter (int): Number of diffusion iterations.
        kappa (float): Edge threshold parameter.
        lambda_ (float): Diffusion speed (must be <= 0.25 for stability).

    Returns:
        np.ndarray: Diffused signal.
    """
    signal = signal.astype(np.float64)
    result = signal.copy()

    for _ in range(num_iter):
        # Compute forward and backward differences
        delta_forward = np.roll(result, -1) - result
        delta_backward = result - np.roll(result, 1)

        # Optional: Zero-padding the ends (or replicate boundary)
        delta_forward[-1] = 0
        delta_backward[0] = 0

        # Compute diffusion coefficients using Perona-Malik (exponential form)
        c_forward = np.exp(-((delta_forward / kappa) ** 2))
        c_backward = np.exp(-((delta_backward / kappa) ** 2))

        # Update the signal
        result += lambda_ * (
            c_forward * delta_forward - c_backward * delta_backward
        )

    return result


def nadaraya_watson(x_train, y_train, x_eval, bandwidth):
    """
    Perform Nadaraya-Watson kernel regression using a Gaussian kernel.

    Parameters:
    - x_train: 1D array of training x-values
    - y_train: 1D array of training y-values
    - x_eval: 1D array of x-values to evaluate the regression at
    - bandwidth: bandwidth (kernel width) for the Gaussian kernel

    Returns:
    - y_pred: 1D array of predicted y-values at x_eval
    """
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_eval = np.asarray(x_eval)

    y_pred = np.zeros_like(x_eval, dtype=float)

    for i, x in enumerate(x_eval):
        weights = norm.pdf((x - x_train) / bandwidth)
        weights_sum = weights.sum()
        y_pred[i] = (
            np.dot(weights, y_train) / weights_sum if weights_sum > 0 else 0.0
        )

    return y_pred


def loocv_kernel_regression(x, y, bandwidths):
    """
    Perform Leave-One-Out Cross-Validation to select the best bandwidth.

    Parameters:
    - x: 1D array of x-values
    - y: 1D array of y-values
    - bandwidths: list or array of bandwidths to try

    Returns:
    - best_bandwidth: bandwidth that minimizes the LOOCV error
    - errors: list of LOOCV errors corresponding to each bandwidth
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    errors = []

    for h in bandwidths:
        preds = []
        for i in range(n):
            x_train = np.delete(x, i)
            y_train = np.delete(y, i)
            x_test = x[i]
            y_test = y[i]
            pred = nadaraya_watson(x_train, y_train, [x_test], bandwidth=h)[0]
            preds.append((pred - y_test) ** 2)
        errors.append(np.mean(preds))

    best_idx = np.argmin(errors)
    best_bandwidth = bandwidths[best_idx]
    return best_bandwidth, errors


def cross_validated_smoothing_spline(x, y, s_values, k=5):
    """
    Selects optimal smoothing parameter s via hold-k-out cross-validation.

    Parameters:
    - x, y: 1D numpy arrays of input data.
    - s_values: List or array of smoothing strengths to evaluate.
    - k: Number of hold-out sets (folds) to use.

    Returns:
    - best_s: Smoothing parameter with lowest CV error.
    - best_spline: Fitted UnivariateSpline on the full data using best_s.
    - cv_errors: List of mean CV errors for each s in s_values.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    cv = KFold(n_splits=k, shuffle=True, random_state=42)

    cv_errors = []

    for s in s_values:
        fold_errors = []

        for train_idx, val_idx in cv.split(x):
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            # Fit smoothing spline on training data
            spline = UnivariateSpline(x_train, y_train, s=s)

            # Predict on validation set
            y_pred = spline(x_val)

            # Compute MSE on the validation set
            fold_errors.append(mean_squared_error(y_val, y_pred))

        # Average error across folds
        cv_errors.append(np.mean(fold_errors))

    # Best s has lowest average CV error
    best_idx = np.argmin(cv_errors)
    best_s = s_values[best_idx]
    best_spline = UnivariateSpline(x, y, s=best_s)
    y_smooth = best_spline(x)

    return best_s, best_spline, cv_errors, y_smooth
