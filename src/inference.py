import numpy as np
from scipy import stats


class StatisticalInference:
    """
    Toolkit for Statistical Inference:
    - Maximum Likelihood Estimation (MLE)
    - Log-Likelihood Calculation
    - Confidence Intervals (Asymptotic Normal Approximation)
    """

    # --- GAUSSIAN MODEL ---
    @staticmethod
    def gaussian_mle(data):
        """Returns MLE: mu_hat (Mean), sigma_hat (Standard Deviation)."""
        return np.mean(data), np.std(data, ddof=0)

    @staticmethod
    def gaussian_log_likelihood(data, mu, sigma):
        """Calculates total Log-Likelihood for Gaussian."""
        n = len(data)
        return - (n/2) * np.log(2 * np.pi) - n * np.log(sigma) - np.sum((data - mu)**2) / (2 * sigma**2)

    @staticmethod
    def gaussian_confidence_interval(data, confidence=0.95):
        """Returns (Lower, Upper) CI for the Mean."""
        n = len(data)
        mu = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(n)  # Standard Error
        h = sem * stats.norm.ppf((1 + confidence) / 2)
        return mu - h, mu + h

    # --- LAPLACE MODEL ---
    @staticmethod
    def laplace_mle(data):
        """Returns MLE: mu_hat (Median), b_hat (Mean Abs Deviation)."""
        mu = np.median(data)
        b = np.mean(np.abs(data - mu))
        return mu, b

    @staticmethod
    def laplace_log_likelihood(data, mu, b):
        """Calculates total Log-Likelihood for Laplace."""
        n = len(data)
        # Log L = -n*log(2b) - (1/b) * sum(|errors|)
        return -n * np.log(2 * b) - np.sum(np.abs(data - mu)) / b

    @staticmethod
    def laplace_confidence_interval(data, confidence=0.95):
        """
        Returns (Lower, Upper) CI for the Median (Location).
        Asymptotic Variance of Median = 1 / (4 * n * f(0)^2)
        """
        n = len(data)
        mu, b = StatisticalInference.laplace_mle(data)
        # Fisher Information Inverse approximation
        se_median = b / np.sqrt(n)
        h = se_median * stats.norm.ppf((1 + confidence) / 2)
        return mu - h, mu + h
