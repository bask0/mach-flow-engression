import numpy as np


def find_rank(obs: float, members: np.ndarray) -> int:
    if np.isnan(obs) or np.isnan(members).any():
        return -1
    return np.digitize(obs, np.sort(members), right=True) + 1

def compute_chi_squared(ranks: np.ndarray, num_members: int) -> float:
    num_in_bin = np.zeros(shape=num_members+1)
    for rank in ranks:
        num_in_bin[rank-1] += 1

    chi_squared = 0
    n_over_mplusone = np.divide(len(ranks), num_members+1)
    for count in num_in_bin:
        add_term = np.square(count - n_over_mplusone)
        add_term = np.divide(add_term, n_over_mplusone)
        chi_squared += add_term
        
    return chi_squared

    
def find_quantile_index(obs: float, quantiles: np.ndarray) -> int:
    if np.isnan(obs) or np.isnan(quantiles).any():
        return -1
    return np.digitize(obs, quantiles, right=True)

def compute_cdf(quantile_indices: np.ndarray, num_quantiles: int) -> np.ndarray:
    cumulative_counts = np.zeros(shape=num_quantiles+1)
    for el in quantile_indices:
        if not np.isnan(el):
            cumulative_counts[int(el):] += 1

    cdf = cumulative_counts / len(quantile_indices)
    cdf = cdf[:-1] # the last element is just 1.0 (highest quantile index is for all values larger than last quantile)

    return cdf

# def compute_cdf(quantile_indices: np.ndarray, quantize_vec: np.ndarray) -> np.ndarray:
#     hist, _ = np.histogram(quantile_indices, quantize_vec, density=True)
#     return hist.cumsum()[:-1]

def compute_sum_of_squared_deviations(cdf_1: np.ndarray, cdf_2: np.ndarray) -> float:
    return np.sum(np.square(cdf_1 - cdf_2))

def compute_nse(mod: np.ndarray, obs: np.ndarray) -> float:
    return 1 - np.sum(
        np.square(mod - obs), axis=-1
    ) / np.sum(
        np.square(obs - np.mean(obs, axis=-1, keepdims=True)), axis=-1
    )
