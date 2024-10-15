import numpy as np
import tinygp
import jax
import jax.numpy as jnp

from tqdm import tqdm

def batch_predict(
    X_test: np.array, 
    gp: tinygp.GaussianProcess, 
    y_train_sc: jax.Array, 
    batch_size: int,
    cpu_device: jax.Device,
    gpu_device: jax.Device,
):
    """
    Splits X_test into smaller subsets (based on batch_size)
    Moves smaller subsets to GPU one at a time
    Predicts on these smaller subsets on the GPU
    Writes the results back to a list on the CPU

    Args:
        X_test: Numpy array to predict (N_test x 3)
        gp: A "fitted" Gaussian Process object from tinygp library
        y_train_sc: The observed training data
        batch_size: Number of rows of X_test to split at a time
        cpu_device: jax.device for storing X_test
        gpu_device: jax.device for computing the GP predict method (matrix inverse etc.)

    Returns:
        y_mean_pred: jnp.array, mean predictions for each input row, hosted on the cpu_device
        y_std_pred: jnp.array, std estimate for each input row, hosted on the cpu_device

    """

    # Get the necessary indices to split X_test over

    idx_range = np.arange(0, X_test.shape[0], batch_size)
    idx_range = np.concatenate([idx_range, [X_test.shape[0] + 1]])

    # Empty lists for results

    y_mean_pred = []
    y_std_pred = []

    # Split X_test into small subsets, make predictions on GPU, then move back to CPU.

    for i, j in tqdm(zip(idx_range[:-1], idx_range[1:])):

        # Create a subset of X_test and move it to the GPU

        X_tmp = jax.device_put(X_test[i:j, :], gpu_device)

        # Mean and variance for X_tmp, move y_tmp_* back to CPU

        cond_gp = gp.condition(y_train_sc.flatten(), X_tmp).gp
        y_tmp_mean = jax.device_put(cond_gp.mean, device=cpu_device)
        y_tmp_std = jax.device_put(np.sqrt(cond_gp.variance), device=cpu_device)

        # Append to the mean and std list on cpu

        y_mean_pred.append(y_tmp_mean)
        y_std_pred.append(y_tmp_std)

    # Concatenate predictive means and stds

    y_mean_pred = jnp.concat(y_mean_pred)
    y_std_pred = jnp.concat(y_std_pred)

    return y_mean_pred, y_std_pred

def batch_sample(
    X_test: np.array, 
    gp: tinygp.GaussianProcess, 
    y_train_sc: jax.Array, 
    batch_size: int,
    cpu_device: jax.Device,
    gpu_device: jax.Device,
    key,
    n_samples: int = 1,
):
    """
    Similar method to `batch_predict`, but samples from the gp instead.
    TODO: maybe integrate the two methods into one?

    Args:
        X_test: Numpy array to predict (N_test x 3)
        gp: A "fitted" Gaussian Process object from tinygp library
        y_train_sc: The observed training data
        batch_size: Number of rows of X_test to split at a time
        cpu_device: jax.device for storing X_test
        gpu_device: jax.device for computing the GP predict method (matrix inverse etc.)
        key: jax.random.key object (TODO: type hint this properly)

    Returns:
        y_samples: jnp.array, samples drawn from the fitted GP

    """

    # Get the necessary indices to split X_test over

    idx_range = np.arange(0, X_test.shape[0], batch_size)
    idx_range = np.concatenate([idx_range, [X_test.shape[0] + 1]])

    # Empty lists for results

    y_samples = []

    # Split X_test into small subsets, make predictions on GPU, then move back to CPU.

    for i, j in tqdm(zip(idx_range[:-1], idx_range[1:])):

        # Create a subset of X_test and move it to the GPU

        X_tmp = jax.device_put(X_test[i:j, :], gpu_device)

        # Mean and variance for X_tmp, move y_tmp_* back to CPU

        cond_gp = gp.condition(y_train_sc.flatten(), X_tmp).gp
        y_sample_tmp = jax.device_put(cond_gp.sample(key, (n_samples, )), device=cpu_device)

        # Samples will have shape (n_samples, batch_size)

        y_samples.append(y_sample_tmp)

    # Concatenate the samples

    y_samples = jnp.hstack(y_samples).T

    return y_samples