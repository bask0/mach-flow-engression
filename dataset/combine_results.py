from argparse import ArgumentParser
import xarray as xr
import numpy as np
from glob import glob
import time
import re
import os


def infer_config(file_path: str) -> dict:
    """Extracts the configuration (beta, esl, nl) from a file path.

    Args:
        file_path: A file path to configuration: .../beta<>_esl<>_nl<>/...

    Returns:
        A dictionary of the configuration {'beta': ..., 'esl': ..., 'nl': ...}
    """
    # Regular expression patterns to match the desired numbers
    beta_pattern = r'beta(\d+\.\d+|\d+)'  # Matches 'beta' followed by a number (with or without a decimal point)
    esl_pattern = r'esl(\d+\.\d+|\d+)'    # Matches 'esl' followed by a number (with or without a decimal point)
    nl_pattern = r'nl(\d+\.\d+|\d+)'      # Matches 'nl' followed by a number (with or without a decimal point)

    # Search for the patterns in the file path
    beta_match = re.search(beta_pattern, file_path)
    esl_match = re.search(esl_pattern, file_path)
    nl_match = re.search(nl_pattern, file_path)

    # Extract the numbers if matches are found
    beta_number = beta_match.group(1) if beta_match else None
    esl_number = esl_match.group(1) if esl_match else None
    nl_number = nl_match.group(1) if nl_match else None

    # Check if values exists and combine into dictionary.
    expand_dim_kwargs = {}
    for key, value in zip(['beta', 'esl', 'nl'], [beta_number, esl_number, nl_number]):
        if value is None:
            raise ValueError(f'could not infer parameter `{key}` from file path `{file_path}`.')
        
        try:
            if key == 'beta':
                value = float(value)
            else:
                value = int(value)
        except ValueError as e:
            raise ValueError(
                f'could not cast parameter `{key}={value}` to numeric.'
            )

        expand_dim_kwargs[key] = value

    return expand_dim_kwargs


def get_fold_dirs(xval_dir: str) -> list[str]:
    """Returns a list of file paths for the CV folds given a CV path.

    Args:
        xval_dir: A path to a CV directory: .../xval/

    Returns:
        A list of paths of all prediction fil;es in the given CV directory.
    """
    return glob(os.path.join(xval_dir, 'fold*/preds.zarr'))


def load_folds(xval_dir: str, return_full_ds: bool = False) -> tuple[xr.Dataset | xr.DataArray, dict]:
    """Loads all folds for a given CV directory.
    
    Args:
        xval_dir: A path to a CV directory: .../xval/
        return_full_ds: if True, the full stacked fold dataset is returned, else only the predictions (Qmm_mod).

    Returns:
        - Either a dataset containing all the folds if return_full_ds=True, or the predicted variables only else.
        - The configuration.
    """
    fold_paths = get_fold_dirs(xval_dir)

    expand_dim_kwargs = infer_config(xval_dir)

    if return_full_ds:
        ds_list = [xr.open_zarr(fold_path) for fold_path in fold_paths]
        ds = xr.concat(ds_list, dim='station')
        return ds.load(), expand_dim_kwargs

    else:
        da_list = [xr.open_zarr(fold_path)['Qmm_mod'] for fold_path in fold_paths]
        da = xr.concat(da_list, dim='station')
        # da = da.expand_dims(**expand_dim_kwargs)
        return da.load(), expand_dim_kwargs


def get_all_configs(config: list[dict[str, list]]) -> dict[str, list]:
    expand_dims = {}

    for el in config:
        for key, value in el.items():
            if key not in expand_dims:
                expand_dims[key] = []

            if value not in expand_dims[key]:
                expand_dims[key].append(value)

    return expand_dims


def postprocess_results(
        save_path: str,
        runs_dir: str = '/net/argon/landclim/kraftb/machflow_engression/runs') -> None:
    """Load all configurations into a dataset.

    The resulting dataset contains all folds and the configurations in a new dimension.

    Args:
        save_path: file path of the combined .zarr file, .../combined.zarr or similar.
        runs_dir: the directory containing the results, default is
            `/net/argon/landclim/kraftb/machflow_engression/runs`

    """

    t = time.time()

    # Get all configuration directories.
    try:
        config_dirs = glob(os.path.join(runs_dir, 'beta*/xval'))

    except IndexError as e:
        config_dirs = glob(os.path.join(runs_dir, 'default/xval'))

    # Load a single dataset and drop the predictions (we add them later below).
    print('> Loading data.')
    ds, _ = load_folds(config_dirs[0], return_full_ds=True)

    # We iterate all the configuration dirs and load all folds from them.
    config_da = []
    config_dicts = []
    for i, config_dir in enumerate(config_dirs):
        if i == 0 or (i + 1) % 5 == 0:
            print(f'  > Loading config {i+1:3d} of {len(config_dirs)}')

        # Append prediction (Qmm_mod) for a certain configuration. Noe that the folds have alraedy been combined.
        da, config_dict = load_folds(config_dir, return_full_ds=False)
        config_dicts.append(config_dict)

        config_da.append(da.sel(station=ds.station))

    # Get all configurations from list of configurations.
    all_configs = get_all_configs(config_dicts)

    print(f'> Combining data into single container.')

    # Create an empty data array within the dataset.
    ds['Qmm_mod'] = xr.full_like(ds['Qmm_mod'], fill_value=np.nan).expand_dims(all_configs).copy()

    # Combine the configurations and add them back to the full dataset. This is faster than using xarray
    # combine method.
    for da, conf in zip(config_da, config_dicts):
        ds['Qmm_mod'].loc[conf] = da

    print(f'> Writing results to file.')

    # Define variable-wise chunks.
    encoding = {}

    for var in ds.data_vars:
        if var == 'Qmm_mod':
            chunks = (1, 1, 1, -1, -1, 1000,)
        elif 'time' in ds[var].dims:
            chunks = (-1, 1000,)
        else:
            chunks = (-1,)

        encoding[var] = {
            'chunks': chunks,
            'compressor': None
        }

    ds.to_zarr(save_path, mode='w', encoding=encoding)

    elapsed_seconds = time.time() - t
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    print(f'> Done! Elapsed time: {minutes} minutes {seconds} seconds.')
    print(f'File saved to: {save_path}')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--runs_dir', type=str, help='run directory',
        default='/net/argon/landclim/kraftb/machflow_engression/runs')
    parser.add_argument(
        '--out_path', type=str, help='output path',
        default='/net/argon/landclim/kraftb/machflow_engression/runs/combined.zarr')

    args = parser.parse_args()

    postprocess_results(save_path=args.out_path, runs_dir=args.runs_dir)
