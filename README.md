# SpatialANN

This repository contains the SpatialANN interpolation code used to generate the MWS03 daily maximum wind speed dataset for China during 1979–2019. The workflow loads the trained fold01 model and produces monthly NetCDF files containing daily predictions. It does not retrain the model.

The scripts run on a **Linux system with Bash and the SLURM workload manager**. Submit jobs from a login node using an activated Python 3.10 environment with PyTorch, NumPy, pandas, SciPy, scikit-learn, openpyxl, and netCDF4 available. The same environment and configured paths must be accessible from the compute nodes. CPU inference is the default; GPU inference is optional.

## Repository contents

```text
SpatialANN/
├── README.md
├── config.sh
├── submit_all.sh
├── run_worker.sh
├── check_results.sh
├── retry_tasks.sh
├── product.py
├── model_source.py
└── data/
    └── fold01/
        ├── Table_fold01_node_split_metadata.csv
        └── ckpt/
            ├── cnp_gpd_tail_no_coordinate_features_5fold_3yr_era5land.pt
            ├── inference_metadata.json
            └── checkpoint_sha256.txt
```

- `model_source.py` defines the model and its feature construction functions.
- `product.py` prepares features, performs gridded inference, and writes and checks the monthly products.
- `data/fold01/ckpt/` contains the trained weights, model configuration, feature order, and saved normalization parameters.
- `Table_fold01_node_split_metadata.csv` preserves station identifiers, coordinates, node indices, and fold01 roles required by the model.

## Running the code

### 1. Set paths and SLURM options

From the repository root, edit `config.sh` or set environment variables:

```bash
export S56_PYTHON="/absolute/path/to/environment/bin/python"
export S56_INPUT_DIR="/absolute/path/to/grid_inputs"
export S56_STATION_CACHE="/absolute/path/to/station_cache"
export S56_PARTITION="your_partition"
export S56_TIME_LIMIT="24:00:00"
```

Set paths to the corresponding inputs in their expected format. `S56_MAIN_RESULT` defaults to the repository's `data/` directory. Outputs, working caches, and logs default to `result/`, `cache/`, and `logs/` within the repository.

The default allocation is **8 CPUs and 48 GB RAM per task**, with at most **10 product tasks running concurrently**. Adjust the partition, time limit, memory, and concurrency to your cluster. These are scheduling defaults, not guaranteed minimum resource requirements. By default, the preparation job uses the same partition as the product jobs.

For GPU inference, also set:

```bash
export S56_DEVICE="cuda"
export S56_PARTITION="your_gpu_partition"
export S56_PREP_PARTITION="your_cpu_partition"
export S56_GRES="gpu:1"
```

### 2. Submit the jobs

```bash
bash submit_all.sh
```

The script first checks the configured paths. It then submits one preparation job followed by a dependent array of 10 product jobs. The product jobs start only after preparation succeeds. Each job processes one group of years; together they cover 1979–2019. Submission IDs are printed and saved in `logs/last_submission.env`.

### 3. Monitor and check the outputs

```bash
squeue -u "$USER"
bash check_results.sh
```

Run `check_results.sh` after all product jobs have finished. It checks that the expected monthly products are complete. Job messages and errors are written to `logs/`.

To resubmit failed product tasks after preparation has succeeded, wait until those tasks have stopped, then specify their array indices, for example:

```bash
bash retry_tasks.sh 3,7
```

If preparation fails, cancel its waiting product array with `scancel PRODUCT_JOB_ID`, correct the reported problem, and rerun `submit_all.sh`.

## Outputs

The complete run produces **492 monthly NetCDF files**, named:

```text
MWS03_dailymax_wind_0p1deg_197901.nc
...
MWS03_dailymax_wind_0p1deg_201912.nc
```

Each file contains daily 10 m maximum wind speed in the variable `wind`, expressed in `m s-1`. Files contain daily values, not monthly averages. The time axis follows the Gregorian calendar and includes leap days.

The workflow retains the supplied grid coordinates without resampling. Depending on the coordinate layout, values are stored as `wind(time, lat, lon)` or `wind(time, point)`. Model weights are loaded without refitting. Feature definitions, saved normalization parameters, and the original fold01 inference conventions are preserved.
