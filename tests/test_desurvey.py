from src.geostax.utils.desurvey import compute_trajectories, bulk_trajectories
from src.geostax.ingestion import read_data

import yaml
import pandas as pd
import numpy as np

CONFIG_PATH = "config/test_paths.yaml"

def test_compute_trajectories():

    """
    Ensures the the maximum elevation of each hole is the first row after sorting by distance.
    """

    # Read data paths

    with open(CONFIG_PATH) as stream:
        try:
            paths = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Read data

    df_collars, df_survey, df_assays = read_data(paths)

    # Idiosyncratic preprocessing needed for collars dataset

    df_collars["season_start"] = df_collars["Season"].str.split("/").str[0]  # .astype(int)
    df_collars["season_start"] = df_collars["season_start"].astype(float)
    df_collars["season_end"] = df_collars["Season"].str.split("/").str[1]  # .astype(int)
    df_collars["season_end"] = df_collars["season_end"].astype(float)

    df_traj = compute_trajectories("FSDH093", df_survey, df_collars)
    assert df_traj["z"].idxmax() == 0

def test_bulk_trajectories():

    """
    Ensures the inferred x, y, z are reasonable for all drillholes
    """

    # Read data paths

    with open(CONFIG_PATH) as stream:
        try:
            paths = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Read data

    df_collars, df_survey, df_assays = read_data(paths)

    # Idiosyncratic preprocessing needed for collars dataset

    df_collars["season_start"] = df_collars["Season"].str.split("/").str[0]  # .astype(int)
    df_collars["season_start"] = df_collars["season_start"].astype(float)
    df_collars["season_end"] = df_collars["Season"].str.split("/").str[1]  # .astype(int)
    df_collars["season_end"] = df_collars["season_end"].astype(float)
    holes_of_interest = df_collars["HOLE-ID"].to_list()

    # Compute trajectories of all drill holes

    df_traj = bulk_trajectories(df_survey, df_collars, holes_of_interest)

    # Test condition to make sure that the maximum z for all holes is close to their "elevation"

    df_traj_zmax = df_traj.groupby("hole_id")[["z", "Distance (m)"]].max().reset_index()
    df_traj_zmax.columns = ["HOLE-ID", "z_max_survey", "distance_m_max"]
    df_elev = df_collars[["HOLE-ID", "Elevation"]]
    df_tmp = pd.merge(df_traj_zmax, df_elev, on="HOLE-ID", how="outer")
    df_tmp["diff"] = np.abs(df_tmp["z_max_survey"] - df_tmp["Elevation"])
    df_tmp = df_tmp.sort_values("diff", ascending=False)
    df_tmp = df_tmp[df_tmp["diff"] > 0]
    assert df_tmp.shape[0] == 0