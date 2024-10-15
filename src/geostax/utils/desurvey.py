import numpy as np
import pandas as pd
import pandas as pd
import numpy as np

from typing import List

from tqdm import tqdm

def compute_trajectories(
    hole_id: str, 
    df_survey: pd.DataFrame, 
    df_collars: pd.DataFrame
):
    """
    Transform dip and azimuth to construct (X, Y, Z) for the boreholes
    """

    # df_survey data

    df_tmp = df_survey.loc[df_survey["HOLE-ID"] == hole_id]
    df_tmp = df_tmp.sort_values("Distance (m)")

    # Extract parameters for the calculation
    # Dip converted to negative from pygslib convention

    L = df_tmp["Distance (m)"].to_numpy()
    dip = (
        -1 * df_tmp["Dip"].to_numpy()
    )  
    azimuth = df_tmp["Azimuth"].to_numpy()

    # Initialize X0, Y0 and Z0 using the collar dataset

    hole_cond = df_collars["HOLE-ID"].isin([hole_id])
    X0 = df_collars.loc[hole_cond, "Easting"].to_numpy()[0]
    Y0 = df_collars.loc[hole_cond, "Northing"].to_numpy()[0]
    Z0 = df_collars.loc[hole_cond, "Elevation"].to_numpy()[0]

    # Intermediate Calculations

    dL = L[1:] - L[:-1]

    # METHOD 1

    # In spherical coordinates theta is measured from zenith down
    # you are measuring it from horizontal plane up

    phi = azimuth[1:] * (np.pi / 180)
    theta = (90.0 - dip[1:]) * (np.pi / 180)
    dX = dL * np.cos(phi) * np.sin(theta)
    dY = dL * np.sin(phi) * np.sin(theta)
    dZ = dL * np.cos(theta)

    # Increment X0, Y0 and Z0 using dX, dY, dZ

    X_f = X0 + np.cumsum(dX)
    Y_f = Y0 + np.cumsum(dY)
    Z_f = Z0 + np.cumsum(dZ)

    # Append X0, Y0 and Z0

    X_f = np.concatenate((np.array([X0]), X_f))
    Y_f = np.concatenate((np.array([Y0]), Y_f))
    Z_f = np.concatenate((np.array([Z0]), Z_f))

    # Create a dataframe with hole id etc.

    df_trajectory = pd.DataFrame(
        {
            "hole_id": hole_id,
            "x": X_f,
            "y": Y_f,
            "z": Z_f,
        }
    )

    # Append distance

    df_trajectory = pd.concat(
        [df_trajectory, df_tmp.set_index(df_trajectory.index)], axis=1
    )

    return df_trajectory


def bulk_trajectories(
    df_survey: pd.DataFrame, df_collars: pd.DataFrame, holes_of_interest: List
):

    # Loop over all drillholes

    df_traj = []

    for hole in tqdm(holes_of_interest):
        df_traj.append(compute_trajectories(hole, df_survey, df_collars))

    df_traj = pd.concat(df_traj).reset_index(drop=True)

    return df_traj

def extract_xyz_assays(df_traj: pd.DataFrame, df_assays: pd.DataFrame):
    """
    """

    # Initalize list to store results
    
    df_assay_coords = []
    
    # Assay matching algorithm
    # TODO: parallelize this over multiple cores?
    
    for hole_id in tqdm(df_traj["HOLE-ID"].unique()):
    
        # Filter df_traj for a given hole
    
        df_traj_filt = df_traj.loc[df_traj["hole_id"] == hole_id].reset_index(drop=True)
    
        # Filter df_assays
    
        df_assays_filt = df_assays.loc[df_assays["HOLE-ID"] == hole_id].reset_index(
            drop=True
        )
        df_assays_filt["mid_distance"] = df_assays_filt[["From (m)", "To (m)"]].mean(axis=1)
    
        # Loop over all assays
    
        for assay_num, _ in df_assays_filt.iterrows():
    
            # Get the mid distance
    
            assay_mid_dist = df_assays_filt.loc[assay_num, "mid_distance"]
    
            # Logic should only work if assay distance is less than max distance and greater than min distance of survey data
    
            allowed_cond = (assay_mid_dist <= df_traj_filt["Distance (m)"].max()) & (
                assay_mid_dist >= df_traj_filt["Distance (m)"].min()
            )
    
            if allowed_cond:
    
                # Find the lower and upper coordinate values from the borehole survey
    
                if any(df_traj_filt["Distance (m)"] == assay_mid_dist):
    
                    # Case 1: where the distance falls exactly on a borehole survey measurement
    
                    df_coords = df_traj_filt.loc[
                        df_traj_filt["Distance (m)"] == assay_mid_dist
                    ]
                    df_coords = df_coords[["hole_id", "x", "y", "z"]]
                    df_coords.columns = ["hole_id", "x_inf", "y_inf", "z_inf"]
    
                else:
    
                    # Case 2: where the distance is in between an interval
    
                    # Get lower and upper bound indices
    
                    distance_diff = df_traj_filt["Distance (m)"] - assay_mid_dist
                    lower_bound = np.argmax(distance_diff[distance_diff < 0])
                    upper_bound = lower_bound + 1
    
                    # Lower and upper bound coordinates and distances
    
                    df_lb = df_traj_filt.loc[lower_bound, ["x", "y", "z", "Distance (m)"]]
                    df_ub = df_traj_filt.loc[upper_bound, ["x", "y", "z", "Distance (m)"]]
    
                    # Compute estimated location of x, y, z for the sample based on a "straight-line between lb and ub coordinates
    
                    df_coords = pd.DataFrame()
                    t = (assay_mid_dist - df_lb["Distance (m)"]) / (
                        df_ub["Distance (m)"] - df_lb["Distance (m)"]
                    )  # parameter for 3d vector form
                    df_coords["hole_id"] = df_traj_filt["hole_id"].unique()
                    df_coords["x_inf"] = df_lb["x"] + t * (df_ub["x"] - df_lb["x"])
                    df_coords["y_inf"] = df_lb["y"] + t * (df_ub["y"] - df_lb["y"])
                    df_coords["z_inf"] = df_lb["z"] + t * (df_ub["z"] - df_lb["z"])
    
            else:
    
                df_coords = pd.DataFrame()
                df_coords["hole_id"] = df_traj_filt["hole_id"].unique()
                df_coords["x_inf"] = np.nan
                df_coords["y_inf"] = np.nan
                df_coords["z_inf"] = np.nan
    
            # Merge df_coords with filtered assay info
    
            df_assay_coords_tmp = pd.concat(
                [
                    df_assays_filt.iloc[[assay_num]],
                    df_coords.set_index(df_assays_filt.iloc[[assay_num]].index),
                ],
                axis=1,
            )
    
            # Append to global df_assay_coords
    
            df_assay_coords.append(df_assay_coords_tmp)

    # Merge all results

    df_assay_coords = pd.concat(df_assay_coords).reset_index(drop=True)

    # Some quick checks

    print("Fraction of assays assigned to grid points: ")
    df_assay_coords.loc[~df_assay_coords.x_inf.isnull()].shape[0] / df_assays.shape[0] * 100
    
    print("Fraction of assays assigned NULL: ")
    df_assay_coords.loc[df_assay_coords.x_inf.isnull()].shape[0] / df_assays.shape[0] * 100
    
    print("Holes in orignal assays not in df_traj: ")
    
    print(
        len(set(df_assays["HOLE-ID"].unique()).difference(set(df_traj["HOLE-ID"].unique())))
    )
    
    holes_missing = set(df_assays["HOLE-ID"].unique()).difference(
        set(df_traj["HOLE-ID"].unique())
    )

    # Convert SAMPLE NUM to a string column to save out to parquet

    df_assay_coords["SAMPLE NUM"] = df_assay_coords["SAMPLE NUM"].astype(str)

    return df_assay_coords