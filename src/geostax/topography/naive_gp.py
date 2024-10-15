import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
import plotly.graph_objects as go


def gp_topography(df_traj: pd.DataFrame, df_collars: pd.DataFrame, res=100):
    
    # Get X and Y extents for reference (in UTM 19S)
    
    x_min = df_traj["x"].min()
    x_max = df_traj["x"].max()
    y_min = df_traj["y"].min()
    y_max = df_traj["y"].max()
    
    # Get observed x and y (this is what the GP will be trained to estimate
    
    x_obs = df_collars["Easting"].to_numpy()
    y_obs = df_collars["Northing"].to_numpy()
    z_obs = df_collars["Elevation"].to_numpy()[:, None]
    
    # Append x and y together
    
    x_y_obs = np.hstack([x_obs[:, None], y_obs[:, None]])
    
    # Re-scale x and y for better computational behavior
    
    xy_scaler = preprocessing.StandardScaler().fit(x_y_obs)
    z_scaler = preprocessing.StandardScaler().fit(z_obs)
    
    # Scale x_y_obs and z_obs for improved computational performance
    
    x_y_obs_scaled = xy_scaler.transform(x_y_obs)
    z_obs_scaled = z_scaler.transform(z_obs)
    
    # Construct an x and y grid to predict over
    
    x_grid = np.linspace(x_min, x_max, res)
    y_grid = np.linspace(y_min, y_max, res)
    x_grid_all, y_grid_all = np.meshgrid(x_grid, y_grid)
    
    # Create a matrix with the combinations
    
    x_grid_all = x_grid_all.flatten()
    y_grid_all = y_grid_all.flatten()
    xy_grid_all = np.concatenate(
        [x_grid_all[:, np.newaxis], y_grid_all[:, np.newaxis]], axis=1
    )
    
    # Scale xy grid all
    
    xy_grid_all_scaled = xy_scaler.transform(xy_grid_all)
    
    # GP training section
    
    kernel = RBF() + WhiteKernel()  # + DotProduct()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(
        x_y_obs_scaled, z_obs_scaled
    )
    
    # For actual vs. predicted checks, predict on x_y_obs_scaled
    
    z_obs_pred_scaled = gpr.predict(x_y_obs_scaled)
    
    # Predict on grid
    
    z_pred_scaled = gpr.predict(xy_grid_all_scaled)
    z_inferred = z_scaler.inverse_transform(z_pred_scaled[:, None])

    # Return objects

    grid_data = {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "x_grid_all": x_grid_all,
        "y_grid_all": y_grid_all,
        "xy_grid_all": xy_grid_all,
        "x_obs": x_obs,
        "y_obs": y_obs,
        "z_obs": z_obs,
    }


    return z_inferred, grid_data

def gp_topo_check(x_grid_all, y_grid_all, z_inferred, x_obs, y_obs, z_obs):

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_grid_all, y=y_grid_all, z=z_inferred.flatten(), mode="markers"
            )
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_obs,
            y=y_obs,
            z=z_obs.flatten(),
            mode="markers",
        )
    )
    fig.update_traces(marker={"size": 2})

    return fig