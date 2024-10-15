from functools import partial

import pandas as pd
import numpy as np

def r1_z_coords_fixed_xy(z_top, z_min, dz):
    """
    For a given x and y coordinate, use the z_top and z_min to get z points
    """
    z_coords = np.arange(z_min + dz, z_top, dz)
    return z_coords

def make_cuboidal_grid(
        df_traj: pd.DataFrame, 
        z_inferred: np.array, 
        x_grid: np.array, 
        y_grid: np.array,
        res: int = 50,
):

    # Region 2: (cuboidal section)
    
    # Get lowest elevation from topography data
    # Underneath this point, we assume the relevant region is a cuboid
    
    z_lowest_collar = np.min(z_inferred)  # TODO: should come from topography
    
    # Get lowest elevation from drill hole data
    
    Z_MIN = df_traj["z"].min() - 5  # 5 meter reduction
    z_grid_r2 = np.linspace(Z_MIN, z_lowest_collar, res)
    
    # Implied dZ
    
    dz_implied = np.diff(z_grid_r2)[0]

    # Assertion checks

    assert z_grid_r2[0] == Z_MIN
    assert z_grid_r2[-1] == z_lowest_collar

    # The cross-product of the following creates the grid indices in the region r2 (X, Y, Z)
    # i.e.: x_grid x y_grid x z_grid
    
    x_grid_r2, y_grid_r2, z_grid_r2_all = np.meshgrid(x_grid, y_grid, z_grid_r2)
    
    # Create a matrix with the combinations
    
    x_grid_r2 = x_grid_r2.flatten()
    y_grid_r2 = y_grid_r2.flatten()
    z_grid_r2_all = z_grid_r2_all.flatten()
    xyz_grid_r2 = np.concatenate(
        [x_grid_r2[:, np.newaxis], y_grid_r2[:, np.newaxis], z_grid_r2_all[:, np.newaxis]],
        axis=1,
    )

    print(xyz_grid_r2.shape)

    return xyz_grid_r2, dz_implied, z_lowest_collar

def make_topographic_grid(xy_grid_all, dz_implied, z_lowest_collar, z_inferred):

    # z_lowest_collar is subsumed inside of xyz_grid_r2
    # I will use the dz_implied of xyz_grid_r2 (Region 2 grid) to create a set of points in the R1 region
    # dz stays constant, but the number of cuboids will change

    # Region 1: (topographic section)
    # Get the coordinates of the topography
    
    top_coords = np.hstack([xy_grid_all, z_inferred])

    # Fix values of the lowest z and dz_implied

    r1_z_coords_fixed_xy_p = partial(
        r1_z_coords_fixed_xy, z_min=z_lowest_collar, dz=dz_implied
    )

    # Use a for loop for now. Make this faster maybe?

    xyz_grid_r1 = list()
    
    for row in top_coords:
    
        # Parse out the variables
    
        x = row[0]
        y = row[1]
        z_max = row[2]
    
        # Obtain vector of all z's
    
        z_coords = r1_z_coords_fixed_xy_p(z_top=z_max)
    
        # Mesh-grid
    
        x_tmp, y_tmp, z_tmp = np.meshgrid(x, y, z_coords)
        xyz_grid_tmp = np.concatenate(
            [
                x_tmp.flatten()[:, np.newaxis],
                y_tmp.flatten()[:, np.newaxis],
                z_tmp.flatten()[:, np.newaxis],
            ],
            axis=1,
        )
        xyz_grid_r1.append(xyz_grid_tmp)
    
    # vstack and print out shape
    
    xyz_grid_r1 = np.vstack(xyz_grid_r1)
    print(xyz_grid_r1.shape)

    return xyz_grid_r1