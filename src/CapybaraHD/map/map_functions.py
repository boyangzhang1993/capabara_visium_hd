import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



def transform_coordinates(coords, homography_matrix):
    """
    Transform coordinates using homography matrix.

    Parameters:
    coords: numpy array of shape (N, 2) containing x,y coordinates
    homography_matrix: 3x3 homography matrix

    Returns:
    transformed_coords: numpy array of shape (N, 2) with transformed coordinates
    """
    # Ensure coordinates are in the right format
    coords = np.array(coords)
    if coords.ndim == 1:
        coords = coords.reshape(1, 2)

    # Add homogeneous coordinate (1) to make it (x,y,1)
    homogeneous_coords = np.column_stack([coords, np.ones(len(coords))])

    # Get inverse of homography matrix (since we want to go in opposite direction)
    h_inverse = np.linalg.inv(homography_matrix)

    # Transform coordinates
    transformed_homogeneous = np.dot(h_inverse, homogeneous_coords.T).T

    # Convert back from homogeneous coordinates to regular coordinates
    transformed_coords = transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2:]

    return transformed_coords


def reverse_transform_coordinates(coords, homography_matrix):
    """
    Transform coordinates using homography matrix in the forward direction.

    Parameters:
    coords: numpy array of shape (N, 2) containing x,y coordinates
    homography_matrix: 3x3 homography matrix

    Returns:
    transformed_coords: numpy array of shape (N, 2) with transformed coordinates
    """
    # Ensure coordinates are in the right format
    coords = np.array(coords)
    if coords.ndim == 1:
        coords = coords.reshape(1, 2)

    # Add homogeneous coordinate (1) to make it (x,y,1)
    homogeneous_coords = np.column_stack([coords, np.ones(len(coords))])

    # Transform coordinates using homography matrix directly
    transformed_homogeneous = np.dot(homography_matrix, homogeneous_coords.T).T

    # Convert back from homogeneous coordinates to regular coordinates
    transformed_coords = transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2:]

    return transformed_coords


def plot_probe_continuous_variable_mapped_on_adata_img(adata_image,
                                                       adata,
                                                       ploting_variable='cell_id',
                                                       filter_none=True,
                                                       alpha=0.2,
                                                       vmin=None,
                                                       vmax=None,
                                                       ):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121)

    min_spatial_x_scaled = adata.obs['spatial_x_scaled'].min()
    min_spatial_y_scaled = adata.obs['spatial_y_scaled'].min()
    max_spatial_x_scaled = adata.obs['spatial_x_scaled'].max()
    max_spatial_y_scaled = adata.obs['spatial_y_scaled'].max()

    cropped_region_adata_image = adata_image.crop((min_spatial_x_scaled,
                                                   min_spatial_y_scaled,
                                                   max_spatial_x_scaled,
                                                   max_spatial_y_scaled,
                                                   ))

    ax.imshow(cropped_region_adata_image,
              extent=(min_spatial_x_scaled,
                      max_spatial_x_scaled,
                      max_spatial_y_scaled,
                      min_spatial_y_scaled,
                      ),
              alpha=1)

    if filter_none:
        valid_points = (
                (adata.obs[ploting_variable] != 'None') &
                (adata.obs[ploting_variable].notna())
        )
        adata = adata[valid_points]

    # Create scatter plot with continuous values

    vmin = adata.obs[ploting_variable].min() if vmin is None else vmin
    vmax = adata.obs[ploting_variable].max() if vmax is None else vmax
    scatter = ax.scatter(
        adata.obs['spatial_x_scaled'],
        adata.obs['spatial_y_scaled'],
        c=adata.obs[ploting_variable],  # Remove .astype('category').cat.codes
        cmap='seismic',
        alpha=alpha,
        # Set vmin and vmax based on data range if needed
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f'{ploting_variable} of Adata')
    # Add colorbar for continuous values
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(ploting_variable, rotation=270, labelpad=15)

    # plt.show()

    df_zeor_counts = adata.obs.loc[adata.obs[ploting_variable] <= 1]

    # fig = plt.figure(figsize=(10, 10))
    ax2 = fig.add_subplot(122)
    ax2.imshow(cropped_region_adata_image,
               extent=(min_spatial_x_scaled,
                       max_spatial_x_scaled,
                       max_spatial_y_scaled,
                       min_spatial_y_scaled,
                       ),
               alpha=1)

    ax2.scatter(
        df_zeor_counts['spatial_x_scaled'],
        df_zeor_counts['spatial_y_scaled'],
        c='b',
        alpha=0.1,
    )

    ax.scatter(
        df_zeor_counts['spatial_x_scaled'],
        df_zeor_counts['spatial_y_scaled'],
        c='b',
        alpha=0.5,
    )

    plt.tight_layout()
    plt.show()

    # plot a rectangle for the cropped region

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(adata_image)
    # ax.scatter(
    #     adata.obs['spatial_x_scaled'],
    #     adata.obs['spatial_y_scaled'],
    #     c='r',
    #     # alpha=0.2,

    # )

    # ax.scatter(adata.obs['spatial_x_scaled'],
    #        adata.obs['spatial_y_scaled'], )

    rect = Rectangle((min_spatial_x_scaled, min_spatial_y_scaled),
                     max_spatial_x_scaled - min_spatial_x_scaled,
                     max_spatial_y_scaled - min_spatial_y_scaled,
                     linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


def plot_probe_categorical_variable_mapped_on_adata_img(
        adata_image=None,
        adata=None,
        plotting_variable='cell_type',
        filter_none=True,
        figsize=(20, 10),
        alpha=0.2,
        show_overview=True
):
    """
    Plot categorical variables on spatial data with an underlying image.

    Parameters:
    -----------
    adata_image : PIL.Image
        The background image to plot on
    adata : AnnData
        Annotated data matrix with spatial coordinates
    plotting_variable : str
        Name of the categorical variable to plot
    filter_none : bool
        Whether to filter out None/NA values
    figsize : tuple
        Figure size for the plot
    alpha : float
        Transparency of the scatter points
    show_overview : bool
        Whether to show the overview plot with region rectangle
    """
    # Create main figure with two subplots
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(121)

    # Get spatial boundaries
    min_spatial_x_scaled = adata.obs['spatial_x_scaled'].min()
    min_spatial_y_scaled = adata.obs['spatial_y_scaled'].min()
    max_spatial_x_scaled = adata.obs['spatial_x_scaled'].max()
    max_spatial_y_scaled = adata.obs['spatial_y_scaled'].max()

    # Crop image to region of interest
    cropped_region_adata_image = adata_image.crop((
        min_spatial_x_scaled,
        min_spatial_y_scaled,
        max_spatial_x_scaled,
        max_spatial_y_scaled,
    ))

    # Show cropped image
    ax.imshow(cropped_region_adata_image,
              extent=(min_spatial_x_scaled,
                      max_spatial_x_scaled,
                      max_spatial_y_scaled,
                      min_spatial_y_scaled,
                      ),
              alpha=1)

    # Filter data if requested
    if filter_none:
        valid_points = (
                (adata.obs[plotting_variable] != 'None') &
                (adata.obs[plotting_variable].notna())
        )
        adata = adata[valid_points]

    # Define fixed color list
    colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Different colors for different cells

    # Get unique categories
    categories = adata.obs[plotting_variable].unique()
    n_categories = len(categories)

    # Create scatter plot with categorical values
    for idx, category in enumerate(categories):
        mask = adata.obs[plotting_variable] == category
        ax.scatter(
            adata.obs.loc[mask, 'spatial_x_scaled'],
            adata.obs.loc[mask, 'spatial_y_scaled'],
            c=[colors[idx % len(colors)]],  # Cycle through colors if more categories than colors
            label=category,
            alpha=alpha
        )

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=plotting_variable)

    # Create distribution plot in second subplot
    ax2 = fig.add_subplot(122)

    # Calculate category counts
    category_counts = adata.obs[plotting_variable].value_counts()

    # Create bar plot
    bars = ax2.bar(range(len(category_counts)), category_counts.values)
    ax2.set_xticks(range(len(category_counts)))
    ax2.set_xticklabels(category_counts.index, rotation=45, ha='right')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Distribution of {plotting_variable}')

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Show overview plot if requested
    if show_overview:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(adata_image)

        # Add rectangle for cropped region
        rect = Rectangle((min_spatial_x_scaled, min_spatial_y_scaled),
                         max_spatial_x_scaled - min_spatial_x_scaled,
                         max_spatial_y_scaled - min_spatial_y_scaled,
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_title('Overview with Selected Region')
        plt.show()