from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from matplotlib.patches import Rectangle, Polygon
import bin2cell as b2c
import anndata
# import spatialdata_io


@dataclass()
class VisiumHDB2C:
    """Class for handling Visium HD data using B2C"""

    out_10x_folder: Path
    out_10x_name: str = field(init=False)
    source_image_path: Path = field(init=False)

    tangram_out_folder: Path = field(init=False)
    tangram_out_csv_gz_f: Path = field(init=False)

    cell2location_folder: Path = field(init=False)
    cell2location_model: Path = field(init=False)
    cell2location_out: Path = field(init=False)

    cell_segmenation_folder: Path = field(init=False)

    def __post_init__(self):
        assert self.out_10x_folder.exists(), f"Folder {self.out_10x_folder} does not exist"
        self.out_10x_name = self.out_10x_folder.name

        print(f'The inferred 10x output name is {self.out_10x_name}')

        self.source_image_path = self.out_10x_folder / 'spatial/cytassist_image.tiff'
        self.source_image_path = self.out_10x_folder / 'spatial/tissue_hires_image.png'

        assert self.source_image_path.exists(), f"Source image {self.source_image_path} does not exist"

        # Tangram output settings
        self.tangram_out_folder = Path('./outputs') / self.out_10x_name / 'tangram'
        self.tangram_out_csv_gz_f = self.tangram_out_folder / f'{self.out_10x_name}_tangram.csv.gz'
        if not self.tangram_out_folder.exists():
            self.tangram_out_folder.mkdir(parents=True, )

        # Cell2location output settings
        self.cell2location_folder = Path('./outputs') / self.out_10x_name / 'cell2location'
        self.cell2location_model = self.cell2location_folder / 'model.pkl'

        self.cell2location_out = self.cell2location_folder / 'cell2location_out.csv.gz'
        if not self.cell2location_folder.exists():
            self.cell2location_folder.mkdir(parents=True, )

        # Cell segmentation settings
        self.cell_segmenation_folder = Path('./outputs') / self.out_10x_name / 'cell_segmentation'
        if not self.cell_segmenation_folder.exists():
            self.cell_segmenation_folder.mkdir(parents=True, )

        # self.adata_008um = self.load_data_b2c()

    def load_out_tissue(self):
        '''
        This is th output of the out tissue from Loupe Browser
        The file name should be out_tissue.csv and should be in the out_10x_folder
        This is manually generated
        '''
        out_tissue_f = self.out_10x_folder / 'out_tissue.csv'
        if out_tissue_f.exists():
            out_tissue = pd.read_csv(out_tissue_f, index_col=0)
            return out_tissue
        return None

    def load_main_tissue(self):

        main_tissue_f = self.out_10x_folder / 'main_tissue.csv'
        if main_tissue_f.exists():
            main_tissue = pd.read_csv(main_tissue_f, index_col=0)
            return main_tissue
        return None

    def get_um_choice(self, um: int):

        if um == 8:
            mm_choice = 'square_008um'
        elif um == 2:
            mm_choice = 'square_002um'
        elif um == 16:
            mm_choice = 'square_016um'
        else:
            raise ValueError(f"um {um} is not supported, only 2, 8, 16 are supported")
        return mm_choice

    def get_um_adata(self, um: int, filter_loupe_browser: bool = True):

        mm_choice = self.get_um_choice(um)
        print(f"Loading {mm_choice} data")

        out_10x_um = self.out_10x_folder / f'binned_outputs/{mm_choice}'
        spaceranger_image_path = self.out_10x_folder / f'binned_outputs/{mm_choice}/spatial'

        assert out_10x_um.exists(), f"Folder {out_10x_um} does not exist"
        assert spaceranger_image_path.exists(), f"Folder {spaceranger_image_path} does not exist"

        adata = b2c.read_visium(self.out_10x_folder / f'binned_outputs/{mm_choice}',
                                source_image_path=self.source_image_path,
                                spaceranger_image_path=spaceranger_image_path)

        adata.obs['spatial_sample_id'] = self.out_10x_name

        out_tissue = self.load_out_tissue()
        main_tissue = self.load_main_tissue()

        if um != 8:
            return adata

        if out_tissue is not None:
            adata.obs = pd.merge(adata.obs,
                                 out_tissue,
                                 left_index=True,
                                 right_index=True,
                                 how='left',
                                 )
            if filter_loupe_browser:
                adata = adata[adata.obs['out_tissue'] != 'out_tissue', :]
        if main_tissue is not None:
            adata.obs = pd.merge(adata.obs,
                                 main_tissue,
                                 left_index=True,
                                 right_index=True,
                                 how='left',
                                 )
            if filter_loupe_browser:
                adata = adata[adata.obs['main_tissue'] == 'main_tissue', :]

        return adata

    def load_high_image_path(self, um: int):
        mm_choice = self.get_um_choice(um)
        high_image_path = self.out_10x_folder / f'binned_outputs/{mm_choice}/spatial' / 'tissue_hires_image.png'
        assert high_image_path.exists(), f"Folder {high_image_path} does not exist"
        print(f"High image path: {str(high_image_path.absolute())}")
        return high_image_path

    def convert_spatialdata(self, ):

        # sdata = spatialdata_io.visium_hd(path=self.out_10x_folder,
        #                                  dataset_id=self.out_10x_name,
        #                                  load_all_images=True,
        #
        #                                  )
        #
        # return sdata
        pass


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



def convert_adata_to_parquet(
        adata: Anndata.adata,
        out_f:Path|str,
        obs_columns: list[str] = None,
)->None:
    expression_df = adata.to_df()
    if obs_columns:
        for obs_column in obs_columns:
            if obs_column in adata.obs.columns:
                expression_df[obs_column] = adata.obs[obs_column].values
            else:
                print(f"Warning: {obs_column} not found in adata.obs")

    expression_df.to_parquet(out_f, index=True)
    print(f"Data saved to {out_f} in Parquet format.")
    
    return expression_df
