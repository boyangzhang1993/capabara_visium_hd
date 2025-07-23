from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .map_functions import transform_coordinates, reverse_transform_coordinates


class Transfer_Adata_To_Slide:


    @staticmethod
    def get_box_from_adata(adata):

        min_spatial_x_scaled = adata.obs['spatial_x_scaled'].min()
        min_spatial_y_scaled = adata.obs['spatial_y_scaled'].min()
        max_spatial_x_scaled = adata.obs['spatial_x_scaled'].max()
        max_spatial_y_scaled = adata.obs['spatial_y_scaled'].max()
        print \
            (f"X min Adata: {min_spatial_x_scaled}, X max Adata: {max_spatial_x_scaled},Y min Adata: {min_spatial_y_scaled},Y max Adata: {max_spatial_y_scaled}")
        return min_spatial_x_scaled, min_spatial_y_scaled, max_spatial_x_scaled, max_spatial_y_scaled

    @staticmethod
    def get_box_corrdinates_of_thumbnail_from_adata_scaled_corrindates(
            adata_subset,
            homography_matrix,
            slide_thumbnail,
            plot=False,
    ):


        min_spatial_x_scaled, min_spatial_y_scaled, max_spatial_x_scaled, max_spatial_y_scaled = Transfer_Adata_To_Slide.get_box_from_adata \
            (adata_subset)



        left_corner = np.array([[min_spatial_x_scaled, min_spatial_y_scaled]])
        thumbnail_coords_left_corner = transform_coordinates(left_corner,
                                                             homography_matrix)

        right_bottom = np.array([[max_spatial_x_scaled, max_spatial_y_scaled]])
        thumbnail_coords_right_bottom = transform_coordinates(right_bottom,
                                                              homography_matrix)




        x_min_thumbnial = thumbnail_coords_left_corner[0, 0]
        y_min_thumbnial = thumbnail_coords_left_corner[0, 1]

        x_max_thumbnial = thumbnail_coords_right_bottom[0, 0]
        y_max_thumbnial = thumbnail_coords_right_bottom[0, 1]


        if x_max_thumbnial < x_min_thumbnial:
            x_min_thumbnial, x_max_thumbnial = x_max_thumbnial, x_min_thumbnial
        if y_max_thumbnial < y_min_thumbnial:
            y_min_thumbnial, y_max_thumbnial = y_max_thumbnial, y_min_thumbnial
        thumbnial_cropped = slide_thumbnail.crop((
            x_min_thumbnial,
            y_min_thumbnial,
            x_max_thumbnial,
            y_max_thumbnial,

        ))
        if plot:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111)
            ax.imshow(thumbnial_cropped)
            ax.set_title('Thumbnail')
            plt.show()

        return x_min_thumbnial, y_min_thumbnial, x_max_thumbnial, y_max_thumbnial

    @staticmethod
    def get_high_res_cropped_from_thumb_transfer(slide, slide_thumbnail, x_min_thumb, y_min_thumb, x_max_thumb, y_max_thumb,
                                                 plot=False):


        level_max = 0
        level_max_dim = slide.level_dimensions[level_max]

        scale_x = level_max_dim[0] / slide_thumbnail.width
        scale_y = level_max_dim[1] / slide_thumbnail.height

        scaled_level_thumbnail_x_max = x_max_thumb * scale_x
        scaled_level_thumbnail_y_min = y_min_thumb * scale_y

        scaled_level_thumbnail_x_min = x_min_thumb * scale_x
        scaled_level_thumbnail_y_max = y_max_thumb * scale_y



        if scaled_level_thumbnail_x_max < scaled_level_thumbnail_x_min:
            scaled_level_thumbnail_x_min, scaled_level_thumbnail_x_max = scaled_level_thumbnail_x_max, scaled_level_thumbnail_x_min
        if scaled_level_thumbnail_y_max < scaled_level_thumbnail_y_min:
            scaled_level_thumbnail_y_min, scaled_level_thumbnail_y_max = scaled_level_thumbnail_y_max, scaled_level_thumbnail_y_min


        scaled_size_x = scaled_level_thumbnail_x_max - scaled_level_thumbnail_x_min
        scaled_size_y = scaled_level_thumbnail_y_max - scaled_level_thumbnail_y_min

        print \
            (f"X min: {scaled_level_thumbnail_x_min}, X max: {scaled_level_thumbnail_x_max},Y min: {scaled_level_thumbnail_y_min},Y max: {scaled_level_thumbnail_y_max},Size X: {scaled_size_x},Size Y: {scaled_size_y}")


        buffer_size = 0
        slide_high_res_subset =  slide.read_region(
            (int(scaled_level_thumbnail_x_min),
             int(scaled_level_thumbnail_y_min)),
            level_max,
            (
                int(scaled_size_x +buffer_size),
                int(scaled_size_y +buffer_size))

        )
        if plot:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111)
            ax.imshow(slide_high_res_subset)
            ax.set_title('High Res')
            plt.show()

        return slide_high_res_subset


    @staticmethod
    def get_high_res_slide_from_slide_croosponding_to_adata(
            adata,
            homography_matrix,
            slide,
            slide_thumbnail,
            plot_thumb = False,
            plot_high_res = False

    ):

        min_thumb_x, min_thumb_y, max_thumb_x, max_thumb_y = Transfer_Adata_To_Slide.get_box_corrdinates_of_thumbnail_from_adata_scaled_corrindates(
            adata,
            homography_matrix,
            slide_thumbnail,
            plot=plot_thumb,
        )

        return Transfer_Adata_To_Slide.get_high_res_cropped_from_thumb_transfer(slide, slide_thumbnail, min_thumb_x, min_thumb_y, max_thumb_x, max_thumb_y, plot=plot_high_res)


        # pass
    @staticmethod
    def plot_adata_continuous_variable_with_high_res_slide(
            adata, ploting_variable,
            homography_matrix, slide, slide_thumbnail,
            low_quality_threshold=1, high_quality_threshold=10,
            alpha=0.5, img_alpha=0.5, cmap='seismic', s=1,
            vmin=0, vmax=10,
    ):

        min_spatial_x_scaled, min_spatial_y_scaled, max_spatial_x_scaled, max_spatial_y_scaled = Transfer_Adata_To_Slide.get_box_from_adata \
            (adata)
        slide_subset = Transfer_Adata_To_Slide.get_high_res_slide_from_slide_croosponding_to_adata(adata, homography_matrix, slide, slide_thumbnail, plot_thumb = False, plot_high_res = False)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(221)
        ax.imshow(slide_subset)
        ax.set_title('High Res Slide')
        ax2 = fig.add_subplot(222)
        ax2.imshow(slide_subset,
                   extent=(
                       min_spatial_x_scaled,
                       max_spatial_x_scaled,
                       max_spatial_y_scaled,
                       min_spatial_y_scaled
                   ),
                   alpha=img_alpha,
                   )
        ax2.scatter(adata.obs['spatial_x_scaled'],
                    adata.obs['spatial_y_scaled'],
                    c=adata.obs[ploting_variable],
                    cmap='seismic',
                    alpha=alpha,
                    s=s,
                    vmin=vmin,
                    vmax=vmax,
                    )
        df_zeor_counts = adata.obs.loc[adata.obs[ploting_variable] <= low_quality_threshold]
        ax3 = fig.add_subplot(223)
        ax3.imshow(slide_subset,
                   extent=(
                       min_spatial_x_scaled,
                       max_spatial_x_scaled,
                       max_spatial_y_scaled,
                       min_spatial_y_scaled
                   )

                   )
        ax3.scatter(
            df_zeor_counts['spatial_x_scaled'],
            df_zeor_counts['spatial_y_scaled'],
            c='b',
            # alpha=0.8,
            s=1,
        )
        ax3.set_title(f'Data with {ploting_variable} <= {low_quality_threshold}')



        df_high_counts = adata.obs.loc[adata.obs[ploting_variable] >= high_quality_threshold]
        ax4 = fig.add_subplot(224)
        ax4.imshow(slide_subset,
                   extent=(
                       min_spatial_x_scaled,
                       max_spatial_x_scaled,
                       max_spatial_y_scaled,
                       min_spatial_y_scaled
                   )

                   )
        ax4.scatter(
            df_high_counts['spatial_x_scaled'],
            df_high_counts['spatial_y_scaled'],
            c='r',
            alpha=0.5,
            s=2,
        )
        ax4.set_title(f'Data with {ploting_variable} >= {high_quality_threshold}')
        plt.show()

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)

        ax.imshow(slide_subset,
                  extent=(
                      min_spatial_x_scaled,
                      max_spatial_x_scaled,
                      max_spatial_y_scaled,
                      min_spatial_y_scaled
                  ),
                  alpha=0,
                  )
        ax.scatter(adata.obs['spatial_x_scaled'],
                   adata.obs['spatial_y_scaled'],
                   c=adata.obs[ploting_variable],
                   cmap=cmap,
                   alpha=1,
                   s=1,
                   vmin=vmin,
                   vmax=vmax,
                   )

        ax2 = fig.add_subplot(122)
        ax2.imshow(slide_subset,
                   extent=(
                       min_spatial_x_scaled,
                       max_spatial_x_scaled,
                       max_spatial_y_scaled,
                       min_spatial_y_scaled
                   )
                   )
        plt.show()

        return None        # pass
    @staticmethod
    def plot_adata_categorical_variable_with_high_res_slide(
            adata, plotting_variable,
            homography_matrix, slide, slide_thumbnail,
            alpha=0.5, img_alpha=0.5, s=1,
            filter_none=True
    ):
        """
        Plot categorical variables on high resolution slide images.

        Parameters:
        -----------
        adata : AnnData
            Annotated data matrix
        plotting_variable : str
            Name of the categorical variable to plot
        homography_matrix : array
            Transformation matrix
        slide : object
            Slide object containing high resolution image
        slide_thumbnail : array
            Thumbnail of the slide
        alpha : float
            Transparency of scatter points
        img_alpha : float
            Transparency of background image
        s : float
            Size of scatter points
        filter_none : bool
            Whether to filter None/NA values
        """

        # Define fixed colors
        colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Different colors for different cells
        # from scipy.stats import gaussian_kde

        # Filter data if requested
        if filter_none:
            valid_points = (
                    (adata.obs[plotting_variable] != 'None') &
                    (adata.obs[plotting_variable].notna())
            )
            adata = adata[valid_points]

        # Get spatial boundaries and high res slide
        min_spatial_x_scaled, min_spatial_y_scaled, max_spatial_x_scaled, max_spatial_y_scaled = Transfer_Adata_To_Slide.get_box_from_adata \
            (adata)
        slide_subset = Transfer_Adata_To_Slide.get_high_res_slide_from_slide_croosponding_to_adata(
            adata, homography_matrix, slide, slide_thumbnail, plot_thumb=False, plot_high_res=False
        )

        # Get unique categories
        categories = adata.obs[plotting_variable].unique()

        # First figure with four subplots
        fig = plt.figure(figsize=(15, 15))

        # High res slide
        ax = fig.add_subplot(221)
        ax.imshow(slide_subset)
        ax.set_title('High Res Slide')

        # Categorical scatter plot
        ax2 = fig.add_subplot(222)
        ax2.imshow(slide_subset,
                   extent=(min_spatial_x_scaled, max_spatial_x_scaled,
                           max_spatial_y_scaled, min_spatial_y_scaled),
                   alpha=img_alpha)

        # Plot each category with different color
        for idx, category in enumerate(categories):
            mask = adata.obs[plotting_variable] == category
            ax2.scatter(
                adata.obs.loc[mask, 'spatial_x_scaled'],
                adata.obs.loc[mask, 'spatial_y_scaled'],
                c=colors[idx % len(colors)],
                label=category,
                alpha=alpha,
                s=s
            )
        # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=plotting_variable)
        ax2.set_title('Category Distribution')


        plt.tight_layout()
        plt.show()

        return None