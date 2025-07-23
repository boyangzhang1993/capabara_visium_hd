from __future__ import annotations
from pathlib import Path
import openslide
import numpy as np
from PIL import Image
import shapely.geometry
import matplotlib.pyplot as plt
import pickle
from .map_functions import transform_coordinates, reverse_transform_coordinates, plot_probe_continuous_variable_mapped_on_adata_img
import cv2
from anndata import AnnData
from skimage.color import rgb2gray


class PreprocessAdataImage:

    def __init__(self,
                 adata: AnnData,
                 wsi_slide: str | Path,
                 *,
                 plot_check_wsi_adata=True,
                 top_match=0.1,
                 plot_check_homography=True, ):

        self.adata = adata
        self.wsi_slide = wsi_slide

        self.get_adata_image()
        self.get_wsi_slide(get_thumbnail=True)

        # Generate Homography
        self.check_adata_matching_wsi(plot_check_wsi_adata)
        self.match_adata_wsi_homography(top_match=top_match, plot=plot_check_homography)

    def get_adata_image(self):
        """
        Get the adata image from the adata object.
        """
        adata = self.adata
        img_key = [k for k in adata.uns['spatial'].keys()][0]
        adata_image_from_h5ad = adata.uns['spatial'][img_key]['images']['hires']

        # Normalize to 0-255 range
        normalized_image = ((adata_image_from_h5ad - adata_image_from_h5ad.min()) * 255.0 /
                            (adata_image_from_h5ad.max() - adata_image_from_h5ad.min()))

        # Convert to uint8
        uint8_image = normalized_image.astype(np.uint8)

        # If needed, reshape to remove singleton dimensions
        if uint8_image.shape[0] == 1 and uint8_image.shape[1] == 1:
            uint8_image = uint8_image.squeeze()

        # Now convert to PIL Image
        adata_image = Image.fromarray(uint8_image)

        self.adata_image = adata_image

    def get_wsi_slide(self, get_thumbnail=True):
        """
        Get the WSI slide.
        """
        slide_f = self.wsi_slide
        self.slide = openslide.OpenSlide(slide_f)
        if get_thumbnail:
            self.slide_thumbnail = self.slide.get_thumbnail(
                (self.adata_image.width, self.adata_image.height)
            )

    def check_adata_matching_wsi(self, plot=True):
        """
        Check if the adata image matches the WSI slide.
        """
        adata = self.adata
        slide = self.wsi_slide

        adata_image = self.adata_image
        slide_thumbnail = self.slide_thumbnail

        adata_image_array = np.array(adata_image)
        adata_image_array_gray = rgb2gray(adata_image)
        thumbnail_array_gray = rgb2gray(slide_thumbnail)

        # Our images are already in grayscale, but let's ensure they're in the right format for OpenCV
        self.adata_gray = adata_gray = (adata_image_array_gray * 255).astype(np.uint8)
        self.thumbnail_gray = thumbnail_gray = (thumbnail_array_gray * 255).astype(np.uint8)

        # Detect ORB features and compute descriptors
        MAX_NUM_FEATURES = 2000  # Increased for more features
        orb = cv2.ORB_create(MAX_NUM_FEATURES)
        keypoints_adata, descriptors_adata = orb.detectAndCompute(adata_gray, None)
        keypoints_wsi, descriptors_wsi = orb.detectAndCompute(thumbnail_gray, None)
        self.adata_keypoints_descriptors = (keypoints_adata, descriptors_adata)
        self.wsi_keypoints_descriptors = (keypoints_wsi, descriptors_wsi)
        # Draw keypoints on images
        adata_display = cv2.drawKeypoints(adata_gray, keypoints_adata,
                                          outImage=np.array([]),
                                          color=(255, 0, 0),
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        thumbnail_display = cv2.drawKeypoints(thumbnail_gray, keypoints_wsi,
                                              outImage=np.array([]),
                                              color=(255, 0, 0),
                                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if not plot:
            return None
        # Display Images with keypoints
        plt.figure(figsize=[20, 10])
        plt.subplot(2, 2, 1)
        plt.axis("off")
        plt.imshow(adata_display, cmap='gray')
        plt.title("10X Image with Keypoints")

        plt.subplot(2, 2, 2)
        plt.axis("off")
        plt.imshow(thumbnail_display, cmap='gray')
        plt.title("Thumbnail (Whole Slide) with Keypoints")

        plt.subplot(2, 2, 3)
        plt.axis("off")
        plt.imshow(adata_image)
        plt.title("10X Image")

        plt.subplot(2, 2, 4)
        plt.axis("off")
        plt.imshow(slide_thumbnail)
        plt.title("Thumbnail (Whole Slide) Image")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("10X Image and WSI Image (make sure the four image are very similar)")
        plt.show()

    def match_adata_wsi_homography(self, *, top_match: float = 0.1, plot=True):

        # Match features
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = list(matcher.match(self.adata_keypoints_descriptors[1],
                                     self.wsi_keypoints_descriptors[1],
                                     None))

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # matches

        # Remove not so good matches - keep top
        numGoodMatches = int(len(matches) * top_match)
        matches = matches[:numGoodMatches]

        # Draw top matches

        im_matches = cv2.drawMatches(
            self.adata_gray,
            self.adata_keypoints_descriptors[0],
            self.thumbnail_gray,
            self.wsi_keypoints_descriptors[0],
            matches, None,
            flags=cv2.DrawMatchesFlags_DEFAULT,
            # matchColor=(0, 0, 255),
        )

        # Display matches
        plt.figure(figsize=[40, 10])
        plt.imshow(im_matches, cmap='gray', alpha=0.98)
        plt.axis("off")
        plt.title("Feature Matches Between Images")
        plt.show()
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = self.adata_keypoints_descriptors[0][match.queryIdx].pt
            points2[i, :] = self.wsi_keypoints_descriptors[0][match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        self.homography_matrix = h
        print("Homography matrix:")
        print(h)
        # Warp thumbnail to align with high-res image
        height, width = self.adata_gray.shape
        thumbnail_aligned = cv2.warpPerspective(self.thumbnail_gray, h, (width, height))

        # Display results
        if not plot:
            return None
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(self.adata_gray, cmap='gray')
        ax[0].set_title('High-res Image')
        ax[1].imshow(self.thumbnail_gray, cmap='gray')
        ax[1].set_title('Histology Image')
        ax[2].imshow(thumbnail_aligned, cmap='gray')
        ax[2].set_title('Aligned Histology')
        plt.show()

        
class map_cell_boundary_from_ultra_high_res_histology_to_adata():

    def __init__(self,
                 analyze_folder: str | Path,
                 slide: openslide.OpenSlide,
                 ) -> None:

        self.set_up_folders(Path(analyze_folder),
                            slide

                            )
        self.slide = slide

    def one_example_tile_boundary_with_tile(self, idx=9):
        boundary_files = [f for f in self.boundary_predictions_folder.iterdir() if f.is_file() and 'tile' in f.stem]
        # random one example
        boundary_f = boundary_files[idx]
        tile_file = str(boundary_f.stem).removesuffix('_coords') + '.png'
        tile_f = self.tiles_folder / tile_file
        assert tile_f.exists(), f"{tile_f} does not exist"

        return boundary_f, tile_f

    def get_all_boundary_tile_files(self):
        boundary_files = []
        tile_files = []
        for f in self.boundary_predictions_folder.iterdir():
            # print(f)
            if f.is_file() and 'tile' in f.stem:
                # print(f)
                boundary_files.append(f)
            tile_file = str(f.stem).removesuffix('_coords') + '.png'
            tile_f = self.tiles_folder / tile_file
            assert tile_f.exists(), f"{tile_f} does not exist"
            tile_files.append(tile_f)

        return boundary_files, tile_files

    def set_up_folders(self, analyze_folder, slide):

        self.boundary_predictions_folder = analyze_folder / 'nucleus_predictions'
        self.tiles_folder = analyze_folder / f'{analyze_folder.stem}_tiles'

        assert self.boundary_predictions_folder.exists(), f"{self.boundary_predictions_folder} does not exist"
        assert self.tiles_folder.exists(), f"{self.tiles_folder} does not exist"

        min_x_tile, min_y_tile, max_x_tile, max_y_tile = map_cell_boundary_from_ultra_high_res_histology_to_adata.get_res_min_max_from_annotation_folder_files(
            self.tiles_folder)
        min_x_boundary, min_y_boundary, max_x_boundary, max_y_boundary = map_cell_boundary_from_ultra_high_res_histology_to_adata.get_res_min_max_from_annotation_folder_files(
            self.boundary_predictions_folder)

        assert min_x_tile == min_x_boundary, f"min_x_tile: {min_x_tile} != min_x_boundary: {min_x_boundary}"
        assert min_y_tile == min_y_boundary, f"min_y_tile: {min_y_tile} != min_y_boundary: {min_y_boundary}"
        assert max_x_tile == max_x_boundary, f"max_x_tile: {max_x_tile} != max_x_boundary: {max_x_boundary}"
        assert max_y_tile == max_y_boundary, f"max_y_tile: {max_y_tile} != max_y_boundary: {max_y_boundary}"

        slide_demension_max = slide.level_dimensions[0]
        slide_demension_max_x = slide_demension_max[0]
        slide_demension_max_y = slide_demension_max[1]

        assert slide_demension_max_x >= max_x_tile
        assert slide_demension_max_y >= max_y_tile

        return None

    @staticmethod
    def get_res_min_max_from_annotation_folder_files(folder: Path | str,
                                                     x_index=2,
                                                     y_index=1,
                                                     ):

        folder = Path(folder)
        x = [f.stem.split('_')[x_index] for f in folder.iterdir() if f.is_file() and 'tile' in f.stem]
        y = [f.stem.split('_')[y_index] for f in folder.iterdir() if f.is_file() and 'tile' in f.stem]
        x = [int(x) for x in x]
        y = [int(y) for y in y]

        max_x = max(x)
        min_x = min(x)
        max_y = max(y)
        min_y = min(y)

        print(f"X min: {min_x}, X max: {max_x},Y min: {min_y},Y max: {max_y}")
        return min_x, min_y, max_x, max_y
        # pass

    @staticmethod
    def transfer_high_res_boundary_to_adata_obs_cell_bounds(

            boundary_f,
            tile_f,
            slide,
            slide_thumbnail,
            adata,
            adata_image,

            homoglaphy_matrix,

            x_index=2,
            y_index=1,
            new_obs_cell_bounds_variable_name='cell_bounds',
            default_cell_bounds=None,
            plot_comprehensively=True,
            save_comprehensively: str | Path = None,
            verbose=False,

    ):
        if verbose:
            print(f'File name: {boundary_f.stem}')

        assert new_obs_cell_bounds_variable_name in adata.obs.columns, f"{new_obs_cell_bounds_variable_name} not in adata.obs.columns"

        boundary_x = int(boundary_f.stem.split('_')[x_index])
        boundary_y = int(boundary_f.stem.split('_')[y_index])

        tile_x = int(tile_f.stem.split('_')[x_index])
        tile_y = int(tile_f.stem.split('_')[y_index])

        assert boundary_x == tile_x, f"boundary_x: {boundary_x} != tile_x: {tile_x}"
        assert boundary_y == tile_y, f"boundary_y: {boundary_y} != tile_y: {tile_y}"
        tile_img = Image.open(tile_f)

        detection_details = np.load(boundary_f, allow_pickle=True)
        detection_details = detection_details.item()
        if verbose:
            print(detection_details.keys())
        detection_coords = detection_details['coord']

        level_max = 0
        level_max_dim = slide.level_dimensions[level_max]

        scale_x = slide_thumbnail.width / level_max_dim[0]
        scale_y = slide_thumbnail.height / level_max_dim[1]

        x_thumbnial = boundary_x * scale_x
        y_thumbnial = boundary_y * scale_y
        width_thumbnail = tile_img.width * scale_x
        height_thumbnail = tile_img.height * scale_y

        left_bottom = np.array([x_thumbnial, y_thumbnial])
        left_top = np.array([x_thumbnial, y_thumbnial + height_thumbnail])
        right_top = np.array([x_thumbnial + width_thumbnail, y_thumbnial + height_thumbnail])
        right_bottom = np.array([x_thumbnial + width_thumbnail, y_thumbnial])

        left_bottom_adata = reverse_transform_coordinates(left_bottom, homoglaphy_matrix)[0]
        left_top_adata = reverse_transform_coordinates(left_top, homoglaphy_matrix)[0]
        right_top_adata = reverse_transform_coordinates(right_top, homoglaphy_matrix)[0]
        right_bottom_adata = reverse_transform_coordinates(right_bottom, homoglaphy_matrix)[0]

        if verbose:
            print(f'left bottom adata: {left_bottom_adata}')
            print(f'left top adata: {left_top_adata}')
            print(f'right top adata: {right_top_adata}')
            print(f'right bottom adata: {right_bottom_adata}')

        left_x_adata = left_bottom_adata[0]
        left_y_adata = left_bottom_adata[1]

        right_x_adata = right_top_adata[0]
        right_y_adata = right_top_adata[1]

        if verbose:
            print(f'left x adata: {left_x_adata}')
            print(f'left y adata: {left_y_adata}')
            print(f'right x adata: {right_x_adata}')
            print(f'right y adata: {right_y_adata}')

        adata_subset = adata[

                       (adata.obs['spatial_x_scaled'] > left_x_adata) &
                       (adata.obs['spatial_x_scaled'] < right_x_adata) &
                       (adata.obs['spatial_y_scaled'] > left_y_adata) &
                       (adata.obs['spatial_y_scaled'] < right_y_adata)
        ,
                       :

                       ]
        if adata_subset.shape[0] == 0:
            print(f'Skipping {boundary_f.stem} as no cells found in the boundary')
            return None

        # ax = fig.add_subplot(221)

        min_spatial_x_scaled = adata_subset.obs['spatial_x_scaled'].min()
        min_spatial_y_scaled = adata_subset.obs['spatial_y_scaled'].min()
        max_spatial_x_scaled = adata_subset.obs['spatial_x_scaled'].max()
        max_spatial_y_scaled = adata_subset.obs['spatial_y_scaled'].max()

        from copy import deepcopy
        detection_coords_copy = deepcopy(detection_coords)
        transformed_coords = []
        for coord in detection_coords_copy:
            coord[:, 0] += boundary_x
            coord[:, 1] += boundary_y

            coord[:, 0] *= scale_x
            coord[:, 1] *= scale_y

            transformed_coord = reverse_transform_coordinates(coord, homoglaphy_matrix)
            transformed_coords.append(transformed_coord)

        # Remap the boundary to adata obs
        transfered_coords = []
        for transformed_coord in transformed_coords:
            if len(transformed_coord) < 4:
                continue
            transfered_coords.append(transformed_coord)
            polygon_cell = shapely.geometry.Polygon(transformed_coord)
            polygon_cell_centroid = polygon_cell.centroid
            polygon_cell_centroid_x_y = np.array([polygon_cell_centroid.x, polygon_cell_centroid.y])
            polygon_cell_centroid_x_y_str = str(round(polygon_cell_centroid_x_y[0], 3)) \
                                            + '_' + \
                                            str(round(polygon_cell_centroid_x_y[1], 3))

            max_x_polygon = max(transformed_coord[:, 0])
            min_x_polygon = min(transformed_coord[:, 0])
            max_y_polygon = max(transformed_coord[:, 1])
            min_y_polygon = min(transformed_coord[:, 1])

            adata_cells = adata_subset[

                          (adata_subset.obs['spatial_x_scaled'] > min_x_polygon) &
                          (adata_subset.obs['spatial_x_scaled'] < max_x_polygon) &
                          (adata_subset.obs['spatial_y_scaled'] > min_y_polygon) &
                          (adata_subset.obs['spatial_y_scaled'] < max_y_polygon)
            ,
                          :
                          ]

            for adata_cell in adata_cells.obs.index:
                point_probe = shapely.geometry.Point(adata_cells.obs.loc[adata_cell, 'spatial_x_scaled'],
                                                     adata_cells.obs.loc[adata_cell, 'spatial_y_scaled'])
                if polygon_cell.contains(point_probe):
                    adata.obs.loc[adata_cell, 'cell_bounds'] = polygon_cell_centroid_x_y_str
                    adata_subset.obs.loc[adata_cell, 'cell_bounds'] = polygon_cell_centroid_x_y_str

        adata_subset_plot = adata_subset[adata_subset.obs[new_obs_cell_bounds_variable_name] != default_cell_bounds, :]
        adata_subset_plot.obs[new_obs_cell_bounds_variable_name] = adata_subset_plot.obs['cell_bounds'].astype(str)

        # set to category variable
        adata_subset_plot.obs['cell_bounds'] = adata_subset_plot.obs['cell_bounds'].astype('category')

        if verbose:
            print(f'adata_subset shape: {adata_subset.shape}')

        # Check the tile and slide with boundary

        if not plot_comprehensively:
            return transfered_coords

        # Create a single large figure with all subplots
        fig = plt.figure(figsize=(20, 20))

        # First row (original first figure's subplots)
        ax1 = fig.add_subplot(421)
        ax1.imshow(tile_img)
        ax1.set_title('Tile')

        ax2 = fig.add_subplot(422)
        ax2.imshow(tile_img)
        for coord in detection_coords:
            ax2.scatter(coord[:, 0], coord[:, 1], c='r', s=1)
        ax2.set_title('Tile and Boundary')

        ax3 = fig.add_subplot(423)
        slide_region = slide.read_region(
            (boundary_x, boundary_y),
            0,
            (tile_img.width, tile_img.height))
        ax3.imshow(slide_region)
        ax3.set_title('Slide Region')

        ax4 = fig.add_subplot(424)
        ax4.imshow(slide_region)
        for coord in detection_coords:
            ax4.scatter(coord[:, 0], coord[:, 1], c='r', s=1)
        ax4.set_title('Slide Region with Boundary')

        # Second row (original second figure's subplots)
        ax5 = fig.add_subplot(425)
        cropped_thumbnail = slide_thumbnail.crop((int(x_thumbnial),
                                                  int(y_thumbnial),
                                                  int(x_thumbnial + width_thumbnail),
                                                  int(y_thumbnial + height_thumbnail)))
        ax5.imshow(cropped_thumbnail)
        ax5.set_title('Thumbnail')

        ax6 = fig.add_subplot(426)
        ax6.imshow(cropped_thumbnail)
        for coord in detection_coords:
            ax6.scatter(coord[:, 0] * scale_x, coord[:, 1] * scale_y, c='r', s=1)
        ax6.set_title('Thumbnail with Boundary')

        ax7 = fig.add_subplot(427)
        # print(f"min_spatial_x_scaled: {min_spatial_x_scaled}, min_spatial_y_scaled: {min_spatial_y_scaled}, max_spatial_x_scaled: {max_spatial_x_scaled}, max_spatial_y_scaled: {max_spatial_y_scaled}")

        cropped_region_adata_image = adata_image.crop((min_spatial_x_scaled,
                                                       min_spatial_y_scaled,
                                                       max_spatial_x_scaled,
                                                       max_spatial_y_scaled))
        ax7.imshow(cropped_region_adata_image,
                   extent=(min_spatial_x_scaled,
                           max_spatial_x_scaled,
                           max_spatial_y_scaled,
                           min_spatial_y_scaled))
        ax7.set_title('Adata Image')
        for transformed_coord in transformed_coords:
            ax7.scatter(transformed_coord[:, 0], transformed_coord[:, 1], c='y', s=0.1, alpha=0.5)
        ax7.scatter(adata_subset_plot.obs['spatial_x_scaled'],
                    adata_subset_plot.obs['spatial_y_scaled'],
                    c=adata_subset_plot.obs['cell_bounds'].cat.codes,
                    cmap='tab20',
                    s=1)

        ax8 = fig.add_subplot(428)
        ax8.imshow(cropped_region_adata_image,
                   extent=(min_spatial_x_scaled,
                           max_spatial_x_scaled,
                           max_spatial_y_scaled,
                           min_spatial_y_scaled),
                   alpha=0.6)
        ax8.scatter(adata_subset.obs['spatial_x_scaled'],
                    adata_subset.obs['spatial_y_scaled'],
                    c=adata_subset.obs['log2_adj_counts'],
                    s=1,
                    alpha=1,
                    vmin=0,
                    vmax=10,
                    cmap='seismic')
        for transformed_coord in transformed_coords:
            ax8.scatter(transformed_coord[:, 0], transformed_coord[:, 1], c='y', s=1)
        data_quality_variable = 'log2_adj_counts'
        ax8.set_title(f'Adata Image with Boundary (Y) and {data_quality_variable} (BR)')

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        if save_comprehensively:
            plt.savefig(save_comprehensively, bbox_inches='tight')
        plt.show()
        # sc.pl.spatial(adata_subset, color=['log2_adj_counts', 'log1p_total_counts'], ncols=2)

        plot_probe_continuous_variable_mapped_on_adata_img(adata=adata_subset,
                                                           adata_image=adata_image,

                                                           #    ploting_variable='spatial_x_scaled'
                                                           ploting_variable=data_quality_variable,

                                                           alpha=1,
                                                           vmin=0,
                                                           vmax=10,
                                                           #  vmax=8,

                                                           )

        return transfered_coords

    @staticmethod
    def transfer_mutliple_high_res_boundary_to_adata_obs_cell_bounds(

            boundary_list,
            tile_list,
            slide,
            slide_thumbnail,
            adata,
            adata_image,

            homoglaphy_matrix,
            save_adata_folder,
            save_adata_suffix,
            x_index=2,
            y_index=1,

            new_obs_cell_bounds_variable_name='cell_bounds',
            default_cell_bounds=None,
            plot_comprehensively_count=5,
            save_comprehensively_folder: str | Path = None,
            force=False,
            debug=False,
    ):
        counter = 0
        assert len(boundary_list) == len(
            tile_list), f"len(boundary_list): {len(boundary_list)} != len(tile_list): {len(tile_list)}"
        transfered_cell_bounds = []
        save_f = save_adata_folder / f'adata_with_cell_bounds_{save_adata_suffix}.h5ad'
        if save_f.exists() and not force:
            print(f"{save_f} already exists")
            print("If you want to overwrite, set force=True")
            return None

        from tqdm import tqdm

        for boundary_f, tile_f in tqdm(zip(boundary_list, tile_list),
                                       total=len(boundary_list),
                                       desc="Processing boundaries"):

            verbose = plot_comprehensively = True if counter <= plot_comprehensively_count else False
            save_comprehensively = None
            if save_comprehensively_folder and plot_comprehensively:
                save_comprehensively_folder.mkdir(parents=True, exist_ok=True)
                save_comprehensively = save_comprehensively_folder / f'{boundary_f.stem}.png'

            transfered_cell_bound = map_cell_boundary_from_ultra_high_res_histology_to_adata.transfer_high_res_boundary_to_adata_obs_cell_bounds(
                boundary_f,
                tile_f,
                slide,
                slide_thumbnail,
                adata,
                adata_image,
                homoglaphy_matrix,
                x_index,
                y_index,
                new_obs_cell_bounds_variable_name,
                default_cell_bounds,
                plot_comprehensively,
                verbose=verbose,
                save_comprehensively=save_comprehensively,

            )
            if transfered_cell_bound is None:
                continue

            counter += 1
            if debug:
                if counter > 10:
                    break
            transfered_cell_bounds.append(transfered_cell_bound)

        # adata.uns[new_obs_cell_bounds_variable_name] = transfered_cell_bounds
        save_transfered_cell_bounds_pickle = save_adata_folder / f'transfered_cell_bounds_{save_adata_suffix}.pkl'
        with open(save_transfered_cell_bounds_pickle, 'wb') as f:
            pickle.dump(transfered_cell_bounds, f)

        if debug:
            return transfered_cell_bounds
        adata.write_h5ad(save_f)

        return transfered_cell_bounds