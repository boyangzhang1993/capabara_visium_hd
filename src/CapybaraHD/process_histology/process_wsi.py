from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
from stardist import random_label_cmap
from stardist.models import StarDist2D
from pathlib import Path
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from csbdeep.utils import Path, normalize
import os
import numpy as np
from skimage import measure
import openslide.deepzoom
import matplotlib.pyplot as plt
import openslide
import sys


class ProcessTile:

    # This class processes a whole slide image (WSI) to generate tiles, detect HE stain, and predict cell instances using StarDist.
    # inputs:
    # - slide_f: Path to the WSI file.
    # - model_path: Path to the directory containing the StarDist model.
    # - model_name: Name of the StarDist model to use (default is 'v2_all_samples').
    # - tiles_folder: Optional path to save the generated tiles. If not provided, a default folder will be created at slide_f.parent / 'analysis' / slide_f.stem / f'{slide_f.stem}_tiles'.
    # outputs:
    # - Generates tiles from the WSI and saves them in the tiles_folder.
    # - Detects HE stain tiles and moves them to a separate folder.
    # - Predicts cell instances in the tiles using the StarDist model and saves the results in a numpy.

    def __init__(self,
                 slide_f: Path | str,
                 model_path: Path | str,
                 model_name: str | None = None,
                 tiles_folder: Path | None = None,

                 ):

        if isinstance(slide_f, str):
            slide_f = Path(slide_f)
        if isinstance(model_path, str):
            model_path = Path(model_path)

        assert all([model_path.exists(), model_path.is_dir(),
                    slide_f.exists()], ), f"Model path {model_path} or slide file {slide_f} does not exist or is not a directory."

        self.slide_f = slide_f
        # Set deeplearn model path and name
        self.model_path = model_path
        self.model_name = model_name
        
        if model_name is None:
            self.model_name = model_path.name
            self.model_path = self.model_path.parent

        self.slide = openslide.OpenSlide(str(slide_f))
        self.slide.get_thumbnail(size=(500, 500))
        # Save
        if tiles_folder is None:
            print(f"Tiles folder not provided. Using default folder.")
            tiles_folder = slide_f.parent / 'analysis' / slide_f.stem / f'{slide_f.stem}_tiles'
            print(f"    Saving tiles to {tiles_folder}")
        if isinstance(tiles_folder, str):
            tiles_folder = Path(tiles_folder)

        self.tiles_folder = tiles_folder
        self.analyze_folder = tiles_folder.parent
        self.not_he_folder = self.analyze_folder / 'not_he_stain'

    def auto(self):
        self.analyze_folder.mkdir(exist_ok=True, parents=True)
        self.tiles_folder.mkdir(exist_ok=True, parents=True)

        # self.output_directory = output_directory
        # self.model = StarDist2D.from_pretrained('2D_versatile_he')

        self.generate_tile()
        self.detect_he_stain_tiles()
        self.load_stardist_model(model_path=self.model_path, model_name=self.model_name)
        self.predict_tile(self.tiles_folder)

    def load_stardist_model(self, model_path: Path, model_name: str):
        # model_path = Path(model_path) / model_name
        print(f"Loading model from {model_path}")
        self.model = StarDist2D(None, model_name, basedir=model_path)

    def predict_tile_v1(self, tiles_folder: Path):
        """
        Predicts cell instances in image tiles using StarDist model.
        Aligns with the logic from the second file's detect_cell method.

        Args:
            tiles_folder (Path): Path to folder containing image tiles
        """

        def convert_np_image_to_mask_v1(image_array,
                                        plot_overley_show=False,
                                        plot_overlay_img=None,
                                        plot_overlay_save=None):
            """
            Converts numpy array of instance predictions to mask and polygons.

            Args:
                image_array: Numpy array of instance predictions
                plot_overley_show: Whether to show overlay plot
                plot_overlay_img: Original image for overlay
                plot_overlay_save: Path to save overlay plot

            Returns:
                dict: Contains polygon coordinates and cell IDs
            """

            if plot_overley_show and plot_overlay_img is None:
                plot_overlay_img = image_array

            if isinstance(image_array, list):
                image_array = np.array(image_array)

            # Get unique cell IDs
            cell_ids = np.unique(image_array)
            cell_ids = cell_ids[cell_ids > 0]

            # Initialize storage for polygons
            all_polygons_coords = []
            all_polygons_ids = []

            # Process each cell
            for cell_id in cell_ids:
                cell_mask = (image_array == cell_id).astype(np.uint8)
                contours = measure.find_contours(cell_mask, 0.5)

                for contour in contours:
                    polygon_coords = np.fliplr(contour)
                    all_polygons_coords.append(polygon_coords)
                    all_polygons_ids.append(cell_id)

            # Create visualization if requested
            if plot_overley_show:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

                # Plot original with polygons
                ax1.imshow(plot_overlay_img)
                colors = ['r', 'g', 'b', 'y', 'c', 'm']
                for idx, coords in enumerate(all_polygons_coords):
                    color = colors[idx % len(colors)]
                    ax1.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1)
                ax1.set_title('Cell Boundaries Overlay')

                # Plot original image
                ax2.imshow(plot_overlay_img)
                ax2.set_title('Original Image')

                if plot_overlay_save:
                    plt.savefig(plot_overlay_save)
                plt.close()

            return {'coord': all_polygons_coords, 'ids': all_polygons_ids}

        print(f"Processing tiles for cell detection")

        # Create output folders
        detection_folder = tiles_folder.parent / 'nucleus_predictions'
        detection_plot_folder = tiles_folder.parent / 'nucleus_predictions_debug'
        detection_folder.mkdir(exist_ok=True, parents=True)
        detection_plot_folder.mkdir(exist_ok=True, parents=True)

        # Get all tiles
        tiles = sorted([x for x in tiles_folder.glob('*.png')])

        for counter, tile_path in enumerate(tqdm(tiles, desc="Processing tiles", unit="tile")):
            # Extract coordinates from filename
            tilt_x, tilt_y = tile_path.stem.split('_')[1:]

            # Define save paths
            save_label_f = detection_folder / f'{tilt_x}_{tilt_y}.npy'
            save_details_f = detection_folder / f'{tilt_x}_{tilt_y}_details.npy'
            save_debug_image_f = detection_plot_folder / f'{tilt_x}_{tilt_y}_detections.png'

            # Skip if already processed
            if save_label_f.exists() and save_details_f.exists():
                print(f"Detection for {tile_path} exists, skipping")
                continue

            # Load and process image
            image_array = np.array(Image.open(tile_path))
            image_array_norm = normalize(image_array, 1, 99.8, axis=(0, 1))

            # Predict instances
            labels, details = self.model.predict_instances(
                image_array_norm,
                n_tiles=self.model._guess_n_tiles(image_array_norm),
                show_tile_progress=False
            )

            # Generate debug visualization
            if counter <= 10:  # Only create detailed plots for first 10 tiles
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

                # Original image
                ax1.imshow(image_array)
                ax1.set_title('Original Image')
                ax1.axis('off')

                # Predictions overlay
                ax2.imshow(image_array)
                ax2.imshow(labels, cmap=random_label_cmap(), alpha=0.5)
                ax2.set_title('Cell Detections')
                ax2.axis('off')

                plt.tight_layout()
                plt.savefig(save_debug_image_f)
                plt.close()

                # Generate polygon visualization
                new_polygons = convert_np_image_to_mask_v1(
                    labels,
                    plot_overley_show=True,
                    plot_overlay_img=image_array,
                    plot_overlay_save=save_debug_image_f.parent / f'{tilt_x}_{tilt_y}_overlay.png'
                )
            else:
                new_polygons = convert_np_image_to_mask_v1(
                    labels,
                    plot_overley_show=False,
                    plot_overlay_img=image_array
                )

            # Save results
            np.save(save_label_f, labels)
            np.save(save_details_f, details)

    def predict_tile(self, tile: Path):

        model = self.model
        all_new_images_folder = Path(tile)

        all_new_images = sorted(all_new_images_folder.glob('*.png'))

        save_coords_folder = self.save_coords_folder = all_new_images_folder.parent / 'nucleus_predictions'
        save_coords_folder.mkdir(exist_ok=True)

        save_debug_image_folder = all_new_images_folder.parent / 'nucleus_predictions_debug'
        save_debug_image_folder.mkdir(exist_ok=True)

        def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
            lbl_cmap = random_label_cmap()
            fig, (ai, al) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=(1.25, 1)))
            im = ai.imshow(img, cmap='gray', clim=(0, 1))
            ai.set_title(img_title)
            fig.colorbar(im, ax=ai)
            al.imshow(lbl, cmap=lbl_cmap)
            al.set_title(lbl_title)
            plt.tight_layout()

        def convert_np_image_to_mask(image_array,
                                     plot_overley_show=False,
                                     plot_overlay_img=None,
                                     plot_overlay_save=None,
                                     ):

            if plot_overley_show and plot_overlay_img is None:
                plot_overlay_img = image_array

            if isinstance(image_array, list):
                image_array = np.array(image_array)

            cell_ids = np.unique(image_array)
            cell_ids = cell_ids[cell_ids > 0]

            cell_ids = np.unique(image_array)
            cell_ids = cell_ids[cell_ids > 0]

            # Initialize list to store all cell polygons
            all_polygons = []
            all_polygons_coords = []
            all_polygons_ids = []
            # Process each cell
            for cell_id in cell_ids:
                # Create binary mask for current cell
                cell_mask = (image_array == cell_id).astype(np.uint8)

                # Find contours of the cell
                contours = measure.find_contours(cell_mask, 0.5)

                # Process each contour found for the cell
                for contour in contours:
                    # Convert contour to polygon format
                    # Swap x and y coordinates because find_contours returns (row, col)
                    polygon_coords = np.fliplr(contour)

                    # Store the polygon coordinates directly without Shapely
                    all_polygons.append({
                        'cell_id': cell_id,
                        'coords': polygon_coords
                    })
                    all_polygons_coords.append(polygon_coords)
                    all_polygons_ids.append(cell_id)

            # Create figure and axis
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(121)

            # Plot the original image
            ax.imshow(plot_overlay_img, )

            # Plot each polygon
            colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Different colors for different cells
            for idx, cell_polygon in enumerate(all_polygons):
                color = colors[idx % len(colors)]  # Cycle through colors
                coords = cell_polygon['coords']
                ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2,
                        label=f"Cell {cell_polygon['cell_id']}")

            ax.legend().set_visible(False)
            # ax.title('Cell Boundaries Overlay')
            # ax.axis('image')
            ax.set_title('Cell Boundaries Overlay')

            ax = plt.subplot(122)
            # Plot the original image
            ax.imshow(plot_overlay_img, )
            ax.set_title('Original Image')
            # plt.legend()
            if plot_overlay_save:
                plt.savefig(plot_overlay_save)
            plt.show()

            return {'coord': all_polygons_coords, 'ids': all_polygons_ids}

        counter = 0
        # len_debug = len
        axis_norm = (0, 1)
        for new_image in all_new_images:
            save_npy_corrds_f = save_coords_folder / (new_image.stem + '_coords.npy')
            if save_npy_corrds_f.exists():
                continue

            image_array = np.array(Image.open(new_image))
            image_array_norm = normalize(image_array, 1, 99.8, axis=axis_norm)
            image_pred = model.predict_instances(image_array_norm, n_tiles=model._guess_n_tiles(image_array_norm),
                                                 show_tile_progress=False)[0]
            plot_img_label(image_array_norm, image_pred, lbl_title="label Pred")

            plot_overlay_save = save_debug_image_folder / (new_image.stem + '_overlay.png')
            plot_overley_show = True
            if counter > 100:
                plot_overlay_save = None
                plot_overley_show = False

            new_polygons = convert_np_image_to_mask(image_pred,
                                                    plot_overley_show=plot_overley_show,
                                                    plot_overlay_img=image_array,
                                                    plot_overlay_save=plot_overlay_save,
                                                    )

            counter += 1
            np.save(save_npy_corrds_f, new_polygons)

    def generate_tile(self, tile_size: int = 512):

        print(f"Generating tiles")

        slide = self.slide
        saving_tiles_folder = self.tiles_folder
        not_he_folder = self.analyze_folder / 'not_he_stain'
        print(f'    Saving tiles to {saving_tiles_folder}')
        self.deepzoom = deepzoom = openslide.deepzoom.DeepZoomGenerator(slide,
                                                                        tile_size=tile_size,
                                                                        overlap=0,
                                                                        limit_bounds=False)
        # = deepzoom

        # deepzoom._t_dimensions
        level_max = deepzoom.level_count - 1

        row_max = deepzoom._t_dimensions[level_max][0]
        col_max = deepzoom._t_dimensions[level_max][1]

        with tqdm(total=row_max * col_max) as pbar:
            # pbar.set_description("Generating tiles: ")
            # pbar.set_postfix({"Total tiles": row_max * col_max})
            for i in range(0, row_max, 1):
                print(f'    Generating tiles for row: {i} (total rows: {row_max})')

                # pbar.set_description(f"Generating tiles: {i}/{row_max}")
                for j in range(0, col_max, 1):
                    pbar.update(1)
                    #
                    #
                    # for i in tqdm(range(0, row_max, 1), desc='Rows'):
                    #     for j in tqdm(range(0, col_max, 1), desc='Columns', leave=False):
                    # Calculate pixel coordinates
                    x = j * tile_size  # column * tile_size gives x coordinate
                    y = i * tile_size  # row * tile_size gives y coordinate

                    # Create both naming conventions
                    new_name = f'{i}_{j}.png'  # row_col format
                    old_name = f'tile_{x}_{y}.png'  # pixel coordinate format

                    # Define file paths
                    save_img_f = saving_tiles_folder / old_name  # Using old naming convention
                    save_img_f_new = saving_tiles_folder / new_name  # Using new naming convention
                    moved_img_f = self.not_he_folder / old_name

                    not_he_f = self.not_he_folder / old_name

                    # Check if file should be skipped
                    if moved_img_f.exists():
                        for f in [save_img_f, save_img_f_new]:
                            if f.exists():
                                f.unlink()
                        continue

                    if not_he_f.exists():
                        continue
                    if save_img_f.exists() or save_img_f_new.exists():
                        continue

                    # Generate and save tile
                    img = deepzoom.get_tile(level_max, (i, j))

                    # Save with both naming conventions
                    img.save(save_img_f, 'PNG')

    def detect_he_stain(self, image_path: Path, move_path: Path = None, threshold=95):

        def classify_image(image_path, target_color=np.array([157, 71, 119]), threshold=50):
            # Open the image
            # Image.MAX_TEXT_CHUNK = 100 * (1024 * 1024)  # 100MB chunk size

            from PIL import PngImagePlugin
            LARGE_ENOUGH_NUMBER = 100
            PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)
            print(image_path)
            img = Image.open(image_path)

            # Convert image to numpy array
            img_array = np.array(img)

            # Reshape the array to 2D (each row is a pixel)
            pixels = img_array.reshape(-1, 3)

            # Calculate distances from each pixel to the target color
            distances = np.sqrt(np.sum((pixels - target_color) ** 2, axis=1))

            # Classify pixels
            close_mask = distances <= threshold

            # Calculate percentages
            total_pixels = len(pixels)
            close_percentage = np.sum(close_mask) / total_pixels * 100
            not_close_percentage = 100 - close_percentage

            return not_close_percentage

        target_color = np.array([157, 71, 119])

        not_close_percentage = classify_image(image_path, target_color)

        if not_close_percentage > threshold:
            # print("This image does not appear to be a HE stain.")
            if move_path is None:
                return False
            destination = move_path / image_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            image_path.rename(destination)
            # image_path.

    def detect_he_stain_tiles(self, ):
        print(f"Detecting HE stain")

        move_path = self.not_he_folder
        if move_path.exists():
            print(f"Folder {move_path} already exists.")
            print('The detection may have been run before.')
            print("Please remove it before running this function.")
            return

        move_path.mkdir(parents=True, exist_ok=False)

        tiles_folder = self.tiles_folder
        tiles = [x for x in tiles_folder.iterdir() if x.is_file() and x.suffix == '.png']
        with tqdm(total=len(tiles), desc="Total tiles", unit="tile") as pbar:
            pbar.set_postfix({"Total tiles": len(tiles)})

            for tile in tiles:
                pbar.update(1)
                self.detect_he_stain(tile, move_path=move_path)





