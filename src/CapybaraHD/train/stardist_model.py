from __future__ import print_function, unicode_literals, absolute_import, division, annotations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw
from stardist import fill_label_holes, random_label_cmap, gputools_available, calculate_extents
import shapely
from pathlib import Path
import json
from typing import List
from shutil import copytree
from csbdeep.utils import normalize
from stardist.models import Config2D, StarDist2D
import sys




np.random.seed(42)
lbl_cmap = random_label_cmap()

def create_polygon_mask_xml(image_array, vertices_by_region):
    """
    Create a mask matrix where each polygon region is filled with a unique ID.
    
    Parameters:
    image_array: numpy array of the image
    vertices_by_region: dictionary of region_id: vertex_coordinates
    
    Returns:
    numpy array with same height/width as image_array, where each polygon
    region is filled with its ID (starting from 1) and background is 0
    """
    height, width = image_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.int32)
    
    # Create a PIL image for drawing
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # For each region, draw the polygon and assign unique ID
    for i, (region_id, vertices) in enumerate(vertices_by_region.items(), start=1):
        # Convert vertices to tuple format for PIL
        polygon_coords = [(x, y) for x, y in vertices]
        
        # Draw the polygon filled with the region ID
        draw.polygon(polygon_coords, fill=i)
    
    # Convert PIL image back to numpy array
    mask = np.array(img)
    
    return mask



def create_polygon_mask(image_array, polygon_coordinates, polygon_id=1):
    # import numpy as np
    from matplotlib.path import Path
    """
    Creates a mask matrix where pixels inside the polygon get a unique ID
    and pixels outside get 0.
    
    Parameters:
    image_array: numpy array of the image
    polygon_coordinates: list of (x,y) coordinates defining the polygon
    polygon_id: integer value to assign to pixels inside the polygon
    
    Returns:
    mask_matrix: numpy array with same shape as image_array
    """
    # Create empty mask with same shape as input image
    mask = np.zeros(image_array.shape[:2], dtype=np.int32)
    
    # Create a meshgrid of pixel coordinates
    y, x = np.mgrid[:image_array.shape[0], :image_array.shape[1]]
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    
    # Create Path object from polygon coordinates
    polygon_path = Path(polygon_coordinates)
    
    # Test which points are inside the polygon
    mask_flat = polygon_path.contains_points(points)
    mask_reshaped = mask_flat.reshape(image_array.shape[:2])
    
    # Set the value for points inside the polygon
    mask[mask_reshaped] = polygon_id
    
    return mask

# Create the mask for all polygons in your GeoJSON
def create_multi_polygon_mask(image_array, geojson_features):
    """
    Creates a mask matrix for multiple polygons from GeoJSON features.
    
    Parameters:
    image_array: numpy array of the image
    geojson_features: GeoJSON features containing polygon coordinates
    
    Returns:
    mask_matrix: numpy array with same shape as image_array
    """
    final_mask = np.zeros(image_array.shape[:2], dtype=np.int32)
    
    for idx, polygon_coords in enumerate(geojson_features[0]['geometry']['coordinates'], 1):
        # Convert coordinates to the format expected by matplotlib Path
        coords = np.array(polygon_coords[0])
        
        # Create mask for this polygon
        polygon_mask = create_polygon_mask(image_array, coords, polygon_id=idx)
        
        # Combine with final mask (later polygons will overwrite earlier ones)
        final_mask = np.maximum(final_mask, polygon_mask)
    
    return final_mask
def load_img_into_channel(image_f:str|Path,):
    image_f = Path(image_f)
    assert image_f.exists(), f"Image file {image_f} does not exist"
    
    # Load the image
    image = Image.open(image_f)
    image_array = np.array(image)
    
    # Handle different channel configurations
    if image_array.ndim == 2:  # Grayscale
        # Convert to 3 channels by repeating the same data
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.ndim == 3:
        if image_array.shape[-1] == 4:  # RGBA
            # Convert to RGB by dropping alpha channel
            image_array = image_array[..., :3]
        elif image_array.shape[-1] != 3:  # Unknown format
            raise ValueError(f"Unexpected number of channels: {image_array.shape[-1]}")
    
    return image_array


def load_annotation(annotation_f:str|Path):
    annotation_f = Path(annotation_f)
    assert annotation_f.exists(), f"Annotation file {annotation_f} does not exist"
    
    # Load the geojson
    with open(annotation_f) as f:
        data = json.load(f)
    assert len(data['features']) == 1, f"Expected 1 feature in GeoJSON, found {len(data['features'])}"
    return data


def plot_img_with_mask(image_array:np.array, data: dict, show: bool=True,
                       ):
        
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.imshow(image_array)
    ax_2 = fig.add_subplot(212)
    ax_2.imshow(image_array)
    for object in data['features'][0]['geometry']['coordinates']:
        # print(object['geometry']['coordinates'])
        polygon = shapely.geometry.Polygon(object[0])
        plt.fill(*polygon.exterior.xy,
                    edgecolor='red',
                    fill=False,
                    lw=2,
                 )
    if show:
        plt.show()
        
    return fig, ax, ax_2
    # plt.show()
    
    
def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()
    
def generate_img_array_mask(image_f:str|Path,
                            annotation_f:str|Path,
                            show_labels: bool=True,
                            ):
    
    """
    Generate the image array and mask matrix from the image and annotation files.
    Parameters:
    image_f: str or Path, path to the image file
    annotation_f: str or Path, path to the annotation file (GeoJSON)
    show_labels: bool, whether to show the labels on the image
    """
    
    
    # Load the image
    image_array = load_img_into_channel(image_f)

    # Load the annotation in geojson
    data = load_annotation(annotation_f)
    # Get the coordinates of the polygons
    
    plot_img_with_mask(image_array, data, show=show_labels)

    mask_matrix = create_multi_polygon_mask(image_array, data['features'])
    img, lbl = image_array, fill_label_holes(mask_matrix)
    assert img.ndim in (2,3)
    img = img if img.ndim==2 else img[...,:3]
    plot_img_label(img, lbl, show_labels)
    
    

    
    return image_array, mask_matrix


def check_channel(all_images: List[np.ndarray]):
    
    """
    Check if all images have the same number of channels.
    If they do, return the number of channels.
    """
    
    
    n_channels = set()
    for image in all_images:
        if image.ndim == 2:
            n_channel = 1
        else:
            n_channel = image.shape[-1]
        n_channels.add(n_channel)
        
    assert len(n_channels) == 1, f"All images should have the same number of channels. Found: {n_channels}"
    
    return n_channels.pop()


def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y


def generate_train(all_images: List[np.ndarray], all_masks: List[np.ndarray], 
                   axis_norm: tuple = (0,1) ):
    

    # X = preprocess_images(all_images)
    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(all_images)]
    Y = [fill_label_holes(y) for y in tqdm(all_masks)]

    
    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))
    
    return X_trn, Y_trn, X_val, Y_val, X, Y


    
    
def train_stardist(
    X_trn, Y_trn, X_val, Y_val, X, Y, 
    n_channels: int, saving_f:str|Path, 
    *,
    epochs: int = 100,
    steps_per_epoch: int = 100,
    n_rays: int = 32,
    grid: tuple = (2,2),
    agumenter_demo: bool = False,
    quick_demo: bool = False,
    gpu: bool = False,
):

    
    use_gpu = False if not gputools_available() else gpu and gputools_available()
    # print(f"Using GPU: {use_gpu}")
    if agumenter_demo:
        img, lbl = X[0],Y[0]
        plot_img_label(img, lbl)
        for _ in range(3):
            img_aug, lbl_aug = augmenter(img,lbl)
            plot_img_label(img_aug, lbl_aug, img_title="image augmented", lbl_title="label augmented")

    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        use_gpu      = use_gpu,
        n_channel_in = n_channels,
        # n_channel_in = 1,
    )
    print(conf)
    


    model = StarDist2D(conf, name='stardist', basedir='models')
    median_size = calculate_extents(list(Y), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    if quick_demo:
        print (
            "NOTE: This is only for a quick demonstration!\n"
            "      Please set the variable 'quick_demo = False' for proper (long) training.",
            file=sys.stderr, flush=True
        )
        model.train(X_trn, Y_trn, 
                    validation_data=(X_val,Y_val), 
                    augmenter=augmenter,
                    epochs=10, steps_per_epoch=50)

        print("====> Stopping training and loading previously trained demo model from disk.", file=sys.stderr, flush=True)
        # model = StarDist2D.from_pretrained('2D_demo')
        model.optimize_thresholds(X_val[:2], Y_val[:2])
    else:
        model.train(X_trn, Y_trn, 
                    validation_data=(X_val,Y_val), 
                    augmenter=augmenter)
        model.optimize_thresholds(X_val, Y_val)
        print("====> Training finished.", file=sys.stderr, flush=True)
        
    # Save the model
    print(f"Saving model to {saving_f}")
    saving_f = Path(saving_f)
    saving_f.mkdir(parents=True, exist_ok=True)
    copytree(model.logdir, str(saving_f), dirs_exist_ok=True)
    
    
    return model


def convert_np_image_to_mask(image_array, 
                             plot_overley_show=False, 
                             plot_overlay_img=None,
                             plot_overlay_save=None,
                             ):
    import numpy as np
    from skimage import measure
    # import shapely.geometry as geometry
    import matplotlib.pyplot as plt
    
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
    ax =plt.subplot(121)

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
    
    return {'coord':all_polygons_coords, 'ids':all_polygons_ids}



def predict_new_image(image_f: str|Path, 
                      model: StarDist2D, 
                      save_npy_corrds_f: str|Path,
                      *,
                      plot_overlay_save: str|Path|None= None,
                      axis_norm: tuple = (0,1),
                      pro_thresh: float|None = None,
                      ):
    
    
    image_array = np.array(Image.open(image_f))
    image_array_norm = normalize(image_array,1,99.8,axis=axis_norm)
    image_pred = model.predict_instances(image_array_norm, 
                                         n_tiles=model._guess_n_tiles(image_array_norm), 
                                         show_tile_progress=False)[0]
    plot_img_label(image_array_norm,image_pred, lbl_title="label Pred")
    plot_overley_show = False
    if plot_overlay_save:
        plot_overley_show = True
        plot_overlay_save = Path(plot_overlay_save)
        plot_overlay_save.parent.mkdir(parents=True, exist_ok=True)
        
    new_polygons = convert_np_image_to_mask(image_pred,
                                            plot_overley_show=plot_overley_show,
                                            plot_overlay_img=image_array,
                                            plot_overlay_save=plot_overlay_save,
                                           )
    np.save(save_npy_corrds_f, new_polygons)    