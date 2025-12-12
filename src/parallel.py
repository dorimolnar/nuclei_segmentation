# parallel_tiles.py
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
import numpy as np
import os
from tqdm import tqdm

from segmentation import watershed_segmentation
from classification import classify_nuclei_by_brownness

def process_tile(tile: np.ndarray, tile_origin: Tuple[int,int], tile_inner_box: Tuple[int,int,int,int]) -> Tuple[Tuple[int,int,int,int], np.ndarray]:
    """
    Run segmentation and classification on a tile and return the center region to place back.
    Returns (global_inner_box, classified_center_region)
    """
    # tile: small numpy array (H_tile, W_tile, 3)
    # tile_origin: (y0, x0) global origin in full image
    # tile_inner_box: (iy0, ix0, iy1, ix1) coords inside tile (non-overlap region) in tile-local coords

    # Run segmentation on the tile (returns labeled mask and maybe colored seg)
    labeled, _ = watershed_segmentation(tile, thr_method='adaptive')  # adapt to your function signature
    classified_tile = classify_nuclei_by_brownness(tile, labeled)

    iy0, ix0, iy1, ix1 = tile_inner_box
    center = classified_tile[iy0:iy1, ix0:ix1].copy()  # copy to avoid referencing shared memory

    # Global coordinates of the center region
    y0_global = tile_origin[0] + iy0
    x0_global = tile_origin[1] + ix0
    y1_global = tile_origin[0] + iy1
    x1_global = tile_origin[1] + ix1
    global_box = (y0_global, x0_global, y1_global, x1_global)

    return global_box, center


def process_image_in_parallel(image: np.ndarray,
                              tile_size: int = 1024,
                              overlap: int = 200,
                              max_workers: int | None = None) -> np.ndarray:
    H, W = image.shape[:2]
    output = np.zeros_like(image)

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)

    # Build list of tiles (tile origin and tile local inner region)
    tasks = []
    step = tile_size - overlap
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            tile = image[y0:y1, x0:x1]

            # inner region coords inside tile (non-overlap center)
            iy0 = overlap // 2 if y0 != 0 else 0
            ix0 = overlap // 2 if x0 != 0 else 0
            iy1 = (y1 - y0) - (overlap // 2) if y1 != H else (y1 - y0)
            ix1 = (x1 - x0) - (overlap // 2) if x1 != W else (x1 - x0)

            # ensure valid
            iy0 = int(max(0, iy0)); ix0 = int(max(0, ix0))
            iy1 = int(max(iy0+1, iy1)); ix1 = int(max(ix0+1, ix1))

            tasks.append((tile.copy(), (y0, x0), (iy0, ix0, iy1, ix1)))

    # Dispatch in processes
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_tile, tile, origin, inner): (tile, origin, inner)
                   for tile, origin, inner in tasks}

        # Optionally use tqdm
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Tiles"):
            try:
                global_box, center = fut.result()
                results.append((global_box, center))
            except Exception as e:
                # decide whether to fail or skip
                raise

    # Stitch results back (no overlap) - if multiple results overlap, last write wins (but we designed non-overlap)
    for (y0, x0, y1, x1), center in results:
        output[y0:y1, x0:x1] = center

    return output