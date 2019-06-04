import cv2
import numpy as np


def mask_iterator(skip_n=20, take_n=10):
    """
    Determines the filtering pattern of the image.

	Infinately loops per skip_n, take_n, and generates
	a 0/1 value, used to build a mask over an image

    Parameters
    ----------
    skip_n: int
    	How many sequential pixels are skipped
    take_n: int
    	How many sequential pixels are taken
    """
    while True:
        for _ in range(skip_n):
            yield 0
        for _ in range(take_n):
            yield 1


def generate_mask_array(arr_size, mask_iter):
	"""
	Iterate through the length of one dimension, per
	the definition of mask_iter

	Parameters
	----------
	arr_size: int
		Length of whatever dimension you want to make a
		mask for
	mask_iter: infinite iterable of ints
		Defined using `mask_iterator()`
	"""
	mask = []

	for _, mask_val in zip(range(arr_size), mask_iter):
	    mask.append(mask_val)

	return np.array(mask)


def im_to_sprite(im_fpath, skip_n=20, take_n=10, auto_size=False):
	"""
	Parameters
    ----------
    skip_n: int
    	How many sequential pixels are skipped
    take_n: int
    	How many sequential pixels are taken
    """
	raw = cv2.imread(im_fpath)
	raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

	raw_h, raw_w, _ = raw.shape

	if auto_size:
		max_dim = max(raw.shape[:2])
		take_n = max_dim // 30
		skip_n = take_n * 2

	mask_iter_row = mask_iterator(skip_n, take_n)
	mask_iter_col = mask_iterator(skip_n, take_n)

	row_mask = generate_mask_array(raw_h, mask_iter_row)
	col_mask = generate_mask_array(raw_w, mask_iter_col)

	row_repeated = np.repeat(row_mask, repeats=raw_w).reshape(raw_h, raw_w)
	col_repeated = np.repeat(col_mask, repeats=raw_h).reshape(raw_w, raw_h).T

	final_mask = row_repeated & col_repeated

	cols_activated = final_mask.sum(axis=1).max()
	rows_activated = final_mask.sum(axis=0).max()

	masked_raw = raw[final_mask.astype(bool)]

	sprite = masked_raw.reshape(rows_activated, cols_activated, 3)

	return sprite
