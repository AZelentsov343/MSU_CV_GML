import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=2000):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """

    descriptor_extractor = ORB(n_keypoints=n_keypoints)
    descriptor_extractor.detect_and_extract(rgb2gray(img))
    return descriptor_extractor.keypoints, descriptor_extractor.descriptors


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])

    means = np.mean(points, axis=0)
    cx = means[0]
    cy = means[1]
    N = np.sqrt(2) / (np.sum(np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)) / points.shape[0])
    matrix = np.array([[N, 0, -N*cx],
                       [0, N, -N*cy],
                       [0, 0, 1]])
    pointsh = matrix @ pointsh
    points_back_to_norm = pointsh[:2].T

    return matrix, points_back_to_norm


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    A = np.zeros((src.shape[0] * 2, 9))

    for i in range(src.shape[0]):
        A[i * 2] = np.array([-src[i][0], -src[i][1], -1, 0, 0, 0,
                             dest[i][0] * src[i][0], dest[i][0] * src[i][1], dest[i][0]])
        A[i * 2 + 1] = np.array([0, 0, 0, -src[i][0], -src[i][1], -1,
                                 dest[i][1] * src[i][0], dest[i][1] * src[i][1], dest[i][1]])
    u, s, v = np.linalg.svd(A)
    h = v[v.shape[0] - 1]
    H = h.reshape(3, 3)
    return inv(dest_matrix) @ H @ src_matrix


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=1000,
                     residual_threshold=6, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """

    match = match_descriptors(src_descriptors, dest_descriptors)
    src_keypoints = src_keypoints[match[:, 0]]
    dest_keypoints = dest_keypoints[match[:, 1]]

    best_inliers = 0
    best_inliers_index = None

    for trial in range(max_trials):
        random_index = np.random.choice(src_keypoints.shape[0], 4)

        H = find_homography(src_keypoints[random_index, :], dest_keypoints[random_index, :])

        src_homo = np.row_stack([src_keypoints.T, np.ones((src_keypoints.shape[0]), )])
        transformed = H @ src_homo
        transformed[0] /= transformed[2]
        transformed[1] /= transformed[2]
        transformed = transformed[:2].T

        distances = np.sqrt((dest_keypoints[:, 0] - transformed[:, 0]) ** 2 +
                            (dest_keypoints[:, 1] - transformed[:, 1]) ** 2)
        inliers = distances < residual_threshold
        if np.sum(inliers) >= best_inliers:
            best_inliers = np.sum(inliers)
            best_inliers_index = inliers

    H_res = find_homography(src_keypoints[best_inliers_index], dest_keypoints[best_inliers_index])
    if return_matches:
        return ProjectiveTransform(H_res), match[best_inliers_index]
    return ProjectiveTransform(H_res)


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1

    result = [None] * image_count
    result[(image_count - 1) // 2] = DEFAULT_TRANSFORM()
    cur = DEFAULT_TRANSFORM()
    i = (image_count - 1) // 2 - 1
    while i >= 0:
        cur = cur + forward_transforms[i]
        result[i] = cur
        i -= 1
    cur = DEFAULT_TRANSFORM()
    i = (image_count - 1) // 2
    while i < image_count - 1:
        cur = cur + ProjectiveTransform(inv(forward_transforms[i].params))
        result[i + 1] = cur
        i += 1

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
        """
    corners = tuple(get_corners(image_collection, simple_center_warps))
    min_coords, max_coords = get_min_max_coords(corners)
    width = max_coords[0] - min_coords[0]
    height = max_coords[1] - min_coords[1]
    shift_matrix = np.array([[1, 0, -min_coords[1]],
                             [0, 1, -min_coords[0]],
                             [0, 0, 1]])
    result = []
    for w in simple_center_warps:
        result.append(w + ProjectiveTransform(shift_matrix))
    return tuple(result), (int(height) + 1, int(width) + 1)


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    transformed_img = warp(image, rotate_transform_matrix(transform), output_shape=output_shape)
    mask = (transformed_img[:, :, 0] > 0) | (transformed_img[:, :, 1] > 0) | (transformed_img[:, :, 2] > 0)
    return transformed_img, mask.astype(bool)


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    h, w = output_shape
    result = np.zeros((h, w, 3))
    result_mask = np.zeros((h, w), dtype=bool)
    for i in range(len(image_collection)):
        cur_img, cur_mask = warp_image(image_collection[i], ProjectiveTransform(inv(final_center_warps[i].params)),
                                       output_shape)
        cur_mask &= ~result_mask
        cur_img = np.dstack((cur_img[:, :, 0] * cur_mask, cur_img[:, :, 1] * cur_mask, cur_img[:, :, 2] * cur_mask))
        result_mask |= cur_mask
        result += cur_img
    return np.clip(np.rint(result * 255), 0, 255).astype('uint8')


def get_gaussian_pyramid(image, n_layers, sigma):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    cur_image = np.copy(image)
    result = [cur_image]
    for i in range(n_layers - 1):
        cur_image = gaussian(cur_image, sigma)
        result.append(cur_image)
    return tuple(result)


def get_laplacian_pyramid(image, n_layers=4, sigma=1):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """

    result = list(get_gaussian_pyramid(image, n_layers, sigma))

    for i in range(len(result) - 1):
        result[i] -= result[i + 1]
    return tuple(result)


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=4, image_sigma=2, merge_sigma=10):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """

    h, w = output_shape
    result = np.zeros((h, w, 3))
    result_mask = np.zeros((h, w), dtype=bool)
    for i in range(len(image_collection)):
        image, mask = warp_image(image_collection[i], ProjectiveTransform(inv(final_center_warps[i].params)),
                                 output_shape)

        middle_mask = np.ones(output_shape, dtype=bool)
        middler = np.nonzero((result_mask & mask).any(axis=0))
        border = (middler[0][0] + middler[0][-1]) // 2 if middler[0].size != 0 else 0
        middle_mask[:, :border] = 0

        la = get_laplacian_pyramid(image, n_layers, image_sigma)
        lb = get_laplacian_pyramid(result, n_layers, image_sigma)
        gm = get_gaussian_pyramid(middle_mask.astype(float), n_layers, merge_sigma)
        laplas = []
        for j in range(n_layers):
            ima = np.dstack([la[j][:, :, 0] * gm[j], la[j][:, :, 1] * gm[j], la[j][:, :, 2] * gm[j]])
            imb = np.dstack([lb[j][:, :, 0] * (1 - gm[j]), lb[j][:, :, 1] * (1 - gm[j]),
                             lb[j][:, :, 2] * (1 - gm[j])])
            laplas.append(ima + imb)

        result_mask |= (~result_mask) & mask
        result = merge_laplacian_pyramid(laplas)

    return np.clip(np.rint(result * 255), 0, 255).astype('uint8')


