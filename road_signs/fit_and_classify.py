import sklearn.svm as svm
from skimage.transform import resize
import numpy as np
from scipy.signal import convolve2d


def take_grad(img, kernel='sobel'):
    if kernel == 'difference':
        kernelX = np.array([[1., 0., -1.]])
        kernelY = kernelX.reshape(-1, 1)
    else:
        kernelX = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        kernelY = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
    gradX = convolve2d(img, kernelX, boundary='fill', fillvalue=0.0, mode='same')
    gradY = convolve2d(img, kernelY, boundary='fill', fillvalue=0.0, mode='same')
    return gradX, gradY


def grad_dirmod(gradX, gradY):
    mod = np.sqrt(gradX.astype('float64') ** 2 + gradY.astype('float64') ** 2)
    direction = np.arctan2(gradY, gradX) + np.pi
    return mod, direction


def build_hist(mod, direction, cell_rows, cell_cols, bin_count):
    assert (np.all(mod.shape == direction.shape))
    direction_bin_size = 2 * np.pi / bin_count
    direction_mask = np.zeros((mod.shape[0], mod.shape[1], bin_count), dtype='float64')
    hist = np.zeros((mod.shape[0] // cell_rows, mod.shape[1] // cell_cols, bin_count), dtype='float64')

    for k in range(bin_count):
        direction_mask[:, :, k] = ((direction >= k * direction_bin_size) &
                                   (direction < k * direction_bin_size + direction_bin_size)) * mod

    for k in range(bin_count):
        for i in range(mod.shape[0] // cell_rows):
            for j in range(mod.shape[1] // cell_cols):
                hist[i, j, k] = np.sum(direction_mask[i * cell_rows:i * cell_rows + cell_rows,
                                                      j * cell_cols:j * cell_cols + cell_cols, k])

    return hist


def merge_blocks(hist, block_row_cells, block_col_cells):
    feats = np.array([], dtype='float64')

    for i in range(0, hist.shape[0] - block_row_cells + 1):
        for j in range(0, hist.shape[1] - block_col_cells + 1):
            part_hist = np.ravel(hist[i:i + block_row_cells, j:j + block_col_cells])
            norm = part_hist / np.sqrt(np.sum(part_hist ** 2) + 1e-6)
            feats = np.append(feats, norm)

    return np.ravel(feats)


def extract_hog(img):
    block_row_cells = 2
    block_col_cells = 2
    cell_rows = 8
    cell_cols = 8
    bin_count = 8
    cut_c = 0.05

    height, width, _ = img.shape

    cut_image = img[int(np.rint(height * cut_c)): int(height - np.rint(height * cut_c)),
                    int(np.rint(width * cut_c)):  int(width - np.rint(width * cut_c))]
    resize_image = resize(cut_image, (64, 64))

    gray = (0.299 * resize_image[:, :, 0] + 0.587 * resize_image[:, :, 1] +
            0.114 * resize_image[:, :, 2]).astype(np.float64)

    gradx, grady = take_grad(gray, 'difference')
    mod, direction = grad_dirmod(gradx, grady)
    hist = build_hist(mod, direction, cell_rows, cell_cols, bin_count)
    features = merge_blocks(hist, block_row_cells, block_col_cells)
    return features


def fit_and_classify(x_train, y_train, x_test):
    clf = svm.SVC(kernel='linear', C=0.5)

    clf.fit(x_train, y_train)
    return clf.predict(x_test)
