import numpy as np
from scipy.signal import convolve2d


def get_bayer_masks(n_row, n_col):
    res = np.full((n_row, n_col, 3), False)
    r_mask = np.array([[False, True],
                       [False, False]])
    g_mask = np.array([[True, False],
                       [False, True]])
    b_mask = np.array([[False, False],
                       [True, False]])
    main_full = np.dstack((np.tile(r_mask, (n_row // 2, n_col // 2)), np.tile(g_mask, (n_row // 2, n_col // 2)),
                          np.tile(b_mask, (n_row // 2, n_col // 2))))

    last_row = np.dstack((np.tile(np.array([False, True]), n_col // 2).reshape(-1),
                          np.tile(np.array([True, False]), n_col // 2).reshape(-1),
                          np.tile(np.array([False, False]), n_col // 2).reshape(-1)))

    last_col = np.dstack((np.tile(np.array([False, False]), n_row // 2).reshape(-1),
                          np.tile(np.array([True, False]), n_row // 2).reshape(-1),
                          np.tile(np.array([False, True]), n_row // 2).reshape(-1)))

    if n_row % 2 and n_col % 2:
        res[:n_row - 1, :n_col - 1, :] = main_full
        res[n_row - 1, :n_col - 1, :] = last_row
        res[:n_row - 1, n_col - 1, :] = last_col
        res[n_row - 1, n_col - 1, :] = np.array([False, True, False])
    elif n_row % 2:
        res[:n_row - 1, :, :] = main_full
        res[n_row - 1, :, :] = last_row
    elif n_col % 2:
        res[:, :n_col - 1, :] = main_full
        res[:, n_col - 1, :] = last_col
    else:
        res = main_full

    return res


def get_colored_img(raw_img):
    n_row, n_col = raw_img.shape
    masks = get_bayer_masks(n_row, n_col)
    img3 = np.dstack((raw_img, raw_img, raw_img))
    return img3 * masks


def bilinear_interpolation(colored_img):
    colored_img = colored_img.astype('float64')
    n_row, n_col, _ = colored_img.shape
    masks = get_bayer_masks(n_row, n_col)

    r_m = masks[:, :, 0]
    g_m = masks[:, :, 1]
    b_m = masks[:, :, 2]

    gb_gr = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]]) / 4.0
    br_rb = np.array([[1, 0, 1],
                      [0, 0, 0],
                      [1, 0, 1]]) / 4.0
    hor = np.array([[0, 0, 0],
                    [1, 0, 1],
                    [0, 0, 0]]) / 2.0
    vert = np.transpose(hor)

    r = colored_img[:, :, 0]
    g = colored_img[:, :, 1]
    b = colored_img[:, :, 2]

    arr = np.zeros_like(r)
    arr[::2, :] = 1.0

    rg1 = convolve2d(r, hor, mode='same')
    rg2 = convolve2d(r, vert, mode='same')
    bg1 = convolve2d(b, vert, mode='same')
    bg2 = convolve2d(b, hor, mode='same')

    g = np.where(np.logical_or(r_m, b_m), convolve2d(g, gb_gr, mode='same'), g)
    r = np.where(b_m, convolve2d(r, br_rb, mode='same'), r)
    b = np.where(r_m, convolve2d(b, br_rb, mode='same'), b)
    r = np.where(np.logical_and(g_m, arr == 1), rg1, r)
    r = np.where(np.logical_and(g_m, arr == 0), rg2, r)
    b = np.where(np.logical_and(g_m, arr == 1), bg1, b)
    b = np.where(np.logical_and(g_m, arr == 0), bg2, b)

    return np.clip(np.rint(np.dstack((r, g, b))), 0, 255).astype('uint8')


def improved_interpolation(raw_img):
    raw_img = raw_img.astype('float64')
    h = raw_img.shape[0]
    w = raw_img.shape[1]

    rg_rb_bg_br = (1 / 8) * np.array([
        [0, 0, 1/2, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 1/2, 0, 0]
    ])

    rg_br_bg_rb = np.transpose(rg_rb_bg_br)

    rb_bb_br_rr = (1 / 8) * np.array([
        [0, 0, -3/2, 0, 0],
        [0, 2, 0, 2, 0],
        [-3/2, 0, 6, 0, -3/2],
        [0, 2, 0, 2, 0],
        [0, 0, -3/2, 0, 0]
    ])

    grgb = (1 / 8) * np.array([
        [0, 0, -1, 0, 0],
        [0, 0, 2, 0, 0],
        [-1, 2, 4, 2, -1],
        [0, 0, 2, 0, 0],
        [0, 0, -1, 0, 0]
    ])
    masks = get_bayer_masks(h, w)

    mask_r = masks[:, :, 0]
    mask_g = masks[:, :, 1]
    mask_b = masks[:, :, 2]

    r = raw_img * mask_r
    g = raw_img * mask_g
    b = raw_img * mask_b

    del mask_g

    g = np.where(np.logical_or(mask_r, mask_b), convolve2d(raw_img, grgb, mode='same'), g)

    rbg_rbbr = convolve2d(raw_img, rg_rb_bg_br, mode='same')
    rbg_brrb = convolve2d(raw_img, rg_br_bg_rb, mode='same')
    rbgr_bbrr = convolve2d(raw_img, rb_bb_br_rr, mode='same')

    del grgb, rg_rb_bg_br, rg_br_bg_rb, rb_bb_br_rr

    r_r = np.transpose(np.any(mask_r == 1, axis=1)[np.newaxis]) * np.ones(r.shape)
    r_c = np.any(mask_r == 1, axis=0)[np.newaxis] * np.ones(r.shape)
    b_r = np.transpose(np.any(mask_b == 1, axis=1)[np.newaxis]) * np.ones(b.shape)
    b_c = np.any(mask_b == 1, axis=0)[np.newaxis] * np.ones(b.shape)

    del mask_r, mask_b

    r = np.where(np.logical_and(r_r == 1, b_c == 1), rbg_rbbr, r)
    r = np.where(np.logical_and(b_r == 1, r_c == 1), rbg_brrb, r)

    b = np.where(np.logical_and(b_r == 1, r_c == 1), rbg_rbbr, b)
    b = np.where(np.logical_and(r_r == 1, b_c == 1), rbg_brrb, b)

    r = np.where(np.logical_and(b_r == 1, b_c == 1), rbgr_bbrr, r)
    b = np.where(np.logical_and(r_r == 1, r_c == 1), rbgr_bbrr, b)

    del rbg_rbbr, rbg_brrb, rbgr_bbrr, r_r, r_c, b_r, b_c

    return np.clip(np.rint(np.dstack((r, g, b))), 0, 255).astype('uint8')


def compute_psnr(img_pred, img_gt):
    img_pred, img_gt = img_pred.astype('float64'), img_gt.astype('float64')
    h, w, c = img_pred.shape
    mse = np.sum((img_pred - img_gt) ** 2) / (h * w * c)
    if np.isclose(mse, 0):
        raise ValueError()
    return 10 * np.log10(np.max(img_gt ** 2) / mse)
