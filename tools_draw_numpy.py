import numpy
from skimage.draw import circle, line_aa


# ----------------------------------------------------------------------------------------------------------------------
def draw_circle(array_bgr, row, col, rad, color_brg, alpha_transp=0):
    color_brg = numpy.array(color_brg)
    res_rgb = array_bgr.copy()
    if alpha_transp > 0:
        res_rgb[circle(int(row), int(col), int(rad), shape=array_bgr.shape)] = array_bgr[circle(int(row), int(col), int(rad), shape=array_bgr.shape)] * alpha_transp + color_brg * (1 - alpha_transp)
    else:
        res_rgb[circle(int(row), int(col), int(rad), shape=array_bgr.shape)] = color_brg
    return res_rgb


# ----------------------------------------------------------------------------------------------------------------------
def draw_line(array_bgr, row1, col1, row2, col2, color_bgr, alpha_transp=0):
    res_rgb = array_bgr.copy()

    rr, cc, vv = line_aa(int(row1), int(col1), int(row2), int(col2))
    for i in range(0, rr.shape[0]):
        if (rr[i] >= 0 and rr[i] < array_bgr.shape[0] and cc[i] >= 0 and cc[i] < array_bgr.shape[1]):
            clr = numpy.array(color_bgr) * vv[i] + array_bgr[rr[i], cc[i]] * (1 - vv[i])
            if alpha_transp > 0:
                xxx = clr * (1 - alpha_transp)
                base = array_bgr[rr[i], cc[i]]
                xxx+= base * alpha_transp
                res_rgb[rr[i], cc[i]] = xxx
            else:
                res_rgb[rr[i], cc[i]] = clr
    return res_rgb


# ----------------------------------------------------------------------------------------------------------------------
def draw_rect(array_bgr, row_up, col_left, row_down, col_right, color_bgr, alpha_transp=0):
    res_rgb = array_bgr.copy()
    res_rgb = draw_line(res_rgb, row_up  , col_left , row_up  , col_right, color_bgr, alpha_transp)
    res_rgb = draw_line(res_rgb, row_up  , col_right, row_down, col_right, color_bgr, alpha_transp)
    res_rgb = draw_line(res_rgb, row_down, col_right, row_down, col_left , color_bgr, alpha_transp)
    res_rgb = draw_line(res_rgb, row_down, col_left , row_up  , col_left , color_bgr, alpha_transp)
    return res_rgb

# ----------------------------------------------------------------------------------------------------------------------
