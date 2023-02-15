import os
import cv2
import numpy
import matplotlib.cm as cm
import pandas as pd
from matplotlib import colors
from scipy.spatial import ConvexHull
from skimage.draw import disk, line_aa,ellipse,circle_perimeter_aa
from PIL import Image, ImageDraw,ImageFont,ImageColor
import matplotlib.pyplot as plt
from tabulate import tabulate
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
color_black = (0, 0, 0)
color_white = (255, 255, 255)
color_gray = (64, 64, 64)
color_light_gray = (180,180,180)
color_bright_red = (0, 32, 255)
color_red   = (0, 0, 200)
color_amber = (0, 128, 255)
color_coral = (0, 90, 255)
color_aqua    = (180,200,0)
color_diamond = (120,200,0)
color_green   = (0,200,0)
color_grass   = (0,200,140)
color_marsala = (0,0,180)
color_gold = (0,200,255)
color_grass_dark = (63, 77, 73)
color_grass_dark2 = (97, 111, 107)
color_blue = (255, 128, 0)
# ----------------------------------------------------------------------------------------------------------------------
cuboid_lines_idx1 = [(1,0),(0,2),(2,3),(3,1), (7,6),(6,4),(4,5),(5,7), (1,0),(0,6),(6,7),(7,1), (3,2),(2,4),(4,5),(5,3)]
cuboid_lines_idx2 = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,1),(1,5),(5,4),(4,0), (2,3),(3,7),(7,6),(6,2)]
# ----------------------------------------------------------------------------------------------------------------------
def gre2jet(rgb):
    return cv2.applyColorMap(numpy.array(rgb, dtype=numpy.uint8).reshape((1, 1, 3)), cv2.COLORMAP_JET).reshape(3)
# ----------------------------------------------------------------------------------------------------------------------
def gre2viridis(rgb):
    colormap = numpy.flip((numpy.array(cm.cmaps_listed['viridis'].colors) * 256).astype(int), axis=1)
    return colormap[rgb[0]]
# ----------------------------------------------------------------------------------------------------------------------
def draw_points(image, points,color=(0,0,200),w=4,transperency=0,put_text=False,labels=None):

    H, W = image.shape[:2]
    pImage = Image.fromarray(image)
    draw = ImageDraw.Draw(pImage, 'RGBA') if len((image.shape))==3 else ImageDraw.Draw(pImage)

    for i,p in enumerate(points):
        if p is None: continue
        if numpy.any(numpy.isnan(p)): continue
        clr = color if (len(numpy.array(color).shape) == 1) or type(color) == int else color[i].tolist()
        draw.ellipse((p[0]-w//2,p[1]-w//2,p[0]+1+w//2,p[1]+1+w//2),fill=(clr[0], clr[1], clr[2], int(255 - transperency * 255)),outline=None)

    result = numpy.array(pImage)

    for i,p in enumerate(points):
        if p is None: continue
        if numpy.any(numpy.isnan(p)): continue
        clr = color if (len(numpy.array(color).shape) == 1) or type(color) == int else color[i].tolist()
        if put_text:
            cv2.putText(result, '%d %d'%(p[0],p[1]), (min(W - 10, max(10, p[0])), min(H - 5, max(10, p[1]))),cv2.FONT_HERSHEY_SIMPLEX, 1*w/12, clr, 1, cv2.LINE_AA)

        if labels is not None:
            draw_text(result, labels[i], (min(W - 10, max(10, p[0])), min(H - 5, max(10, p[1]))), clr, clr_bg=None, font_size=16, alpha_transp=0)
            #cv2.putText(result, '{0}'.format(labels[i]), (min(W - 10, max(10, p[0])), min(H - 5, max(10, p[1]))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1, cv2.LINE_AA)

    del draw
    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_line(array_bgr, row1, col1, row2, col2, color_bgr, alpha_transp=0.0,antialiasing=True):
    res_rgb = array_bgr.copy()

    if antialiasing:
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
    else:
        pImage = Image.fromarray(array_bgr)
        draw = ImageDraw.Draw(pImage, 'RGBA') if len((array_bgr.shape)) == 3 else ImageDraw.Draw(pImage)
        draw.line(((int(col1), int(row1)), (int(col2), int(row2))), fill=color_bgr)
        res_rgb = numpy.array(pImage)

    return res_rgb
# ----------------------------------------------------------------------------------------------------------------------
def draw_lines(image, lines,color=(0,0,200),w=1,transperency=0,antialiasing=True):

    result = image.copy()
    if lines is None or len(lines)==0:
        return result

    if (not isinstance(lines,list)) and len(lines.shape)==1:
        lines = [lines]

    if antialiasing:
        result = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line
            result = draw_line(result, int(y1), int(x1), int(y2), int(x2), color, alpha_transp=0.0, antialiasing=True)

    else:

        pImage = Image.fromarray(image)
        draw = ImageDraw.Draw(pImage, 'RGBA') if len((image.shape)) == 3 else ImageDraw.Draw(pImage)

        for id,line in enumerate(lines):
            if line is None:continue
            x1, y1, x2, y2 = line
            if numpy.any(numpy.isnan((x1, y1, x2, y2))): continue
            clr = color
            clr = (clr[0], clr[1], clr[2], int(255 - transperency * 255))
            draw.line(((int(x1), int(y1)), (int(x2), int(y2))), fill=clr,width=w)

        result = numpy.array(pImage)

    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_rect(image, col_left, row_up, col_right, row_down ,color, w=1, alpha_transp=0.8,font_size=16,label=None):

    lines = numpy.array(((col_left, row_up, col_left, row_down),(col_left, row_down, col_right, row_down),(col_right, row_down, col_right, row_up),(col_right, row_up,col_left,row_up)))
    points_2d = numpy.array(((col_left, row_up),(col_left, row_down),(col_right, row_down),(col_right, row_up)))
    result = draw_convex_hull(image, points_2d, color, transperency=alpha_transp)
    result = draw_lines(result, lines, color=color, w=w)

    if label is not None:
        result = draw_text(result,label,(col_left,row_up), color_fg=(0,0,0),clr_bg=color,font_size=font_size)

    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_rects(image, rects, color, w=2, alpha_transp=0.8):
    for r in rects:
        image = draw_rect(image,r[0,0], r[0,1], r[1,0], r[1,1], color, w, alpha_transp)

    return image
# ----------------------------------------------------------------------------------------------------------------------
def draw_text(image,label,xy, color_fg,clr_bg=None,font_size=16,alpha_transp=0):
    if label=='us':
        ii=0

    x,y = xy[0],xy[1]
    pImage = Image.fromarray(image)
    draw = ImageDraw.Draw(pImage, 'RGBA')

    if os.name in ['nt']:
        #font = ImageFont.truetype("calibri.ttf", size=font_size, encoding="utf-8")
        font = ImageFont.truetype("UbuntuMono-R.ttf", size=font_size, encoding="utf-8")
    else:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=font_size, encoding="utf-8")

    total_display_str_height = 1.1 * (font.getsize(label)[1])
    text_bottom = y if y > total_display_str_height else y + total_display_str_height

    clr = (numpy.array(color_fg)*(1-alpha_transp)+numpy.array(clr_bg)*(alpha_transp)).astype(numpy.int) if clr_bg is not None else numpy.array(color_fg).astype(numpy.int)

    text_width, text_height = font.getsize(label)
    margin = numpy.ceil(0.05 * text_height)
    if clr_bg is not None:
        draw.rectangle([(x, text_bottom - text_height - 2 * margin), (x + text_width, text_bottom)], fill=(clr_bg[0], clr_bg[1], clr_bg[2]))

    draw.text((x + margin, text_bottom - text_height - margin), label, fill=(clr[0],clr[1],clr[2]), font=font)
    result = numpy.array(pImage)

    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_mat(M, posx, posy, image,color=(128, 128, 0),text=None):

    df = pd.DataFrame(M)
    if df.shape[1]==1:
        df=df.T
    for c in df.columns:
        V = numpy.array(['%+1.2f' % v for v in df[c]]).astype(numpy.str)
        df[c] = pd.Series(V).astype(numpy.str)

    string1 = tabulate(df,headers=[],showindex=False,tablefmt='plsql',disable_numparse=True)
    string1 = string1.split('\n')[1: -1]
    if text is not None:
        string1[0]=  string1[0] + ' ' + text

    string1 = '\n'.join(string1)


    image_res = draw_text(image,string1, (posx, posy), color_fg=color, font_size=16)
    return image_res
# ----------------------------------------------------------------------------------------------------------------------
def draw_cuboid(image, points_2d, idx_mode = 1, color=(255, 255, 255), w=1,idx_face = [0,1,2,3]):

    lines_idx = cuboid_lines_idx1 if idx_mode==1 else cuboid_lines_idx2
    lines = [numpy.array((points_2d[i[0]], points_2d[i[1]])).flatten() for i in lines_idx]

    result = image.copy()
    result = draw_convex_hull(result, points_2d, color, transperency=0.90)
    result = draw_convex_hull(result, points_2d[idx_face], color, transperency=0.80)
    result = draw_lines(result, numpy.array(lines), color, w)
    result = draw_lines(result, numpy.array(lines)[idx_face], color, w+1)

    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_contours(image, points, color=(255,255,255),w=1,transperency=0.0):

    idx = numpy.arange(0,points.shape[0])
    lines = numpy.array([[p1[0],p1[1],p2[0],p2[1]] for p1,p2 in zip(points,points[numpy.roll(idx,1)])])
    image_res = draw_contours_cv(image, points, color, w=-1, transperency=transperency)
    image_res = draw_lines(image_res, lines, color, w)

    return image_res
# ----------------------------------------------------------------------------------------------------------------------
def draw_contours_cv(image, points, color=(255,255,255),w=-1,transperency=0.0):

    pnts = points.reshape(1, -1, 1, 2).astype(numpy.int)
    res = image.copy()
    color = (int(color[0]), int(color[1]), int(color[2]))
    res = cv2.drawContours(res, pnts, -1,color,thickness=w)

    if transperency>0:
        res = res*(1-transperency)+image*(transperency)

    return res.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
def draw_convex_hull_cv(image, points, color=(255,255,255),transperency=0.0):

    hull = ConvexHull(numpy.array(points,dtype=numpy.int))
    cntrs = numpy.array(points,dtype=numpy.int)[hull.vertices].reshape(1,-1, 1, 2).astype(numpy.int)
    res = cv2.drawContours(image.copy(), cntrs,-1,color,thickness=-1)

    if transperency>0:
        res = numpy.add(res*(1-transperency),image*(transperency))

    return res.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
def draw_convex_hull(image,points,color=(255, 0, 0),transperency=0.0):

    if points[:,0].max()-points[:,0].min()<1:return image
    if points[:,1].max()-points[:,1].min()<1:return image

    pImage = Image.fromarray(image)
    draw = ImageDraw.Draw(pImage, 'RGBA')

    try:
        hull = ConvexHull(numpy.array(points, dtype=numpy.int))
        cntrs = numpy.array(points, dtype=numpy.int)[hull.vertices]
        polypoints = [(point[0],point[1]) for point in cntrs]
        draw.polygon(polypoints, (color[0], color[1], color[2], int(255-transperency * 255)))
        result = numpy.array(pImage)
        del draw
    except:
        return image
    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_circle(array_bgr, row, col, rad, color_brg, alpha_transp=0):
    color_brg = numpy.array(color_brg)
    res_rgb = array_bgr.copy()
    if alpha_transp > 0:
        res_rgb[disk((int(row), int(col)), int(rad), shape=array_bgr.shape)] = array_bgr[disk((int(row), int(col)), int(rad), shape=array_bgr.shape)] * alpha_transp + color_brg * (1 - alpha_transp)
    else:
        res_rgb[disk((int(row), int(col)), int(rad), shape=array_bgr.shape)] = color_brg


    return res_rgb
# ----------------------------------------------------------------------------------------------------------------------
def draw_circle_aa(image,row, col, radius,color_brg, clr_bg,alpha_transp=0):
    image_res = image.copy()
    cc, rr, val = circle_perimeter_aa(int(col), int(row), int(radius))

    if alpha_transp > 0:
        color_brg = numpy.array(color_brg)*(1-alpha_transp)+numpy.array(clr_bg)*(alpha_transp)

    idx = numpy.array(cc>0)*numpy.array(rr>0)*numpy.array(cc<image.shape[1])*numpy.array(rr<image.shape[0])
    image_res[rr[idx], cc[idx], 0] = val[idx] * color_brg[0] + (1-val[idx]) * clr_bg[0]
    image_res[rr[idx], cc[idx], 1] = val[idx] * color_brg[1] + (1-val[idx]) * clr_bg[1]
    image_res[rr[idx], cc[idx], 2] = val[idx] * color_brg[2] + (1-val[idx]) * clr_bg[2]

    return image_res
# ----------------------------------------------------------------------------------------------------------------------
def draw_ellipse0(array_bgr, row, col, r_radius, c_radius, color_brg, alpha_transp=0):
    color_brg = numpy.array(color_brg)
    res_rgb = array_bgr.copy()
    if alpha_transp > 0:
        res_rgb[ellipse(int(row), int(col), int(r_radius),int(c_radius), shape=array_bgr.shape)] = array_bgr[ellipse(int(row), int(col), int(r_radius),int(c_radius), shape=array_bgr.shape)] * alpha_transp + color_brg * (1 - alpha_transp)
    else:
        res_rgb[ellipse(int(row), int(col), int(r_radius),int(c_radius), shape=array_bgr.shape)] = color_brg
    return res_rgb
# ----------------------------------------------------------------------------------------------------------------------
def draw_ellipse(image,p,color=(255, 0, 0),col_edge=(200, 0, 0),transperency=0.0):

    if p is None or len(p)==0 or numpy.any(numpy.isnan(p)):
        return image

    pImage = Image.fromarray(image)
    if len((image.shape))==3:
        draw = ImageDraw.Draw(pImage, 'RGBA')
        clr_fill = (color[0], color[1], color[2], int(255-transperency*255)) if color is not None else None
        clr_edge = (col_edge[0], col_edge[1], col_edge[2],int(255-transperency*255)) if col_edge is not None else None
        draw.ellipse((int(p[0]),int(p[1]),int(p[2]),int(p[3])), fill=clr_fill,outline= clr_edge)
    else:
        draw = ImageDraw.Draw(pImage)
        draw.ellipse((int(p[0]), int(p[1]), int(p[2]), int(p[3])),fill=color,outline=color)

    result = numpy.array(pImage)
    del draw
    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_ellipses(image, ellipses,color=(255,255,255),w=4,draw_axes=False,labels=None):
    image_ellipses = image.copy()

    for id, ellipse in enumerate(ellipses):
        if ellipse is None: continue
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        rotation_angle = ellipse[2]
        cv2.ellipse(image_ellipses, center, axes, rotation_angle, startAngle=0, endAngle=360, color=color,thickness=w)
        if draw_axes:
            draw_ellipse_axes(image_ellipses, ellipse, color=color, w=1)

        if labels is not None:
            cv2.putText(image_ellipses, '{0}'.format(labels[id]), (center[0]+20,center[1]+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    return image_ellipses
# ----------------------------------------------------------------------------------------------------------------------
def draw_ellipse_axes(image, ellipse, color=(255, 255, 255), w=4):
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
    rotation_angle = ellipse[2]*numpy.pi/180

    ct = numpy.cos(rotation_angle)
    st = numpy.sin(rotation_angle)

    p1 = (int(center[0] - axes[0]*ct), int(center[1] - axes[0] * st))
    p2 = (int(center[0] + axes[0]*ct), int(center[1] + axes[0] * st))

    p3 = (int(center[0] - axes[1] * st), int(center[1] + axes[1] * ct))
    p4 = (int(center[0] + axes[1] * st), int(center[1] - axes[1] * ct))
    cv2.line(image,p1,p2,color=color,thickness=w)
    cv2.line(image,p3,p4,color=color, thickness=w)

    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_segments(image, segments,color=(0,0,200),w=1,put_text=False):

    result = image.copy()
    H, W = image.shape[:2]
    if len(segments)==0:return result

    for id, segment in enumerate(segments):
        if len(numpy.array(color).shape) == 1:
            clr = color
        else:
            clr = color[id].tolist()

        if segment is None: continue
        if numpy.any(numpy.isnan(segment)): continue

        for point in segment:
            if w == 1:
                if 0<=point[1]<H and 0<=point[0]<W:result[int(point[1]), int(point[0])]=clr
            else:
                cv2.circle(result, (int(point[0]), int(point[1])), radius=w, color=clr,thickness=-1)

        if put_text:
            x, y = int(segment[:,0].mean()+ -30+60 * numpy.random.rand()) , int(segment[:,1].mean()-30+60 * numpy.random.rand())
            cv2.putText(result, '{0}'.format(id), (min(W - 10, max(10, x)), min(H - 5, max(10, y))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1, cv2.LINE_AA)
    return result
# ----------------------------------------------------------------------------------------------------------------------
def extend_view(XY,H,W,factor = 4):

    if XY is None: return XY

    if isinstance(XY, tuple):
        XY = [numpy.array(XY)]

    if (not isinstance(XY,list)) and len(XY.shape) == 1:
        XY = [XY]

    result = []
    for xy in XY:
        if (xy is None) or numpy.any(numpy.isnan(xy)) or len(xy)==0:
            result.append(xy)
        else:
            r = numpy.array(xy,dtype=numpy.float32)

            if len(xy.flatten()) == 2:
                ix = 0
                iy = 1
            else:
                ix = [0, 2]
                iy = [1, 3]

            r[ix] /= factor
            r[ix] += (W - W / factor) / 2
            r[iy] /= factor
            r[iy] += (H - H / factor) / 2
            result.append(r)

    result = numpy.array(result)

    return result
# ----------------------------------------------------------------------------------------------------------------------
def extend_view_from_image(image,factor=4,color_bg=(32,32,32)):
    H, W = image.shape[:2]

    image_result = numpy.full((H,W,3),color_bg,dtype=numpy.uint8)
    image_small = cv2.resize(tools_image.saturate(image),(int(W/factor),int(H/factor)))
    image_result = tools_image.put_image(image_result,image_small,int((H-H/factor)/2),int((W-W/factor)/2))


    return image_result
# ----------------------------------------------------------------------------------------------------------------------
def get_position_sign(sign,W, H,font_size,pos=None):

    if pos is None:
        pos = (int(10), int(H // 2))

    image = numpy.zeros((H, W, 3), dtype=numpy.uint8)
    image = draw_text(image, sign, pos, (255, 255, 255), (0, 0, 0),font_size=int(font_size))
    A = numpy.max(image[:, :, 0], axis=0)
    NZ = numpy.nonzero(A)
    min_x = NZ[0].min()
    max_x = NZ[0].max()

    A = numpy.max(image[:, :, 0], axis=1)
    NZ = numpy.nonzero(A)
    min_y = NZ[0].min()
    max_y = NZ[0].max()

    sx = max_x - min_x
    sy = max_y - min_y

    shift_x = pos[0] -sx/2 - min_x
    shift_y = pos[1]  -sy/2 - min_y
    # image_small = image[min_y:max_y+1, min_x:max_x+1]
    # cv2.imwrite('./folder_out/'+'small.png',image_small)

    return shift_x, shift_y, sx, sy
# ---------------------------------------------------------------------------------------------------------------------
def blend(col1_255,col2_255,alpha):
    c1 = numpy.array(col1_255).astype(numpy.uint8)
    c2 = numpy.array(col2_255).astype(numpy.uint8)
    res = cv2.addWeighted(c1.reshape((1,1,3)), alpha, c2.reshape((1,1,3)), 1-alpha, 0)
    return res[0,0]
# ----------------------------------------------------------------------------------------------------------------------
def interpolate_points_by_curve(points_xy,N=100,trans=False):
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    if trans: y, x = x, y
    f = numpy.poly1d(numpy.polyfit(x, y, 2))
    x_new = numpy.linspace(x[0], x[-1], num=N, endpoint=True)
    y_new = f(x_new)
    if trans:y_new,x_new = x_new,y_new
    points = numpy.concatenate([x_new.reshape(-1, 1), y_new.reshape(-1, 1)], axis=1)
    lines  = numpy.concatenate([points[1:], points[:-1]], axis=1)
    return lines
# ----------------------------------------------------------------------------------------------------------------------
def get_colors(N, shuffle = False,colormap = 'jet',interpolate=True,alpha_blend=None,clr_blend=(255,255,255)):

    N_orig = N
    if N == 1:
        N_orig,N = 1,2

    do_inv = False
    if (isinstance(colormap,str)) and ('~' in colormap):
        colormap=colormap[1:]
        do_inv = True

    if colormap=='rainbow':
        colors = [int(180 * i / (N - 1)) for i in range(N)]
        colors = [cv2.cvtColor(numpy.array([c,255,255], dtype=numpy.uint8).reshape((1, 1, 3)), cv2.COLOR_HSV2BGR)[0,0] for c in colors]

    elif colormap == 'jet':
        colors = numpy.array([(int(255 * i / (N - 1)), int(255 * i / (N - 1)), int(255 * i / (N - 1))) for i in range(N)])
        colors = [gre2jet(c) for c in colors]

    elif colormap=='viridis':
        colors = numpy.array([(int(255 * i / (N - 1)), int(255 * i / (N - 1)), int(255 * i / (N - 1))) for i in range(N)])
        colors = [gre2viridis(c) for c in colors]

    elif colormap == 'gray':
        colors = numpy.array([(int(255 * i / (N - 1)), int(255 * i / (N - 1)), int(255 * i / (N - 1))) for i in range(N)])

    elif colormap == 'warm':
        colors = get_colors_warm(N, dark_mode=False)

    elif colormap=='cool':
        colors = get_colors_cool(N, dark_mode=False)
    else:
        #clrs_base = 255*numpy.array(plt.get_cmap(colormap).colors)[:,[2, 1, 0]]
        clrs_base = 255*numpy.array([plt.get_cmap(colormap)(i) for i in range(plt.get_cmap(colormap).N)])[:,[2, 1, 0]]

        if interpolate:
            colors = [clrs_base[int(i*(clrs_base.shape[0]-1)/(N-1))] for i in range(N)]
        else:
            colors = [clrs_base[i%(clrs_base.shape[0])] for i in range(N)]

    if alpha_blend is not None:
        colors = [((alpha_blend) * numpy.array(clr_blend) + (1 - alpha_blend) * numpy.array(color)) for color in colors]

    colors = numpy.array(colors, dtype=numpy.uint8)

    if shuffle:
        numpy.random.seed(1024)
        idx = numpy.random.choice(len(colors), len(colors))
        colors = colors[idx]

    colors = colors[:N_orig]
    if do_inv:
        colors=colors[::-1]

    return colors
# ----------------------------------------------------------------------------------------------------------------------
def get_colors_warm(N,dark_mode=False):

    if dark_mode:
        colors_warm = get_colors(256, colormap='YlOrRd', alpha_blend=0.2,clr_blend=(0, 0, 0), shuffle=False)
    else:
        colors_warm = get_colors(256, colormap='YlOrRd', alpha_blend=0.0, clr_blend=(0, 0, 0), shuffle=False)

    res_colors = numpy.array([colors_warm[int(i)] for i in numpy.linspace(64,192, N)])
    return res_colors
# ----------------------------------------------------------------------------------------------------------------------
def get_colors_cool(N,dark_mode=False):

    if dark_mode:
        c1 = get_colors(256, colormap='Blues', alpha_blend=0.2, clr_blend=(0, 0, 0), shuffle=False)
        c2 = get_colors(256, colormap='Greens', alpha_blend=0.2, clr_blend=(0, 0, 0),shuffle=False)
    else:
        c1 = get_colors(256, colormap='Blues', alpha_blend=0.0, clr_blend=(255, 255, 255), shuffle=False)
        c2 = get_colors(256, colormap='Greens', alpha_blend=0.0, clr_blend=(255, 255, 255),shuffle=False)

    colors_cool = (6.0 * c1 + 4.0 * c2)[::-1] / 10

    res_colors = numpy.array([colors_cool[int(i)] for i in numpy.linspace(64, 192, N)])
    return res_colors
# ---------------------------------------------------------------------------------------------------------------------
def get_colors_custom():
    # map_RdBu = [cm.RdBu(i) for i in numpy.linspace(0, 1, 15)]
    # map_PuOr = [cm.PuOr(i) for i in numpy.linspace(0, 1, 15)]
    # my_cmap = map_RdBu[7:][::-1] + map_PuOr[:7][::-1]
    # my_cmap = colors.ListedColormap(my_cmap)

    #
    map_c = [plt.cm.cividis(i) for i in numpy.linspace(0, 1, 15)]
    my_cmap = colors.ListedColormap(map_c)



    # my_cmap2 = ['#0057b8','#2E598A','#5C5C5C','#808080','#C0AC40','#ffd700']
    # my_cmap2 = ['#0057b8','#0B57AD','#1558A3','#29598F','#3D5A7B','#515B67','#5B5C5D','#82817E','#908B70','#AC9F54','#C8B238','#E4C51C','#F2CF0E','#FFD700']
    # my_cmap2 = [BGR_from_HTML(c)/255.0 for c in my_cmap2]
    # my_cmap = colors.ListedColormap(my_cmap2)
    return my_cmap
# ---------------------------------------------------------------------------------------------------------------------
def BGR_from_HTML(color_HTML='#000000'):
    col = numpy.array(ImageColor.getcolor(color_HTML, "RGB"))
    return col
# ---------------------------------------------------------------------------------------------------------------------
def BGR_to_HTML(bgr):
    return ('#%02x%02x%02x' % (int(bgr[0]),int(bgr[1]),int(bgr[2]))).upper()
# ---------------------------------------------------------------------------------------------------------------------
def create_color_discrete_map(df, col_label, col_color=None, palette='RdBu'):

    if col_color is None:
        L = df[col_label].unique().tolist()
        colors = get_colors(len(L), colormap=palette, shuffle=False, interpolate=False)
        colors_HTML = [BGR_to_HTML(c[[2,1,0]]) for c in colors]
        color_discrete_map = dict(zip(L, colors_HTML))
    else:
        colors255 = get_colors(255, colormap=palette, shuffle=False)
        colors_HTML = [BGR_to_HTML(c[[2, 1, 0]]) for c in colors255]
        #df_agg = tools_DF.my_agg(df, cols_groupby=[col_label], cols_value=[col_color], aggs=['top'])
        df_agg = df.groupby(col_label, dropna=False).agg({col_color:'top'})

        ccc = [(label, colors_HTML[int(col_id)]) for label, col_id in zip(df_agg.iloc[:, 0], df_agg.iloc[:, 1])]
        color_discrete_map = dict(ccc)

    return color_discrete_map
# ----------------------------------------------------------------------------------------------------------------------
def values_to_colors(values,palette='viridis'):
    values_scaled = values - values.min()
    values_scaled /= values_scaled.max() / 255
    col255 = get_colors(256, shuffle=False, colormap=palette)
    colors = [col255[int(v)] if not numpy.isnan(v) else col255[0] for v in values_scaled]
    return  colors
# ----------------------------------------------------------------------------------------------------------------------