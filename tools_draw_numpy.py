import cv2
import numpy
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
from skimage.draw import circle, line_aa,ellipse
from PIL import Image, ImageDraw
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
def gre2jet(rgb):
    return cv2.applyColorMap(numpy.array(rgb, dtype=numpy.uint8).reshape((1, 1, 3)), cv2.COLORMAP_JET).reshape(3)
# ----------------------------------------------------------------------------------------------------------------------
def gre2viridis(rgb):
    colormap = numpy.flip((numpy.array(cm.cmaps_listed['viridis'].colors) * 256).astype(int), axis=1)
    return colormap[rgb[0]]
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
def draw_circle(array_bgr, row, col, rad, color_brg, alpha_transp=0):
    color_brg = numpy.array(color_brg)
    res_rgb = array_bgr.copy()
    if alpha_transp > 0:
        res_rgb[circle(int(row), int(col), int(rad), shape=array_bgr.shape)] = array_bgr[circle(int(row), int(col), int(rad), shape=array_bgr.shape)] * alpha_transp + color_brg * (1 - alpha_transp)
    else:
        res_rgb[circle(int(row), int(col), int(rad), shape=array_bgr.shape)] = color_brg
    return res_rgb
# ----------------------------------------------------------------------------------------------------------------------
def draw_line(array_bgr, row1, col1, row2, col2, color_bgr, alpha_transp=0.0):
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
def draw_contours_cv(image, points, color=(255,255,255),transperency=0.0):

    pnts = points.reshape(1, -1, 1, 2).astype(numpy.int)
    res = image.copy()
    res = cv2.drawContours(res, pnts, -1,color,thickness=-1)

    if transperency>0:
        res = res*(1-transperency)+image*(transperency)

    return res.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
def draw_contours(image, points, color_fill=(255,255,255),color_outline=(255,255,255),transp_fill=0.0,transp_outline=0.0):

    pImage = Image.fromarray(image)
    draw = ImageDraw.Draw(pImage, 'RGBA')

    polypoints = [(point[0],point[1]) for point in points]
    if color_fill is not None:
        clr_fill   = (color_fill[0], color_fill[1], color_fill[2], int(255 - int(255 - transp_fill * 255)))
    else:
        clr_fill = None

    if color_outline is not None:
        clr_outline= (color_outline[0], color_outline[1], color_outline[2], int(255 - int(255 - transp_outline * 255)))
    else:
        clr_outline=None

    draw.polygon(polypoints, fill=clr_fill,outline=clr_outline)


    result = numpy.array(pImage)
    del draw
    return result
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
def draw_rectangle(image,p1,p2,color=(255, 0, 0),transperency=0.0):

    pImage = Image.fromarray(image)
    if len((image.shape)) == 3:
        draw = ImageDraw.Draw(pImage, 'RGBA')
        draw.rectangle((p1,p2), fill=(color[0], color[1], color[2], int(transperency * 255)),outline= (color[0], color[1], color[2],255))
    else:
        draw = ImageDraw.Draw(pImage)
        draw.rectangle((p1, p2))

    result = numpy.array(pImage)
    del draw
    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_ellipse(image,p,color=(255, 0, 0),transperency=0.0):

    if p is None or len(p)==0:
        return image

    pImage = Image.fromarray(image)
    if len((image.shape))==3:
        draw = ImageDraw.Draw(pImage, 'RGBA')
        draw.ellipse((int(p[0]),int(p[1]),int(p[2]),int(p[3])), fill=(color[0], color[1], color[2], int(255-transperency*255)),outline= (color[0], color[1], color[2],255))
    else:
        draw = ImageDraw.Draw(pImage)
        draw.ellipse((int(p[0]), int(p[1]), int(p[2]), int(p[3])),fill=color,outline=color)

    result = numpy.array(pImage)
    del draw
    return result
# ----------------------------------------------------------------------------------------------------------------------
def get_colors(N, shuffle = False,colormap = 'jet',alpha_blend=None):
    colors = []
    if N==1:
        colors.append(numpy.array([255, 0, 0]))
        return colors

    if colormap=='rainbow':
        colors = [int(180 * i / (N - 1)) for i in range(N)]
        colors = [cv2.cvtColor(numpy.array([c,255,255], dtype=numpy.uint8).reshape((1, 1, 3)), cv2.COLOR_HSV2BGR)[0,0] for c in colors]

    elif colormap == 'jet':
        colors = numpy.array([(int(255 * i / (N - 1)), int(255 * i / (N - 1)), int(255 * i / (N - 1))) for i in range(N)])
        colors = [gre2jet(c) for c in colors]

    elif colormap=='viridis':
        colors = numpy.array([(int(255 * i / (N - 1)), int(255 * i / (N - 1)), int(255 * i / (N - 1))) for i in range(N)])
        colors = [gre2viridis(c) for c in colors]

    else:
        colors = numpy.array([(int(255 * i / (N - 1)), int(255 * i / (N - 1)), int(255 * i / (N - 1))) for i in range(N)])

    if alpha_blend is not None:
        colors = [((alpha_blend) * numpy.array((255, 255, 255)) + (1 - alpha_blend) * numpy.array(color)) for color in colors]

    colors = numpy.array(colors, dtype=numpy.uint8)

    if shuffle:
        numpy.random.seed(1024)
        idx = numpy.random.choice(len(colors), len(colors))
        colors = colors[idx]
    return numpy.array(colors)
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
def draw_ellipses(image, ellipses,color=(255,255,255),w=4,draw_axes=False):
    image_ellipses = image.copy()

    for ellipse in ellipses:
        if ellipse is None: continue
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        rotation_angle = ellipse[2]
        cv2.ellipse(image_ellipses, center, axes, rotation_angle, startAngle=0, endAngle=360, color=color,thickness=w)
        if draw_axes:
            draw_ellipse_axes(image_ellipses, ellipse, color=color, w=1)

    return image_ellipses
# ----------------------------------------------------------------------------------------------------------------------
def draw_segments(image, segments,color=(255,255,255),w=1,put_text=False):

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
def draw_lines(image, lines,color=(255,255,255),w=4,put_text=False):

    result = image.copy()
    if lines is None or len(lines)==0:
        return result

    if (not isinstance(lines,list)) and len(lines.shape)==1:
        lines = [lines]

    H, W = image.shape[:2]
    for id,line in enumerate(lines):
        if line is None:continue
        (x1, y1, x2, y2) = line
        if numpy.any(numpy.isnan((x1, y1, x2, y2))): continue

        if (len(numpy.array(color).shape)==1) or type(color)==int:
            clr = color
        else:
            clr = color[id].tolist()

        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), clr, w)
        if put_text:
            x,y = int(x1+x2)//2, int(y1+y2)//2
            cv2.putText(result, '{0}'.format(id),(min(W - 10, max(10, x)), min(H - 5, max(10, y))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1, cv2.LINE_AA)


    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_points(image, points,color=(255,255,255),w=4,put_text=False,labels=None):

    result = image.copy()
    H, W = image.shape[:2]
    for id, point in enumerate(points):
        if point is None: continue

        if numpy.any(numpy.isnan(point)): continue
        x1, y1 = int(point[0]), int(point[1])

        if (len(numpy.array(color).shape) == 1) or type(color) == int:
            clr = color
        else:
            clr = color[id].tolist()

        cv2.circle(result, (int(x1), int(y1)), w, clr,thickness=-1)
        if put_text:
            cv2.putText(result, '{0}'.format(id), (min(W - 10, max(10, x1)), min(H - 5, max(10, y1))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1, cv2.LINE_AA)

        if labels is not None:
            cv2.putText(result, '{0}'.format(labels[id]), (min(W - 10, max(10, x1)), min(H - 5, max(10, y1))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1, cv2.LINE_AA)

    return result
# ----------------------------------------------------------------------------------------------------------------------
def draw_signals(signals,lanes=None,colors=None,w=3):

    height = 255

    image_signal = numpy.full((height, len(signals[0]), 3), 32, dtype=numpy.uint8)
    if colors is None:
        colors = get_colors(len(signals),alpha_blend=0.1)

    for signal,color in zip(signals,colors):
        if signal is None: continue
        for col, value in enumerate(signal):
            if value==45:
                print(color)
            if not numpy.isnan(value):
                if w>1:
                    cv2.circle(image_signal, (col, height - int(value)), radius=w, color=color.tolist(),thickness=-1)
                else:
                    cv2.line(image_signal,(col, height - int(value)),(col, height - int(value)),color=color.tolist(),thickness=1)

    if lanes is not None:
        if lanes[0] is not None:
            cv2.line(image_signal, (lanes[0], 0), (lanes[0], height), color=(255, 255, 255), thickness=2)
        for p in lanes[1:]:
            if p is not None:
                cv2.line(image_signal, (p,0),(p,height), color=(180,180,180), thickness=1)

    return image_signal
# ----------------------------------------------------------------------------------------------------------------------
def draw_signals_v2(signals,lanes,colors,w=3):

    height = 512

    image_signal = numpy.full((height, len(signals[0]), 3), 32, dtype=numpy.uint8)

    for signal in signals:
        if signal is None: continue
        for col, value in enumerate(signal):
            if value>height:
                value=height-1

            if not numpy.isnan(value):
                color = colors[value%len(colors)].tolist()
                if w>1:
                    cv2.circle(image_signal, (col, height - int(value)), radius=w, color=color,thickness=-1)
                else:
                    cv2.line(image_signal,(col, height - int(value)),(col, height - int(value)),color=color,thickness=1)

    if lanes is not None:
        for p in lanes:
            if p is not None:
                cv2.line(image_signal, (p,0),(p,height), color=(64,64,64), thickness=1)

    return image_signal
# ----------------------------------------------------------------------------------------------------------------------
def draw_signals_lines(signals,colors=None,w=3):

    height = 255

    image_signal = numpy.full((height, len(signals[0]), 3), 32, dtype=numpy.uint8)
    if colors is None:
        colors = get_colors(len(signals),alpha_blend=0.0)

    for signal,color in zip(signals,colors):
        if signal is None: continue
        for col in range(len(signal)-1):
            value = signal[col]
            value2 = signal[col+1]

            if not numpy.isnan(value) and not numpy.isnan(value2):
                cv2.line(image_signal,(col, height - int(value)),(col+1, height - int(value2)),color=color.tolist(),thickness=w)

    return image_signal
# ----------------------------------------------------------------------------------------------------------------------
def draw_cuboid(image, points, color=(255, 255, 255), w=2, put_text=False, labels=None):
    lines = []
    for i in [(1, 0), (0, 2), (2, 3), (3, 1), (7, 6), (6, 4), (4, 5), (5, 7), (1, 0), (0, 6), (6, 7), (7, 1), (3, 2),(2, 4), (4, 5), (5, 3), (1, 3), (3, 5), (5, 7), (7, 1), (0, 2), (2, 4), (4, 6), (6, 0)]:
        lines.append(numpy.array((points[i[0]], points[i[1]])).flatten())

    idx_face = [0,1,2,3]

    result = draw_convex_hull(image, points, color, transperency=0.80)
    result = draw_convex_hull(result, points[idx_face], color, transperency=0.70)
    result = draw_lines(result, numpy.array(lines), color, w)
    if put_text:
        result = draw_points(result, points, color,put_text=put_text)

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

    image_result = numpy.full((H,W,3),0,dtype=numpy.uint8)
    image_result[:,:]=color_bg

    image_small = cv2.resize(tools_image.saturate(image),(int(W/factor),int(H/factor)))

    image_result = tools_image.put_image(image_result,image_small,int((H-H/factor)/2),int((W-W/factor)/2))



    return image_result
# ----------------------------------------------------------------------------------------------------------------------