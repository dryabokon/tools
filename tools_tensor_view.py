import os
import math
import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
def to_devidable0(n):
    success = False
    inc = -1
    max_inc = int(n*0.2)
    while (not success) and inc < max_inc:
        inc +=1
        candidate = n+inc
        nd1 = numerical_devisor(candidate)
        nd2 = candidate//nd1

        if (nd2*nd1 == candidate) and (nd1/nd2 > 0.3):
            success = True

    if not success:inc =0
    return n+inc
# ----------------------------------------------------------------------------------------------------------------------
def to_devidable(n,basis=1000):

    if n%basis==0:
        return n
    inc = basis-n%basis

    return int(n + inc)
# ----------------------------------------------------------------------------------------------------------------------
def numerical_devisor(n):

    for i in numpy.arange(int(math.sqrt(n))+1,1,-1):
        if n%i==0:
            return i

    return n
#--------------------------------------------------------------------------------------------------------------------------
def apply_blue_shift(image,mode=0):

    if (mode == 0):
        red, green, blue = 0.95, 0.95, 1.05
    else:
        red, green, blue = 1.05, 1.05, 0.95

    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            image[r,c, 0] = numpy.clip(image[r,c, 0] * blue, 0, 255)
            image[r,c, 1] = numpy.clip(image[r,c, 1] * green, 0, 255)
            image[r,c, 2] = numpy.clip(image[r,c, 2] * red, 0, 255)

    return
# ----------------------------------------------------------------------------------------------------------------------
def colorize_chess(image, W, H):

    if len(image.shape)==2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_color = image
    r, c = numpy.meshgrid(numpy.linspace(0, H - 1, num=H), numpy.linspace(0, W - 1, num=W))
    r, c = r.astype(numpy.int), c.astype(numpy.int)


    for R in numpy.arange(0, image.shape[0], H):
        for C in numpy.arange(0, image.shape[1], W):
            if ((int(R/H)+int(C/W))%2 == 0):
                red, green, blue = 0.95, 0.95, 1.05
            else:
                red, green, blue = 1.05, 1.05, 0.95
            image_color[R + r, C + c, 0] = numpy.clip(image_color[R + r, C + c, 0] * blue, 0, 255)
            image_color[R + r, C + c, 1] = numpy.clip(image_color[R + r, C + c, 1] * green, 0, 255)
            image_color[R + r, C + c, 2] = numpy.clip(image_color[R + r, C + c, 2] * red, 0, 255)

    return image_color
# ----------------------------------------------------------------------------------------------------------------------
def tensor_gray_1D_to_image(tensor,orientation = 'landscape'):

    rows = numerical_devisor(tensor.shape[0])
    cols = int(tensor.shape[0] / rows)

    if orientation=='landscape':
        image = numpy.reshape(tensor,(numpy.minimum(rows,cols),numpy.maximum(rows, cols)))
    else:
        image = numpy.reshape(tensor, (numpy.maximum(rows, cols), numpy.minimum(rows, cols)))


    return image
# ----------------------------------------------------------------------------------------------------------------------
def tensor_gray_3D_to_image(tensor, do_colorize = False,do_scale = False,force_rows_dim2 = None):

    if force_rows_dim2  is None:
        rows = numerical_devisor(tensor.shape[2])
    else:
        rows = force_rows_dim2

    h, w, R, C = tensor.shape[0], tensor.shape[1], rows, int(tensor.shape[2] / rows)
    image = numpy.zeros((h * R, w * C), dtype=numpy.float32)
    for i in range(0, tensor.shape[2]):
        col, row = i % C, int(i / C)
        image[h * row:h * row + h, w * col:w * col + w] = tensor[:, :, i]

    if do_colorize:
        image = colorize_chess(image,tensor.shape[0],tensor.shape[1])

    if do_scale==True:
        image -= image.min()
        image *= 255 / image.max()

    return image.astype(dtype=numpy.uint8)
# ---------------------------------------------------------------------------------------------------------------------
def image_to_tensor_color_4D(image,shape):
    tensor = numpy.zeros(shape,numpy.float32)

    h,w =shape[0],shape[1]
    rows = numerical_devisor(shape[3])
    C = int(tensor.shape[3] / rows)

    for i in range(0,96):
        col, row = i % C, int(i / C)
        tensor[:, :, :, i]=image[h * row:h * row + h, w * col:w * col + w]

    return tensor
# ---------------------------------------------------------------------------------------------------------------------
def tensor_color_4D_to_image(tensor,rows=None,cols=None):

    if rows is None and cols is not None:
        h, w, R, C = tensor.shape[0], tensor.shape[1], int(tensor.shape[3] / cols), cols
    elif rows is not None and cols is None:
        h, w, R, C = tensor.shape[0], tensor.shape[1], rows, int(tensor.shape[3] / rows)
    else:
        rows = numerical_devisor(tensor.shape[3])
        h, w, R, C = tensor.shape[0], tensor.shape[1], rows, int(tensor.shape[3] / rows)
        # cols = numerical_devisor(tensor.shape[3])
        # h, w, C, R = tensor.shape[0], tensor.shape[1], cols, int(tensor.shape[3] / cols)

    image = numpy.zeros((h * R,w * C, tensor.shape[2]),dtype=numpy.uint8)
    for i in range(0, tensor.shape[3]):
        col, row = i % C, int(i / C)
        image[h * row:h * row + h, w * col:w * col + w, :] = tensor[:, :, :, i]

    return image
# ---------------------------------------------------------------------------------------------------------------------
def tensor_gray_4D_to_image(tensor,do_colorize = False,do_scale=False,force_rows_dim3=None,force_rows_dim2=None):

    sub_image = tensor_gray_3D_to_image(tensor[:, :, :, 0],force_rows_dim2)
    h, w = sub_image.shape[0], sub_image.shape[1]

    if force_rows_dim3 is None:
        R = numerical_devisor(tensor.shape[3])
    else:
        R = force_rows_dim3

    C = int(tensor.shape[3]/R)


    if do_colorize:
        image = numpy.zeros((h * R, w * C, 3), dtype=numpy.float32)
    else:
        image = numpy.zeros((h * R, w * C), dtype=numpy.float32)
    for i in range (0,tensor.shape[3]):
        col, row = i % C, int(i / C)
        sub_image = tensor_gray_3D_to_image(tensor[:,:,:,i])

        if do_colorize:
            sub_image = cv2.cvtColor(sub_image, cv2.COLOR_GRAY2BGR)
            apply_blue_shift(sub_image,(col+row)%2)

        if do_colorize:
            image[h * row:h * row + h, w * col:w * col + w,:] = sub_image[:, :, :]
        else:
            image[h * row:h * row + h, w * col:w * col + w] = sub_image[:, :]

    if do_scale==True:
        image -= image.min()
        image *= 255 / image.max()

    return image
# ----------------------------------------------------------------------------------------------------------------------
def stack_frames(folder_in, filename_out, h=32, w=18,stride=1,basis=1000,columns=100,put_ID=False):

    filenames = numpy.array([folder_in + '/' + f for f in os.listdir(folder_in)])
    idx = numpy.arange(0,len(filenames),stride)
    filenames = filenames[idx]

    N = len(filenames)
    N = to_devidable(N,basis=basis)
    tensor_frames = numpy.zeros((h, w, 3, N), dtype=numpy.uint8)
    for i, filename in enumerate(filenames):
        image = cv2.resize(cv2.imread(filename), (w, h))
        if put_ID:
            m = image.mean()
            if m>128:
                clr = (0,0,0)
            else:
                clr = (255,255,255)
            cv2.putText(image, '{0}'.format(i),(5,10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, clr, 1, cv2.LINE_AA)

        tensor_frames[:, :, :, i] = image

    cv2.imwrite(filename_out, tensor_color_4D_to_image(tensor_frames,cols=columns))
    return
# ---------------------------------------------------------------------------------------------------------------------
def fill_range_color(tensor, the_range, color, limit, do_add=False,borders_only=False):
    color = numpy.array(color,dtype=numpy.uint8)

    for start, stop in the_range:
        if borders_only:
            if start<limit:
                tensor[:, :5, :, int(start)] = color
            if stop<limit:
                tensor[:,-5:, :, int(stop)] = color
        else:
            for i in range(int(start), int(stop+1)):
                if i < limit:
                    if do_add:
                        tensor[:, :, :, i]+= color
                    else:
                        tensor[:, :, :, i] = color

    return
# ---------------------------------------------------------------------------------------------------------------------
def fill_empties(tensor):
    step = 8
    clr_bg = 72
    clr_fg = 90


    image_na = numpy.full((tensor.shape[0], tensor.shape[1], 3), clr_bg, dtype=numpy.uint8)
    R = tensor.shape[0]
    for c in numpy.arange(-tensor.shape[1],tensor.shape[1],step):
        cv2.line(image_na,(c,0),(c+R,0+R),(clr_fg,clr_fg,clr_fg),1)

    for i in range(tensor.shape[3]):
        if tensor[:, :, :, i].max()==0:
            tensor[:, :, :, i] = image_na

    return
# ---------------------------------------------------------------------------------------------------------------------
def fill_ids(tensor, stride=1):
    h,w = tensor.shape[0],tensor.shape[1]

    for i in range(tensor.shape[3]):
        image = tensor[:, :, :, i].copy()
        clr = image[h // 2, w // 2, :]
        if clr.sum() > 128 * 3:
            clr = (0.8 * clr + 0.2 * numpy.array((0, 0, 0)))
        else:
            clr = (0.8 * clr + 0.2 * numpy.array((255, 255, 255)))

        cv2.putText(image, '{0}'.format(i * stride), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(int(clr[0]), int(clr[1]), int(clr[2])), 1, cv2.LINE_AA)
        tensor[:, :, :, i] = image
    return
# ---------------------------------------------------------------------------------------------------------------------