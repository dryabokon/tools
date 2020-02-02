import cupy as numpy
#--------------------------------------------------------------------------------------------------------------------------
def sliding_2d(A,h,w,stat='avg',mode='reflect'):

    B = numpy.pad(A,(h,w),mode)
    B = numpy.roll(B, 1, axis=0)
    B = numpy.roll(B, 1, axis=1)

    C1 = numpy.cumsum(B , axis=0)
    C2 = numpy.cumsum(C1, axis=1)

    up = numpy.roll(C2, h, axis=0)
    S1 = numpy.roll(up, w, axis=1)
    S2 = numpy.roll(up,-w, axis=1)

    dn = numpy.roll(C2,-h, axis=0)
    S3 = numpy.roll(dn, w, axis=1)
    S4 = numpy.roll(dn, -w, axis=1)

    if stat=='avg':
        R = (S1-S2-S3+S4)/((2*w)*(2*h))
    else:
        R = (S1 - S2 - S3 + S4)

    return R[h:-h,w:-w]
#--------------------------------------------------------------------------------------------------------------------------
def do_blend(large,small,mask):

    if len(mask.shape)==2:
        mask = numpy.array([mask,mask,mask]).transpose([1,2,0])

    if mask.max()>1:
        mask = (mask.astype(numpy.float)/255.0)

    background = numpy.multiply(mask,large)
    background = numpy.clip(background, 0, 255)

    foreground = numpy.multiply(1 - mask, small)
    foreground = numpy.clip(foreground, 0, 255)

    result = numpy.add(background, foreground)
    result = numpy.clip(result, 0, 255)

    result = numpy.array(result).astype(numpy.uint8)
    return result
#----------------------------------------------------------------------------------------------------------------------
def blend_multi_band_large_small(large, small, background_color=(255, 255, 255), adjust_colors=None, filter_size=50,  do_debug=False):

    large = numpy.array(large)


    mask_original = numpy.array(1*(small[:, :] == background_color))

    small = numpy.array(small)
    mask_bin = mask_original.copy()
    mask_bin = numpy.array(numpy.min(mask_bin,axis=2))

    mask = sliding_2d(mask_bin,filter_size//2,filter_size//2,'avg')

    mask = numpy.clip(2 * mask, 0, 1.0)

    if adjust_colors is not None:
        large = large.astype(numpy.float)
        small = small.astype(numpy.float)

        cnt_small = sliding_2d(1-mask_bin,filter_size,filter_size,'cnt')
        for c in range(3):

            avg_large = sliding_2d(large[:, :, c],filter_size,filter_size,'avg')
            sum_small = sliding_2d(small[:, :, c],filter_size,filter_size,'cnt')
            avg_small = sum_small/cnt_small


            scale = avg_large/avg_small
            scale = numpy.nan_to_num(scale)
            small[:,:,c]=small[:,:,c]*scale

    mask = numpy.stack((mask,mask,mask),axis=2)
    result = do_blend(large,small,mask)
    return result
#----------------------------------------------------------------------------------------------------------------------
def asnumpy(A):
    return numpy.asnumpy(A)
#----------------------------------------------------------------------------------------------------------------------