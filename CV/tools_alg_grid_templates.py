#----------------------------------------------------------------------------------------------------------------------
import os
import numpy
import math
#from photutils import find_peaks as find_peaks_2d
#from astropy.stats import sigma_clipped_stats
#from skimage.feature import peak_local_max
#from skimage.measure import compare_ssim
from scipy.stats import  entropy
from scipy import signal
import cv2
#from scipy.misc import toimage
from scipy import ndimage
from scipy.signal import fftconvolve
#----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_draw_numpy
import tools_alg_match
import tools_image
#----------------------------------------------------------------------------------------------------------------------
def get_hits(image_base, image_pattern,grid=0.1,path_debug=''):
    hitmap_2d = tools_alg_match.calc_hit_field(image_base, image_pattern)
    coordinates, chains_hor, chains_ver = detect_peaks_2d(image_base,image_pattern,hitmap_2d, grid=0.05,path_debug=path_debug)
    return coordinates,chains_hor, chains_ver,hitmap_2d, 0
# ----------------------------------------------------------------------------------------------------------------------
def find_local_peaks(hitmap_2d, row_step, col_step):

    R_steps = 1+int(hitmap_2d.shape[0] / row_step)
    C_steps = 1+int(hitmap_2d.shape[1] / col_step)
    margin = 2

    coord = numpy.zeros((R_steps * C_steps, 2), int)
    R_shift = (hitmap_2d.shape[0] - R_steps*row_step)/2
    C_shift = (hitmap_2d.shape[1] - C_steps*col_step)/2

    for row in range(0, R_steps):
        for col in range(0, C_steps):
            k = C_steps * row + col
            coord[k, 0] = R_shift + row * row_step
            coord[k, 1] = C_shift + col * col_step


    for row in range(0, R_steps):
        for col in range(0, C_steps):
            k = C_steps * row + col
            left   = coord[k, 0] - int(col_step / 2) - margin
            right  = coord[k, 0] - int(col_step / 2) + col_step + margin
            top    = coord[k, 1] - int(row_step / 2) - margin
            bottom = coord[k, 1] - int(row_step / 2) + row_step + margin

            left   = numpy.maximum(left, 0)
            right  = numpy.minimum(right, hitmap_2d.shape[0])
            top    = numpy.maximum(top, 0)
            bottom = numpy.minimum(bottom, hitmap_2d.shape[1])

            field = hitmap_2d[left:right, top:bottom]
            c = numpy.argmax(field)
            x, y = numpy.divmod(c, field.shape[1])
            if not (x==0 or x==field.shape[0]-1 or y==0 or y==field.shape[1]-1):
                coord[k, 0], coord[k, 1] = x + left, y + top
            else:
                coord[k, 0], coord[k, 1] = 0,0


    coord = numpy.unique(coord, axis=0)[1:,:]

    coord = filter_unique(coord,hitmap_2d,(row_step+col_step)/2)

    return coord
#----------------------------------------------------------------------------------------------------------------------
def find_local_peaks_greedy(hitmap_2d, row_step, col_step, min_value= 150):

    coord_pattern = []
    field  = hitmap_2d.copy()
    field2 = numpy.full(hitmap_2d.shape,255)

    value = numpy.max(field)

    while value > min_value:
        row, col = numpy.divmod(numpy.argmax(field), field.shape[1])

        left   = numpy.maximum(col - int(col_step / 2),0)
        right  = numpy.minimum(col - int(col_step / 2) + col_step,field.shape[1])
        top    = numpy.maximum(row - int(row_step / 2),0)
        bottom = numpy.minimum(row - int(row_step / 2) + row_step,field.shape[0])

        cnt = numpy.count_nonzero(field[top:bottom,left:right])
        if cnt >  row_step*col_step*0.25:
            coord_pattern.append([row, col])

        field[top:bottom,left:right]=0
        field2[top:bottom, left:right] = 0
        value = numpy.max(field)

    #cv2.imwrite('../_pal/Ex02_out/'+'field2.png',field2)
    misses  = ndimage.uniform_filter(field2, (int(row_step*0.75), int(col_step*0.75)), mode='constant')
    misses_rgb = numpy.zeros((hitmap_2d.shape[0],hitmap_2d.shape[1],3),dtype=numpy.uint8)
    misses_rgb[misses==255]=(0,0,255)
    #cv2.imwrite('../_pal/Ex02_out/' + 'misses_rgb.png', misses_rgb)

    return numpy.array(coord_pattern), misses_rgb
# ----------------------------------------------------------------------------------------------------------------------
def filter_unique(coord,hitmap_2d,tol):

    idx = []
    for i in range (0,coord.shape[0]-1):
        for j in range(i+1, coord.shape[0]):
            v = coord[i, :] - coord[j, :]
            dst = numpy.linalg.norm(v)

            if dst<tol:
                if hitmap_2d[coord[i,0],coord[i,1]]>hitmap_2d[coord[j,0],coord[j,1]]:
                    idx.append(j)
                else:
                    idx.append(i)

    idx_inv= numpy.array([x for x in range(0, coord.shape[0]) if x not in idx])

    result = coord[idx_inv]

    return result
#----------------------------------------------------------------------------------------------------------------------
def filter_high_peaks(hitmap_2d,coord,top=0.5):

    v = []
    for i in range(0,coord.shape[0]):
        v.append(hitmap_2d[coord[i,0],coord[i,1]])

    min=numpy.min(v)
    max = numpy.max(v)

    idx=[]
    for i in range(0,coord.shape[0]):
        if( (hitmap_2d[coord[i,0],coord[i,1]]-min)/(max-min) > top ):
            idx.append(i)


    return coord[idx]
#----------------------------------------------------------------------------------------------------------------------
def are_hor_nbr(row1,col1,row2,col2,tolerance_angle,max_nbr_distance):
    if abs(row1-row2)<=tolerance_angle*abs(col1-col2) and abs(col1-col2)<=max_nbr_distance:
        return 1
    return 0
# ----------------------------------------------------------------------------------------------------------------------
def are_ver_nbr(row1,col1,row2,col2,tolerance_angle,max_nbr_distance):
    if abs(col1-col2)<=tolerance_angle*abs(row1-row2) and abs(row1-row2)<=max_nbr_distance:
        return 1
    return 0
#----------------------------------------------------------------------------------------------------------------------
def get_nbr(coord,tolerance_angle,max_nbr_distance):
    nbr_hor = numpy.zeros((coord.shape[0], coord.shape[0])).astype(int)
    nbr_ver = numpy.zeros((coord.shape[0], coord.shape[0])).astype(int)
    for i in range(0, coord.shape[0]):
        for j in range(i+1, coord.shape[0]):
            nbr_hor[i, j] = are_hor_nbr(coord[i,0],coord[i,1],coord[j,0],coord[j,1],tolerance_angle,max_nbr_distance)
            nbr_hor[j, i] = nbr_hor[i,j]
            nbr_ver[i, j] = are_ver_nbr(coord[i, 0], coord[i, 1], coord[j, 0], coord[j, 1],tolerance_angle,max_nbr_distance)
            nbr_ver[j, i] = nbr_ver[i, j]
    return nbr_hor,nbr_ver
#----------------------------------------------------------------------------------------------------------------------
def intersect_lines(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    v1 = (bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)
    v2 = (bx2 - bx1) * (ay2 - by1) - (by2 - by1) * (ax2 - bx1)
    v3 = (ax2 - ax1) * (by1 - ay1) - (ay2 - ay1) * (bx1 - ax1)
    v4 = (ax2 - ax1) * (by2 - ay1) - (ay2 - ay1) * (bx2 - ax1)
    if (numpy.sign(v1)!=numpy.sign(v2)) and(numpy.sign(v3)!=numpy.sign(v4)):
        return True
    else:
        return False
#----------------------------------------------------------------------------------------------------------------------
def interceсt_chains(coord,all_chains, idx1,idx2):

    res=False

    r11 = coord[idx1, 0]
    c11 = coord[idx1, 1]
    r12 = coord[idx2, 0]
    c12 = coord[idx2, 1]

    for each in all_chains:
        if res==False:
            for i in range(0,len(each)-1):
                if res==False:
                    r21 = coord[each[i  ],0]
                    c21 = coord[each[i  ],1]
                    r22 = coord[each[i+1],0]
                    c22 = coord[each[i+1],1]
                    res = intersect_lines( r11,c11,r12,c12,r21,c21,r22,c22)

    return res
#----------------------------------------------------------------------------------------------------------------------
def get_best_chain(coord,hitmap_2d, Nbr,sort_mode,all_chains):

    if (coord.size==0):
        return []

    L=coord.shape[0]


    if sort_mode=='h':
        idx = numpy.lexsort((coord[:, 0], coord[:, 1]))
    else:
        idx = numpy.lexsort((coord[:, 1], coord[:, 0]))

    # search best path ->
    SR = numpy.zeros(L)
    dirR = numpy.full(L, -1)
    SR[L - 1] = hitmap_2d[coord[idx[L - 1],0],coord[idx[L - 1],1]]

    for l in reversed(range(0, L - 1)):
        maxS = 0
        for ll in range(l + 1, L):
            if (Nbr[idx[l], idx[ll]] == 1 and SR[ll] > maxS and interceсt_chains(coord, all_chains,idx[l],idx[ll])==0):
                maxS = SR[ll]
                dirR[l] = ll
        SR[l] = maxS + hitmap_2d[coord[idx[l],0],coord[idx[l],1]]

    # search best path <-
    SL = numpy.zeros(L)
    dirL = numpy.full(L, -1)
    SL[0] = hitmap_2d[coord[idx[0],0],coord[idx[0],1]]

    for l in range(1, L):
        maxS = 0
        for ll in range(0, l):
            if (Nbr[idx[l]][idx[ll]] == 1 and SL[ll] > maxS and interceсt_chains(coord, all_chains,idx[l],idx[ll])==0):
                maxS = SL[ll]
                dirL[l] = ll

        SL[l] = maxS + hitmap_2d[coord[idx[l],0],coord[idx[l],1]]

    for ll in range(0, L):
        SL[ll] += SR[ll] - hitmap_2d[coord[idx[ll],0],coord[idx[ll],1]]

    res=[]
    l = ll = l_best = numpy.argmax(SL, axis=0)
    res.append(idx[ll])
    while (dirR[l] != -1):
        l = dirR[l]
        res.append(idx[l])


    l = ll = l_best
    while (dirL[l] != -1):
        l = dirL[l]
        res.append(idx[l])


    return res
#----------------------------------------------------------------------------------------------------------------------
def get_chains(coord,hitmap_2d, nbr_mat,sort_mode):

    if coord.shape[0]<=1:
        return[]

    all_chains=[]
    pos=[]
    idx = get_best_chain(coord,hitmap_2d,nbr_mat,sort_mode,all_chains)
    while len(idx)>=3:
        all_chains.append(idx)
        rr = coord[idx, 0]
        cc = coord[idx, 1]
        if (sort_mode=='h'):
            pos.append(rr[numpy.argmin(cc)])
        else:
            pos.append(cc[numpy.argmin(rr)])

        nbr_mat[idx,:] = 0
        nbr_mat[:,idx] = 0
        idx = get_best_chain(coord,hitmap_2d, nbr_mat,sort_mode,all_chains)

    all_chains_sorted=[all_chains[x] for x in numpy.argsort(pos)]

    return all_chains_sorted
#----------------------------------------------------------------------------------------------------------------------
def get_chains_hor_vert(coord, hitmap_2d,tolerance_angle, max_nbr_distance):

    if (coord.size==0):
        return numpy.array([]),numpy.array([])

    nbr_hor, nbr_ver = get_nbr(coord, tolerance_angle, max_nbr_distance)
    chains_hor = get_chains(coord, hitmap_2d, nbr_hor, sort_mode='h')
    chains_ver = get_chains(coord, hitmap_2d, nbr_ver, sort_mode='v')

    return chains_hor,chains_ver
# ----------------------------------------------------------------------------------------------------------------------
def filter_isolated_chains(chains_hor,chains_ver):

    flag=1
    while flag==1:
        flag=0

        all_hor = numpy.concatenate(chains_hor).ravel()
        all_ver = numpy.concatenate(chains_ver).ravel()

        for each in chains_hor:
            res = []
            for vertex in each:
                idx = numpy.array([i for i, v in enumerate(all_ver) if (v == vertex)])
                res.append(len(idx))
            if numpy.sum(res)<=3:
                chains_hor.remove(each)
                flag=1

        for each in chains_ver:
            res = []
            for vertex in each:
                idx = numpy.array([i for i, v in enumerate(all_hor) if (v == vertex)])
                res.append(len(idx))
            if numpy.sum(res)<=3:
                chains_ver.remove(each)
                flag=1

        if (len(chains_hor)==0) or (len(chains_ver)==0):
            flag=0

    return chains_hor,chains_ver
#----------------------------------------------------------------------------------------------------------------------
def keep_fully_connected_vertices(coord,chains_hor,chains_ver):

    all_hor = numpy.concatenate(chains_hor).ravel()
    all_ver = numpy.concatenate(chains_ver).ravel()
    idx=[]

    for i in range(0, coord.shape[0]):
        idx_h = numpy.array([c for c, v in enumerate(all_hor) if (v == i)])
        idx_v = numpy.array([c for c, v in enumerate(all_ver) if (v == i)])
        if len(idx_h)>0 and len(idx_v)>0:
            idx.append(i)

    idx=numpy.array(idx)

    return coord[idx]
#----------------------------------------------------------------------------------------------------------------------
def keep_vertices_in_chains(coord,chains_hor,chains_ver):

    all_hor = numpy.concatenate(chains_hor).ravel()
    all_ver = numpy.concatenate(chains_ver).ravel()
    idx=[]

    for i in range(0, coord.shape[0]):
        idx_h = numpy.array([c for c, v in enumerate(all_hor) if (v == i)])
        idx_v = numpy.array([c for c, v in enumerate(all_ver) if (v == i)])
        if len(idx_h)>0 or len(idx_v)>0:
            idx.append(i)

    idx=numpy.array(idx)

    return coord[idx]
#----------------------------------------------------------------------------------------------------------------------
def are_chains_aligned(ch1,ch2,chains):

    matches = 0
    for chain in chains:
        list1 = list(set(ch1).intersection(chain))
        list2 = list(set(ch2).intersection(chain))

        for each1 in list1:
            for each2 in list2:
                idx1 = numpy.array([i for i, v in enumerate(chain) if (v == each1)])
                idx2 = numpy.array([i for i, v in enumerate(chain) if (v == each2)])
                if abs(idx2-idx1)<=2:
                    matches += 1

    if(matches>=3):
        return True

    return False
#----------------------------------------------------------------------------------------------------------------------
def filter_aligned_chains(chains_hor, chains_ver):

    flag = 1
    while flag==1:
        flag=0

        algn_hor = numpy.zeros((len(chains_hor), len(chains_hor))).astype(int)
        for i in range (0,len(chains_hor)-1):
            for j in range(i+1, len(chains_hor)):
                algn_hor[i,j] = are_chains_aligned(chains_hor[i], chains_hor[j], chains_ver)
                algn_hor[j,i] = algn_hor[i,j]

        chains_hor_new, chains_ver_new =[], []

        for i in range(0, len(chains_hor)):
            if numpy.sum(algn_hor[i,:])>0:
                chains_hor_new.append(chains_hor[i])


        algn_ver = numpy.zeros((len(chains_ver), len(chains_ver))).astype(int)
        for i in range (0,len(chains_ver)-1):
            for j in range(i+1, len(chains_ver)):
                algn_ver[i,j] = are_chains_aligned(chains_ver[i], chains_ver[j], chains_hor)
                algn_ver[j,i] = algn_ver[i,j]

        for i in range(0, len(chains_ver)):
            if numpy.sum(algn_ver[i,:])>0:
                chains_ver_new.append(chains_ver[i])

        if(len(chains_hor_new)!=len(chains_hor)) or (len(chains_ver_new)!=len(chains_ver)):
            flag=1

        chains_hor = chains_hor_new
        chains_ver = chains_ver_new


    return chains_hor, chains_ver
# ----------------------------------------------------------------------------------------------------------------------
def filter_grid(coord,hitmap_2d,tolerance_angle,max_nbr_distance):

    if (coord.size==0):
        return numpy.array([])

    any_wiped=True

    while any_wiped==True:
        coord0 = coord.copy()

        chains_hor, chains_ver = get_chains_hor_vert(coord, hitmap_2d,tolerance_angle, max_nbr_distance)
        coord = keep_vertices_in_chains(coord, chains_hor, chains_ver)

        chains_hor, chains_ver = get_chains_hor_vert(coord, hitmap_2d, tolerance_angle, max_nbr_distance)
        coord = keep_fully_connected_vertices(coord, chains_hor, chains_ver)

        any_wiped = (coord0.shape[0] != coord.shape[0])
        #any_wiped = False

    return numpy.array(coord)
#----------------------------------------------------------------------------------------------------------------------
def detect_peaks_2d(image_base,image_pattern,hitmap_2d,grid=0.1,path_debug=''):

    if path_debug!='':
        path_outliers_debug=path_debug+'outliers/'
    else:
        path_outliers_debug=''

    row_step = int(hitmap_2d.shape[0]*grid)
    col_step = int(hitmap_2d.shape[1]*grid)

    coord = find_local_peaks(hitmap_2d, row_step, col_step)
    coord = filter_high_peaks(hitmap_2d, coord,top=0.3)

    #coord = filter_outliers(image_base, coord,image_pattern,2.0,path_outliers_debug)

    coord =                          filter_grid(coord, hitmap_2d,tolerance_angle=1.0/5.0,max_nbr_distance=row_step*120)
    chains_hor, chains_ver = get_chains_hor_vert(coord, hitmap_2d,tolerance_angle=1.0/5.0,max_nbr_distance=row_step*120)

    return coord, chains_hor, chains_ver
#----------------------------------------------------------------------------------------------------------------------
def detect_peaks_1d(hitmap_1d, grid=0.1):
    hitmap_1d = hitmap_1d.astype(numpy.float)
    peaks = signal.find_peaks_cwt(hitmap_1d,numpy.arange(1,int(1/grid)))
    return peaks
#----------------------------------------------------------------------------------------------------------------------
def visualize_matches(image, disp_row,disp_col,disp_v1, disp_v2, disp_h1, disp_h2):
    res=image.copy()
    image_disp_hor = numpy.zeros((image.shape[0],image.shape[1])).astype(numpy.float32)
    image_disp_hor [:,:]=disp_col[:,:]

    for row in range(0,image.shape[0]):
        for col in range(0, image.shape[1]):
            r = row + disp_row[row, col]
            c = col + disp_col[row, col]
            r= numpy.maximum(0,numpy.minimum(image.shape[0]-1,r))
            c= numpy.maximum(0,numpy.minimum(image.shape[1]-1,c))
            res[row,col] =image[r,c]

    image_disp_hor=(image_disp_hor-disp_h1)*255/(disp_h2-disp_h1)
    image_disp_hor_RGB_jet = tools_image.hitmap2d_to_jet(image_disp_hor.astype(numpy.uint8))

    return res,image_disp_hor_RGB_jet
# ----------------------------------------------------------------------------------------------------------------------
def mse(im1_gray, im2_gray):

    if len(im1_gray.shape)!=2:
        im1_gray = cv2.cvtColor(im1_gray, cv2.COLOR_BGR2GRAY)

    if len(im2_gray.shape)!=2:
        im2_gray = cv2.cvtColor(im2_gray, cv2.COLOR_BGR2GRAY)

    a=numpy.zeros(im1_gray.shape).astype(numpy.float32)
    a+=im2_gray
    a-=im1_gray
    avg = numpy.average(a)
    a-=avg

    #d=numpy.sum((a)**2)
    #d/=float(im1_gray.shape[0] * im1_gray.shape[1])
    #d=math.sqrt(d)
    th = 64
    x = numpy.where(a>th)
    d2= 255.0*len(x[0])/(im1_gray.shape[0] * im1_gray.shape[1])
    return d2
# ----------------------------------------------------------------------------------------------------------------------
def align_images(im1, im2,mode = cv2.MOTION_AFFINE):

    if len(im1.shape) == 2:
        im1_gray = im1.copy()
        im2_gray = im2.copy()

    else:
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    #mode = cv2.MOTION_TRANSLATION
    #mode = cv2.MOTION_AFFINE

    try:
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, numpy.eye(2, 3, dtype=numpy.float32),mode)
    except:
        return im1, im2

    if len(im1.shape)==2:
        aligned = cv2.warpAffine(im2_gray, warp_matrix, (im2_gray.shape[1], im2_gray.shape[0]),borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return im1_gray, aligned
    else:
        aligned = cv2.warpAffine(im2, warp_matrix, (im2_gray.shape[1], im2_gray.shape[0]),borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return im1, aligned
# ----------------------------------------------------------------------------------------------------------------------
def ssim(im1, im2, k=(0.01, 0.03), l=255):

    if len(im1.shape)!=2:
        im1= cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    if len(im2.shape)!=2:
        im2= cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    win = numpy.array([gaussian(11, 1.5)])
    window = win * (win.T)
    """See https://ece.uwaterloo.ca/~z70wang/research/ssim/"""
    # Check if the window is smaller than the scenes.
    for a, b in zip(window.shape, im1.shape):
        if a > b:
            return None, None
    # Values in k must be positive according to the base implementation.
    for ki in k:
        if ki < 0:
            return None, None

    c1 = (k[0] * l) ** 2
    c2 = (k[1] * l) ** 2
    window = window/numpy.sum(window)

    mu1 = fftconvolve(im1, window, mode='valid')
    mu2 = fftconvolve(im2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = fftconvolve(im1 * im1, window, mode='valid') - mu1_sq
    sigma2_sq = fftconvolve(im2 * im2, window, mode='valid') - mu2_sq
    sigma12 = fftconvolve(im1 * im2, window, mode='valid') - mu1_mu2

    if c1 > 0 and c2 > 0:
        num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / den
    else:
        num1 = 2 * mu1_mu2 + c1
        num2 = 2 * sigma12 + c2
        den1 = mu1_sq + mu2_sq + c1
        den2 = sigma1_sq + sigma2_sq + c2
        ssim_map = numpy.ones(numpy.shape(mu1))
        index = (den1 * den2) > 0
        ssim_map[index] = (num1[index] * num2[index]) / (den1[index] * den2[index])
        index = (den1 != 0) & (den2 == 0)
        ssim_map[index] = num1[index] / den1[index]

    mssim = ssim_map.mean()
    return int(100*mssim)#, ssim_map
# ----------------------------------------------------------------------------------------------------------------------
def coordinates_to_images(image_base, coordinates, dR, dC):
    images=[]

    dR=int(dR)
    dC=int(dC)
    for i in range(0, coordinates.shape[0]):
        row = coordinates[i][0]
        col = coordinates[i][1]

        left = col - int(dC / 2)
        right = col - int(dC / 2) + dC
        top = row - int(dR / 2)
        bottom = row - int(dR / 2) + dR

        if left>0 and top>0 and bottom<image_base.shape[0] and right<image_base.shape[1]:
            cut = tools_image.crop_image(image_base, top,left,bottom,right,extrapolate_border=True)
            images.append(cut)


    return numpy.array(images)
# ----------------------------------------------------------------------------------------------------------------------
def calc_matches(quality_4d,disp_v1, disp_v2, disp_h1, disp_h2):

    quality_3d = numpy.max(quality_4d, axis=3)
    disp_row = numpy.argmax(quality_3d,axis=2)+disp_v1

    quality_3d = numpy.max(quality_4d, axis=2)
    disp_col = numpy.argmax(quality_3d, axis=2)+disp_h1

    return disp_row,disp_col
# ----------------------------------------------------------------------------------------------------------------------
def filter_outliers(image_base, coordinates,pattern,th=1.0,path_debug=''):

    if (coordinates.size == 0):
        return numpy.array([])

    if path_debug != '' and (not os.path.exists(path_debug)):
        os.makedirs(path_debug)
    else:
        tools_IO.remove_files(path_debug)

    images = coordinates_to_images(image_base, coordinates, pattern.shape[0], pattern.shape[1])

    idx=[]
    for i in range(0, images.shape[0]):

        pattern    , im_aligned      = align_images(pattern,images[i],cv2.MOTION_AFFINE)
        value = mse(pattern, im_aligned)
        #value = ssim(pattern, im_aligned)
        if value<=th:
            idx.append(i)

        if path_debug != '':
            toimage(im_aligned).save_model(path_debug + '%03d_%03d.bmp' % (value, i))

    return coordinates[idx]
# ----------------------------------------------------------------------------------------------------------------------
def estimate_pattern_position(filename_in,folder_out,pattern_width,pattern_height):

    scale = 4.0

    image_base = cv2.cvtColor(cv2.imread(filename_in), cv2.COLOR_RGB2BGR)
    image_base_scaled = cv2.resize(image_base,(int(image_base.shape[1]/scale),int(image_base.shape[0]/scale)) )

    tools_IO.remove_files(folder_out)
    for shift_row in range (0,pattern_height):
        for shift_col in range(0, pattern_width):
            row = int(image_base.shape[0]/2)+shift_row
            col = int(image_base.shape[1]/2)+shift_col
            image_pattern = image_base[int(row-pattern_height/2):int(row+pattern_height/2),int(col-pattern_width/2):int(col+pattern_width/2), :]
            image_pattern_scaled = cv2.resize(image_pattern,(int(image_pattern.shape[1] / scale), int(image_pattern.shape[0] / scale)))


            hitmap_2d = tools_alg_match.calc_hit_field(image_base_scaled, image_pattern_scaled)
            hitmap_RGB_jet = tools_image.hitmap2d_to_jet(-hitmap_2d)

            e = entropy(hitmap_2d.flatten()/255.0)

            toimage(hitmap_RGB_jet).save_model(folder_out + ('hit_%1.3f_%03d_%03d.bmp' % (e, shift_row, shift_col)))
            toimage(image_pattern_scaled).save_model(folder_out + ('pat_%03d_%03d.bmp' % (shift_row, shift_col)))
    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_hits(path,coordinates,chains_hor, chains_ver,hitmap_2d):

    hitmap_RGB_gre, hitmap_RGB_jet= tools_image.hitmap2d_to_jet(hitmap_2d)

    for each in chains_hor:
        for i in range (0,len(each)-1):
            hitmap_RGB_gre = tools_draw_numpy.draw_line(hitmap_RGB_gre,coordinates[each[i]][0], coordinates[each[i]][1],coordinates[each[i+1]][0], coordinates[each[i+1]][1],[128,128,0])

    for each in chains_ver:
        for i in range (0,len(each)-1):
            hitmap_RGB_gre = tools_draw_numpy.draw_line(hitmap_RGB_gre,coordinates[each[i]][0], coordinates[each[i]][1],coordinates[each[i+1]][0], coordinates[each[i+1]][1],[128,128,0])

    for i in range(0, coordinates.shape[0]):
        hitmap_RGB_gre=tools_draw_numpy.draw_circle(hitmap_RGB_gre,coordinates[i][0],coordinates[i][1], 4,[255,32,0])

    return hitmap_RGB_gre,hitmap_RGB_jet
# ----------------------------------------------------------------------------------------------------------------------
