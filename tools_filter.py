# ----------------------------------------------------------------------------------------------------------------------
import cv2
import numpy
from filterpy.kalman import KalmanFilter
import pykalman
from scipy.signal import medfilt
import scipy.interpolate
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
def from_fistorical(L):

    cnt = numpy.count_nonzero(L)
    xx = float(cnt/(L.shape[0]*L.shape[1]*L.shape[2]))
    if  xx<0.98:
        return L[0]

    x = L[:, :, 0]
    y = L[:, :, 1]

    res = []

    for i in range(x.shape[1]):
        #x_filtered = do_filter_kalman(numpy.flip(x[:,i]))
        #y_filtered = do_filter_kalman(numpy.flip(y[:,i]))
        x_filtered = do_filter_average(numpy.flip(x[:,i]))
        y_filtered = do_filter_average(numpy.flip(y[:,i]))
        res.append((x_filtered[-1],y_filtered[-1]))

    res = numpy.array(res)

    return res
# ----------------------------------------------------------------------------------------------------------------------
def do_filter_median(X,N=5):
    Y = numpy.zeros(X.shape[0] + 2 * N + 1)
    Y[:N + 1] = X[0]
    Y[-N:] = X[-1]
    Y[N + 1:-N] = X
    R = medfilt(Y,kernel_size=N)

    return R[N:-N-1]
# ----------------------------------------------------------------------------------------------------------------------
def do_filter_average(X,N):
    Y = numpy.zeros(X.shape[0]+2*N+1)
    Y[:N+1]=X[0]
    Y[-N:] = X[-1]
    Y[N+1:-N]=X
    R = numpy.convolve(Y, numpy.ones((N,)) / N,mode='same')#[(N - 1):]
    return R[N:-N-1]
# ----------------------------------------------------------------------------------------------------------------------
def do_filter_dummy(X,N):
    res = X.copy()
    return res
# ----------------------------------------------------------------------------------------------------------------------
def do_filter_kalman_1D(X, noise_level = 1, Q = 0.001):
    #dim_x = X.shape[1]*2
    dim_x=2

    fk = KalmanFilter(dim_x=dim_x, dim_z=1)
    fk.x = numpy.array([0., 1.])  # state (x and dx)
    fk.F = numpy.array([[1., 1.], [0., 1.]])

    fk.H = numpy.array([[1., 0.]])  # Measurement function
    fk.P = 10.  # covariance matrix
    fk.R = noise_level  # state uncertainty
    fk.Q = Q  # process uncertainty

    X_fildered, cov, _, _ = fk.batch_filter(X)

    return X_fildered[:,0]

# ----------------------------------------------------------------------------------------------------------------------
def do_filter_kalman_2D(X):
    initial_state_mean = [X[0, 0], 0, X[0, 1], 0]

    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]

    KF = pykalman.KalmanFilter(transition_matrices=transition_matrix, observation_matrices=observation_matrix,initial_state_mean=initial_state_mean)

    KF = KF.em(X, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = KF.smooth(X)
    return smoothed_state_means[:,[0,2]]
# ----------------------------------------------------------------------------------------------------------------------
def fill_nan(A):

    is_ok = ~numpy.isnan(A)
    if numpy.all(is_ok):
        return A

    if 4*numpy.count_nonzero(1*is_ok)<len(A):
        return A

    xp = is_ok.ravel().nonzero()[0]
    fp = A[~numpy.isnan(A)]
    x = numpy.isnan(A).ravel().nonzero()[0]

    A[numpy.isnan(A)] = numpy.interp(x, xp, fp)
    return A
# ----------------------------------------------------------------------------------------------------------------------
def fill_zeros(A):

    idx = numpy.where(A == 0)[0]
    A[idx] = numpy.nan

    inds = numpy.arange(A.shape[0])
    good = numpy.where(numpy.isfinite(A))
    f = scipy.interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = numpy.where(numpy.isfinite(A),A,f(inds))
    A[idx]=0
    return B
# --------------------------------------------------------------------------------------------------------------------
def sliding_2d(A,h_neg,h_pos,w_neg,w_pos, stat='avg',mode='constant'):

    B = numpy.pad(A,((-h_neg,h_pos),(-w_neg,w_pos)),mode)
    B = numpy.roll(B, 1, axis=0)
    B = numpy.roll(B, 1, axis=1)

    C1 = numpy.cumsum(B , axis=0)
    C2 = numpy.cumsum(C1, axis=1)

    up = numpy.roll(C2, h_pos, axis=0)
    S1 = numpy.roll(up, w_pos, axis=1)
    S2 = numpy.roll(up, w_neg, axis=1)

    dn = numpy.roll(C2, h_neg, axis=0)
    S3 = numpy.roll(dn, w_pos, axis=1)
    S4 = numpy.roll(dn, w_neg, axis=1)

    if stat=='avg':
        R = (S1-S2-S3+S4)/((w_pos-w_neg)*(h_pos-h_neg))
    else:
        R = (S1 - S2 - S3 + S4)

    R = R[-h_neg:-h_pos, -w_neg:-w_pos]

    return R
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
def sliding_I_2d(C2,h_neg,h_pos,w_neg,w_pos,pad=10,stat='avg',mode='constant'):

    up = numpy.roll(C2, h_pos, axis=0)
    S1 = numpy.roll(up, w_pos, axis=1)
    S2 = numpy.roll(up, w_neg, axis=1)

    dn = numpy.roll(C2, h_neg, axis=0)
    S3 = numpy.roll(dn, w_pos, axis=1)
    S4 = numpy.roll(dn, w_neg, axis=1)

    if stat == 'avg':
        R = (S1 - S2 - S3 + S4) / ((w_pos - w_neg) * (h_pos - h_neg))
    else:
        R = (S1 - S2 - S3 + S4)

    R = R[pad:-pad, pad:-pad]

    return R
# --------------------------------------------------------------------------------------------------------------------
def integral_2d(A,pad=10,mode='constant'):
    h_neg,h_pos, w_neg,w_pos = -pad,pad,-pad,pad

    B = numpy.pad(A,((-h_neg,h_pos),(-w_neg,w_pos)),mode)
    B = numpy.roll(B, 1, axis=0)
    B = numpy.roll(B, 1, axis=1)

    C1 = numpy.cumsum(B , axis=0)
    C2 = numpy.cumsum(C1, axis=1)

    return C2
# --------------------------------------------------------------------------------------------------------------------
def filter_hor(gray2d, sobel_H=9, sobel_W = 9, skip_agg=False):

    sobel = numpy.full((sobel_H, sobel_W),+1, dtype=numpy.float32)
    sobel[:,  sobel.shape[1] // 2:] = +1
    sobel[:, :sobel.shape[1] // 2 ] = -1
    if sobel.sum() > 0:
        sobel = sobel / sobel.sum()
    filtered = cv2.filter2D(gray2d, 0, sobel)

    if skip_agg:
        return filtered
    else:
        agg = tools_image.sliding_2d(filtered, -sobel_H, +sobel_H, -(sobel_W//4),+(sobel_W//4))
        neg = numpy.roll(agg,-sobel_W//4, axis=1)
        pos = numpy.roll(agg,+sobel_W//4, axis=1)
        hit = ((255-neg)+pos)/2
        hit[:,  :3] = 0
        hit[:,-3: ] = 0

    return hit
# ----------------------------------------------------------------------------------------------------------------------
def filter_ver(gray2d, sobel_H, sobel_W,skip_agg=False):
    sobel = numpy.full((sobel_H, sobel_W), +1, dtype=numpy.float32)
    sobel[ sobel.shape[0] // 2:, :] = +1
    sobel[:sobel.shape[0] // 2,  :] = -1
    if sobel.sum()>0:
        sobel = sobel / sobel.sum()
    filtered = cv2.filter2D(gray2d, 0, sobel)

    if skip_agg:
        return filtered

    agg = tools_image.sliding_2d(filtered, -(sobel_H // 4), +(sobel_H // 4), -sobel_W, +sobel_W)
    neg = numpy.roll(agg, -sobel_H // 4, axis=0)
    pos = numpy.roll(agg, +sobel_H // 4, axis=0)
    hit = ((255 - neg) + pos) / 2
    hit[  :sobel_H,:] = 128
    hit[-sobel_H:, :] = 128
    return numpy.array(hit,dtype=numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------