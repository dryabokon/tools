# ----------------------------------------------------------------------------------------------------------------------
import numpy
from filterpy.kalman import KalmanFilter
from scipy.signal import medfilt
import scipy.interpolate
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
    R = medfilt(X,kernel_size=N)
    R[:N] = X[:N]
    R[-N:] = X[-N:]
    return R
# ----------------------------------------------------------------------------------------------------------------------
def do_filter_average(X,N):
    R = numpy.convolve(X, numpy.ones((N,)) / N,mode='same')#[(N - 1):]
    R[:N] = X[:N]
    R[-N:] = X[-N:]
    return R
# ----------------------------------------------------------------------------------------------------------------------
def do_filter_dummy(X,N):
    res = X.copy()
    return res
# ----------------------------------------------------------------------------------------------------------------------
def do_filter_kalman(X,noise_level = 1,Q = 0.001):

    fk = KalmanFilter(dim_x=2, dim_z=1)
    fk.x = numpy.array([0., 1.])  # state (x and dx)
    fk.F = numpy.array([[1., 1.], [0., 1.]])

    fk.H = numpy.array([[1., 0.]])  # Measurement function
    fk.P = 10.  # covariance matrix
    fk.R = noise_level  # state uncertainty
    fk.Q = Q  # process uncertainty

    X_fildered, cov, _, _ = fk.batch_filter(X)

    return X_fildered[:,0]

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