import cv2
import numpy
from cv2 import aruco
import pyrr
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_image
import tools_IO
import tools_calibrate
import tools_pr_geom
import tools_render_CV
import tools_GL3D
# ----------------------------------------------------------------------------------------------------------------------
class Calibrator(object):
    def __init__(self,folder_out=None,scale_factor_gps = 100000.0):
        self.folder_out = folder_out
        self.scale_factor_gps = scale_factor_gps

        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_points(self,filename_in):
        X = tools_IO.load_mat_pd(filename_in)
        IDs = numpy.array(X[:, 0],dtype=numpy.int32)
        X = X[:, 1:]
        points_2d = numpy.array(X[:, :2], dtype=numpy.float32)
        points_gps = numpy.array(X[:, 2:], dtype=numpy.float32)
        return points_2d, points_gps, IDs
# ----------------------------------------------------------------------------------------------------------------------
    def shift_origin(self, points_gnss_3d, origin_xy, orignin_z=0, scale_factor_xy=None):
        points_gnss_3d_normed = points_gnss_3d.copy()
        if scale_factor_xy is None: scale_factor_xy = self.scale_factor_gps

        points_gnss_3d_normed[:,[0,1]]-=origin_xy[[0, 1]]
        points_gnss_3d_normed[:,[0,1]]*=scale_factor_xy
        points_gnss_3d_normed[:,2]=orignin_z

        return points_gnss_3d_normed
# ----------------------------------------------------------------------------------------------------------------------
    def BEV_points(self,filename_points):
        base_name = filename_points.split('/')[-1].split('.')[0]
        points_2d, points_gps, IDs = self.load_points(filename_points)
        points_xyz = self.shift_origin(points_gps, points_gps[0])


        xy = points_xyz[:,:2].copy()
        scale = max((xy[:, 0].max()-xy[:, 0].min()) / 1000, (xy[:, 1].max()-xy[:, 1].min()) / 1000)

        xy[:,0]-=xy[:,0].min()
        xy[:,0]/=scale
        xy[:,1]-=xy[:,1].min()
        xy[:,1]/=scale


        image_xy = tools_draw_numpy.extend_view_from_image(numpy.full((1000,1000,3),32,dtype=numpy.uint8), factor=2)
        xy_ext = tools_draw_numpy.extend_view(xy, 1000, 1000, factor=2)
        image_xy = tools_draw_numpy.draw_points(image_xy, xy_ext[:1], color=(0, 0, 190), w=8)
        image_xy = tools_draw_numpy.draw_points(image_xy, xy_ext[1:], color=(0, 190, 255), w=8,labels=['%02d'%id for id in IDs[1:]])

        cv2.imwrite(self.folder_out + base_name + '_BEV.png', image_xy)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def AR_points(self, filename_image, filename_points, camera_matrix_init,dist=numpy.zeros(5), do_debug=False):
        base_name = filename_image.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_image)
        gray = tools_image.desaturate(image)
        points_2d, points_gps, IDs = self.load_points(filename_points)
        points_xyz = self.shift_origin(points_gps, points_gps[0])
        points_xyz,points_2d =points_xyz[1:],points_2d[1:]
        image_AR = gray.copy()

        camera_matrix = numpy.array(camera_matrix_init,dtype=numpy.float32)
        rvecs, tvecs, points_2d_check = tools_pr_geom.fit_pnp(points_xyz, points_2d,camera_matrix, dist)

        labels = ['(%2.1f,%2.1f)'%(p[0],p[1]) for p in points_xyz]
        image_AR = tools_draw_numpy.draw_points(image_AR, points_2d,color=(0,0,190),w=16,labels=labels)
        image_AR = tools_draw_numpy.draw_points(image_AR, points_2d_check, color=(0, 128, 255), w=8)

        for p in points_xyz:
            points_3d = 0.5*numpy.array([[-1, -1, 0], [-1, +1, 0], [+1, +1, 0], [+1, -1, 0], [-1, -1, -2], [-1, +1, -2], [+1, +1, -2],[+1, -1, -2]], dtype=numpy.float32)
            points_3d[:, [0,1,2]] += p[[0,1,2]]
            image_AR = tools_render_CV.draw_cube_numpy(image_AR, camera_matrix, dist, numpy.array(rvecs).flatten(),numpy.array(tvecs).flatten(), (0.5, 0.5, 0.5), points_3d=points_3d)

        cv2.imwrite(self.folder_out+base_name+'.png',image_AR)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def calibrate_aruco(self,folder_in,camera_matrix,do_debug=False):
        marker_length = 0.10
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

        gray_rgb = None

        for filename in tools_IO.get_filenames(folder_in, '*.png'):
            image = cv2.imread(folder_in + filename)
            #camera_matrix = tools_pr_geom.compose_projection_mat_3x3(image.shape[1],image.shape[0])
            gray_rgb = tools_image.desaturate(image)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY), aruco_dict)

            if len(corners) > 0:
                gray_rgb = aruco.drawDetectedMarkers(gray_rgb, corners)
                for each in corners:
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(each, marker_length, camera_matrix, numpy.zeros(5))
                    aruco.drawAxis(gray_rgb, camera_matrix, numpy.zeros(5), rvec[0], tvec[0], marker_length / 2)

        if do_debug:
            cv2.imwrite(self.folder_out + 'aruco_calibration.png',gray_rgb)

        return gray_rgb
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid(self,flat_obj=(-1,-1,0,+1,+1,0)):
        xmin, ymin, zmin = flat_obj[0],flat_obj[1],flat_obj[2]
        xmax, ymax, zmax = flat_obj[3],flat_obj[4],flat_obj[5]

        points_3d = numpy.array(
            [[xmin, ymin, zmin], [xmin, ymax, zmin],
             [xmax, ymax, zmin], [xmax, ymin, zmin],
             [xmin, ymin, zmax], [xmin, ymax, zmax],
             [xmax, ymax, zmax], [xmax, ymin, zmax]], dtype=numpy.float32)
        return  points_3d
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_K_bruteforce_F(self, filename_image, filename_points, a_min=0.4, a_max=0.41,list_of_R = (1,10),virt_obj=None,filename_obj=None,do_verbose=False):
        tools_IO.remove_files(self.folder_out)

        base_name = filename_image.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_image)
        W,H = image.shape[1], image.shape[0]
        gray = tools_image.desaturate(image)
        points_2d_all, points_gps_all,IDs = self.load_points(filename_points)
        render_GL3D = None
        if filename_obj is not None:
            render_GL3D = tools_GL3D.render_GL3D(filename_obj, W, H, is_visible=False,do_normalize_model_file=False,projection_type='P')

        for aperture in numpy.arange(a_min,a_max,0.005):
            points_xyz = self.shift_origin(points_gps_all, points_gps_all[0])
            points_xyz, points_2d = points_xyz[1:], points_2d_all[1:]
            labels = ['(%2.1f,%2.1f)' % (p[0], p[1]) for p in points_xyz]

            image_AR = gray.copy()
            camera_matrix = tools_pr_geom.compose_projection_mat_3x3(W, H, aperture, aperture)
            rvecs, tvecs, points_2d_check = tools_pr_geom.fit_pnp(points_xyz, points_2d, camera_matrix, numpy.zeros(5))
            rvecs, tvecs = numpy.array(rvecs).flatten(), numpy.array(tvecs).flatten()
            err = numpy.sqrt(((points_2d_check-points_2d)**2).sum()/len(points_2d))
            if do_verbose:
                print(aperture,rvecs,tvecs, err)
            #print('P%+1.2f Y%+1.2f R %+1.2f T+%2.1f %+2.1f %+2.1f : err %2.1f' % (rvecs[0]*180/numpy.pi, rvecs[1]*180/numpy.pi,rvecs[2]*180/numpy.pi,tvecs[0],tvecs[1],tvecs[2],err))

            for rad in list_of_R:image_AR = tools_render_CV.draw_compass(image_AR, camera_matrix, numpy.zeros(5), rvecs, tvecs, rad)

            image_AR = tools_draw_numpy.draw_points(image_AR, points_2d, color=(0, 0, 190), w=8, labels=labels)
            image_AR = tools_draw_numpy.draw_points(image_AR, points_2d_check, color=(0, 128, 255), w=4)
            if virt_obj is not None:image_AR = tools_render_CV.draw_cube_numpy(image_AR, camera_matrix, numpy.zeros(5), rvecs, tvecs,color=(255,128,0),points_3d=self.construct_cuboid(virt_obj))



            if render_GL3D is not None:
                image_GL = render_GL3D.get_image_perspective(rvecs, tvecs, aperture, aperture,freeze_mat_view=True,do_debug=True)
                #image_AR = cv2.addWeighted(image_AR,0.25,image_GL,0.7,0)


            # check1
            # cuboid = self.construct_cuboid(virt_obj)
            # #cuboid = numpy.zeros((8,3),dtype=numpy.float32)
            # cuboid_r = tools_pr_geom.apply_rotation(rvecs,cuboid)
            # cuboid_rt = tools_pr_geom.apply_translation(tvecs, cuboid_r)[:, :3]
            # image_AR1 = tools_render_CV.draw_cube_numpy(image_AR, camera_matrix, numpy.zeros(5), rvec=numpy.zeros(3),tvec=numpy.zeros(3), color=(255, 128, 0), points_3d=cuboid_rt)
            # mat_view = tools_pr_geom.compose_RT_mat((0,0,0),(0,0,0),do_rodriges=True,do_flip=True)
            # image_AR1 = tools_render_CV.draw_points_numpy_MVP(cuboid_rt, image_AR1, render_GL3D.mat_projection, mat_view, numpy.eye(4), numpy.eye(4),w=6)
            #
            # #check2
            # MT = tools_pr_geom.compose_RT_mat((0,0,0), tvecs, do_rodriges=True, do_flip=True)
            # MR = tools_pr_geom.compose_RT_mat(rvecs, (0,0,0), do_rodriges=True, do_flip=True)
            # mRT = pyrr.matrix44.multiply(MT, MR)
            # iMT = pyrr.matrix44.inverse(MT)
            # mat_view = pyrr.matrix44.multiply(mRT, iMT)
            #
            #
            # cuboid_t = tools_pr_geom.apply_translation(tvecs, cuboid)[:,:3]
            # #M1 = tools_pr_geom.compose_RT_mat(rvecs,tvecs,do_rodriges=True,do_flip=True)
            # #M2 = tools_pr_geom.compose_RT_mat((0,0,0), tvecs, do_rodriges=True, do_flip=True)
            # #mat_view = pyrr.matrix44.multiply(M1,pyrr.matrix44.inverse(M2))
            # image_AR2 = tools_render_CV.draw_points_numpy_MVP(cuboid_t, image_AR, render_GL3D.mat_projection, mat_view, numpy.eye(4), numpy.eye(4),w=6)


            cv2.imwrite(self.folder_out + base_name + '_%05d'%(aperture*100) + '.png', image_AR)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_K_chessboard(self,folder_in):

        chess_rows, chess_cols = 7,5
        camera_matrix,dist,rvecs, tvecs = tools_calibrate.get_proj_dist_mat_for_images(folder_in, chess_rows, chess_cols,cell_size=0.03, folder_out=self.folder_out)
        print(camera_matrix)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def derive_homography(self, points_xy, points_2d_gnss):
        homography, result = tools_pr_geom.fit_homography(points_xy, points_2d_gnss)
        return homography
# ----------------------------------------------------------------------------------------------------------------------
    def to_gnss_2d(self, points_xy, homography):

        if len(points_xy.shape)==1:
            points_xy = numpy.array([points_xy])

        points_xy1 = numpy.hstack((points_xy,numpy.full((len(points_xy),1), 1)))
        points_xy1  = points_xy1.T

        points_gnss = numpy.matmul(homography, points_xy1)
        points_gnss[[0, 1]] = points_gnss[[0, 1]] / points_gnss[2]
        points_gnss = points_gnss[:-1].T

        return points_gnss
# ----------------------------------------------------------------------------------------------------------------------
    def unit_test(self):

        points_xy = numpy.array([(200, 591), (317, 84), (657, 1075), (381, 952)], numpy.float32)
        points_gnss = numpy.array([(52.228391, 21.001075), (52.227676, 21.000511), (52.227384, 21.001689), (52.227674, 21.001706)],numpy.float32)
        H = self.derive_homography(points_xy, points_gnss)

        point_xy = numpy.array((343, 705), numpy.float32)
        point_gnss = self.to_gnss_2d(point_xy, H)
        print(point_gnss)

        return
# ----------------------------------------------------------------------------------------------------------------------
