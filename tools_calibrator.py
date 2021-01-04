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
    def BEV_points(self,filename_points,list_of_R = None):
        base_name = filename_points.split('/')[-1].split('.')[0]
        points_2d, points_gps, IDs = self.load_points(filename_points)
        points_xyz = self.shift_origin(points_gps, points_gps[0])


        xy = points_xyz[:,:2].copy()
        scale = max((xy[:, 0].max()-xy[:, 0].min()) / 1000, (xy[:, 1].max()-xy[:, 1].min()) / 1000)

        xy[:,0]-=xy[:,0].min()
        xy[:,0]/=scale
        xy[:,1]-=xy[:,1].min()
        xy[:,1]/=scale
        factor = 1.2

        labels = ['ID %02d: %2.1f,%2.1f' % (pid, p[0], p[1]) for pid, p in zip(IDs, points_xyz)]
        image_BEV = tools_draw_numpy.extend_view_from_image(numpy.full((1000,1000,3),(255,255,255),dtype=numpy.uint8), factor=factor,color_bg=(255,255,255))
        xy_ext = tools_draw_numpy.extend_view(xy, 1000, 1000, factor)
        image_BEV = tools_draw_numpy.draw_points(image_BEV, xy_ext[:1], color=(255, 145, 0), w=8)
        image_BEV = tools_draw_numpy.draw_ellipses(image_BEV, [((p[0], p[1]), (25, 25), 0) for p in xy_ext[1:]],color=(0, 0, 190), w=4, labels=labels)


        if list_of_R is not None:
            center_ext = tools_draw_numpy.extend_view(xy[0], 1000, 1000, factor)[0]
            for rad in list_of_R:
                ellipse = ((center_ext[0],center_ext[1]),(rad/scale/factor,rad/scale/factor),0)
                image_BEV = tools_draw_numpy.draw_ellipses(image_BEV, [ellipse], color=(0,128,255), w=1)


        cv2.imwrite(self.folder_out + base_name + '_BEV.png', image_BEV)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def AR_points(self, filename_image, filename_points, camera_matrix_init,dist=numpy.zeros(5), list_of_R =None,virt_obj=None):
        base_name = filename_image.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_image)
        gray = tools_image.desaturate(image)
        points_2d, points_gps, IDs = self.load_points(filename_points)
        points_xyz = self.shift_origin(points_gps, points_gps[0])
        IDs, points_xyz,points_2d =IDs[1:],points_xyz[1:],points_2d[1:]
        image_AR = gray.copy()

        camera_matrix = numpy.array(camera_matrix_init,dtype=numpy.float32)
        rvecs, tvecs, points_2d_check = tools_pr_geom.fit_pnp(points_xyz, points_2d,camera_matrix, dist)
        rvecs, tvecs = numpy.array(rvecs).flatten(), numpy.array(tvecs).flatten()

        if list_of_R is not None:
            for rad in list_of_R:image_AR = tools_render_CV.draw_compass(image_AR, camera_matrix, numpy.zeros(5), rvecs,tvecs, rad)

        if virt_obj is not None:
            for p in points_xyz:
                cuboid_3d = p + self.construct_cuboid_v0(virt_obj)
                M = tools_pr_geom.compose_RT_mat(rvecs,tvecs,do_rodriges=True,do_flip=False,GL_style=False)
                P = tools_pr_geom.compose_projection_mat_4x4(camera_matrix[0, 0], camera_matrix[1, 1],camera_matrix[0, 2] / camera_matrix[0, 0],camera_matrix[1, 2] / camera_matrix[1, 1])
                image_AR = tools_draw_numpy.draw_cuboid(image_AR,tools_pr_geom.project_points_p3x4(cuboid_3d, numpy.matmul(P,M)))

        labels = ['ID %02d: %2.1f,%2.1f'%(pid, p[0], p[1]) for pid, p in zip(IDs,points_xyz)]
        image_AR = tools_draw_numpy.draw_ellipses(image_AR, [((p[0], p[1]), (25, 25), 0) for p in points_2d],color=(0, 0, 190), w=4,labels=labels)
        image_AR = tools_draw_numpy.draw_points(image_AR, points_2d_check, color=(0, 128, 255), w=8)

        cv2.imwrite(self.folder_out+base_name+'.png',image_AR)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid_v0(self,flat_obj=(-1,-1,0,+1,+1,0)):
        xmin, ymin, zmin = flat_obj[0],flat_obj[1],flat_obj[2]
        xmax, ymax, zmax = flat_obj[3],flat_obj[4],flat_obj[5]

        #points_3d = numpy.array([[xmin, ymin, zmin], [xmin, ymax, zmin],[xmax, ymax, zmin], [xmax, ymin, zmin],[xmin, ymin, zmax], [xmin, ymax, zmax],[xmax, ymax, zmax], [xmax, ymin, zmax]], dtype=numpy.float32)
        points_3d = numpy.array([[xmin, ymin, zmin], [xmin, ymax, zmin],[xmax, ymin, zmin], [xmax, ymax, zmin],
                                 [xmax, ymin, zmax], [xmax, ymax, zmax],[xmin, ymin, zmax], [xmin, ymax, zmax]], dtype=numpy.float32)

        yy=0
        return  points_3d
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid(self, dim,tvec,rvec):
        d0, d1, d2 = dim[0], dim[1], dim[2]

        #x_corners = [+d0/2, +d0/2, -d0/2, -d0/2, +d0/2, +d0/2, -d0/2, -d0/2]
        #y_corners = [+d1/2, +d1/2, +d1/2, +d1/2, -d1/2, -d1/2, -d1/2, -d1/2]
        #z_corners = [+d2/2, -d2/2, -d2/2, +d2/2, +d2/2, -d2/2, -d2/2, +d2/2]

        x_corners = [-d0/2, -d0/2, +d0/2, +d0/2, +d0/2, +d0/2, -d0/2, -d0/2]
        y_corners = [-d1/2, +d1/2, -d1/2, +d1/2, -d1/2, +d1/2, -d1/2, +d1/2]
        z_corners = [-d2/2, -d2/2, -d2/2, -d2/2, +d2/2, +d2/2, +d2/2, +d2/2]

        X = numpy.array([x_corners, y_corners, z_corners], dtype=numpy.float32).T

        RT = tools_pr_geom.compose_RT_mat(rvec,tvec,do_rodriges=True,do_flip=False,GL_style=False)
        Xt = tools_pr_geom.apply_matrix(RT,X)[:,:3]


        uu=0
        return Xt
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_aperture(self, filename_image, filename_points, a_min=0.4, a_max=0.41, list_of_R = (1, 10), virt_obj=None, do_debug = False):
        tools_IO.remove_files(self.folder_out)

        base_name = filename_image.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_image)
        W,H = image.shape[1], image.shape[0]
        gray = tools_image.desaturate(image)
        points_2d_all, points_gps_all,IDs = self.load_points(filename_points)
        points_xyz = self.shift_origin(points_gps_all, points_gps_all[0])
        points_xyz, points_2d = points_xyz[1:], points_2d_all[1:]

        err, rvecs, tvecs = [],[],[]
        apertures = numpy.arange(a_min, a_max, 0.005)

        for aperture in apertures:
            camera_matrix = tools_pr_geom.compose_projection_mat_3x3(W, H, aperture, aperture)
            rvec, tvec, points_2d_check = tools_pr_geom.fit_pnp(points_xyz, points_2d, camera_matrix, numpy.zeros(5))
            err.append(numpy.sqrt(((points_2d_check-points_2d)**2).sum()/len(points_2d)))
            rvecs.append(rvec.flatten())
            tvecs.append(tvec.flatten())

        idx_best = numpy.argmin(numpy.array(err))

        if do_debug:
            for i in range(len(apertures)):
                color_markup = (159, 206, 255)
                if i==idx_best:color_markup = (0,128,255)
                camera_matrix = tools_pr_geom.compose_projection_mat_3x3(W, H, apertures[i], apertures[i])
                points_2d_check, jac = cv2.projectPoints(points_xyz, rvecs[i], tvecs[i], camera_matrix, numpy.zeros(5))
                image_AR = tools_draw_numpy.draw_ellipses(gray, [((p[0], p[1]), (25, 25), 0) for p in points_2d],color=(0, 0, 190), w=4)
                image_AR = tools_draw_numpy.draw_points(image_AR, points_2d_check.reshape((-1,2)), color=(0, 128, 255), w=8)
                for rad in list_of_R:
                    image_AR = tools_render_CV.draw_compass(image_AR, camera_matrix, numpy.zeros(5), rvecs[i],tvecs[i], rad, color=color_markup)
                if virt_obj is not None:
                    for p in points_xyz:
                        image_AR = tools_draw_numpy.draw_cuboid(image_AR,tools_pr_geom.project_points(p+self.construct_cuboid_v0(virt_obj), rvecs[i], tvecs[i], camera_matrix, numpy.zeros(5))[0])

                cv2.imwrite(self.folder_out + base_name + '_%05d' % (apertures[i] * 1000) + '.png', image_AR)

        return apertures[idx_best], numpy.array(rvecs)[idx_best], numpy.array(tvecs)[idx_best]
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_matrices_GL(self,image,rvec,tvec,aperture,virt_obj=None,do_debug=False):

        if numpy.any(numpy.isnan(rvec)) or numpy.any(numpy.isnan(tvec)):
            return None, None, None, None, None

        mat_projection = tools_pr_geom.compose_projection_mat_4x4_GL(aperture,aperture)

        cuboid = self.construct_cuboid_v0(virt_obj)

        MT = tools_pr_geom.compose_RT_mat((0, 0, 0), tvec, do_rodriges=True, do_flip=False, GL_style=False)
        MR = tools_pr_geom.compose_RT_mat(rvec, (0, 0, 0), do_rodriges=True, do_flip=False, GL_style=False)
        mRT0 = tools_pr_geom.compose_RT_mat(rvec, tvec, do_rodriges=True, do_flip=False, GL_style=False)
        mRT = pyrr.matrix44.multiply(MT, MR)

        mat_model = tools_pr_geom.compose_RT_mat(rvec, tvec, do_rodriges=True,do_flip=False,GL_style=True)
        mat_view = numpy.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1.0]])

        #check 0
        if do_debug:
            image_AR0 = tools_render_CV.draw_cube_numpy_MVP(image, mat_projection, mat_view, mat_model, numpy.eye(4), points_3d=cuboid)
            cv2.imwrite(self.folder_out + 'AR0.png', image_AR0)

        # check 1
        if do_debug:
            cuboid_rt = tools_pr_geom.apply_matrix(mRT, cuboid)[:, :3]
            image_AR1 = tools_render_CV.draw_cube_numpy_MVP(image, mat_projection, mat_view, numpy.eye(4), numpy.eye(4), points_3d=cuboid_rt)
            cv2.imwrite(self.folder_out + 'AR1.png', image_AR1)

        # check 2
        if do_debug:
            image_AR2 = tools_render_CV.draw_cube_numpy_MVP(image, mat_projection, mat_view, mRT0.T,numpy.eye(4), points_3d=cuboid)
            cv2.imwrite(self.folder_out + 'AR2.png', image_AR2)

        # check 3
        shift = tools_pr_geom.apply_matrix(MR, tvec)[0, :3]
        shift[0] *= 0
        shift[1] *= -1
        MT2 = tools_pr_geom.compose_RT_mat((0, 0, 0), +shift, do_rodriges=True, do_flip=False, GL_style=False)
        iMT2 = tools_pr_geom.compose_RT_mat((0, 0, 0), -shift, do_rodriges=True, do_flip=False, GL_style=False)
        mat_model_new = pyrr.matrix44.multiply(mRT0, iMT2).T
        cuboid_t = tools_pr_geom.apply_matrix(MT2, cuboid)[:, :3]

        if do_debug:
            image_AR3 = tools_render_CV.draw_cube_numpy_MVP(image, mat_projection, mat_view,mat_model_new, numpy.eye(4), points_3d=cuboid_t)
            cv2.imwrite(self.folder_out + 'AR3.png', image_AR3)

        return shift[1], shift[2], aperture, mat_view, mat_model_new
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
    def unit_test_gnss(self):

        points_xy = numpy.array([(200, 591), (317, 84), (657, 1075), (381, 952)], numpy.float32)
        points_gnss = numpy.array([(52.228391, 21.001075), (52.227676, 21.000511), (52.227384, 21.001689), (52.227674, 21.001706)],numpy.float32)
        H = self.derive_homography(points_xy, points_gnss)

        point_xy = numpy.array((343, 705), numpy.float32)
        point_gnss = self.to_gnss_2d(point_xy, H)
        print(point_gnss)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def derive_cuboid_3d(self, mat_camera_3x3, r_vec, t_vec, points_2D, base_name=None, do_debug=False):

        target_centr_top = numpy.mean(points_2D[[0, 2, 4, 6]], axis=0)
        target_centr_bottom = numpy.mean(points_2D[[1, 3, 5, 7]], axis=0)
        target_centr_bottom_3D = \
        tools_pr_geom.reverce_project_points_Z0([target_centr_bottom], r_vec, t_vec, mat_camera_3x3,
                                                numpy.zeros(5))[0]

        diffs, Hs = [], numpy.arange(0, 10, 0.1)
        for h in Hs:
            centr_top_2D, _ = tools_pr_geom.project_points(target_centr_bottom_3D - numpy.array((0, 0, h)), r_vec,t_vec, mat_camera_3x3, numpy.zeros(5))
            diffs.append(numpy.abs(centr_top_2D[0][1] - target_centr_top[1]))

        best_H = Hs[numpy.argmin(diffs)]

        points_3D = tools_pr_geom.reverce_project_points_Z0(points_2D[[1, 3, 5, 7]], r_vec, t_vec, mat_camera_3x3,numpy.zeros(5))

        points_3D = numpy.vstack((points_3D, points_3D))
        points_3D[4:, 2] = -best_H
        points_3D = points_3D[[4, 0, 5, 1, 6, 2, 7, 3]]

        if do_debug:
            sources_2D, _ = tools_pr_geom.project_points(points_3D, r_vec, t_vec, mat_camera_3x3, numpy.zeros(5))
            image_debug = tools_draw_numpy.draw_cuboid(numpy.full((1080, 1920, 3), 32, dtype=numpy.uint8),points_2D, color=(0, 0, 200), put_text=True)
            image_debug = tools_draw_numpy.draw_cuboid(image_debug, sources_2D, color=(255, 140, 0), put_text=True)
            cv2.imwrite(self.folder_out + base_name + '_R.png', image_debug)

        return points_3D
# ----------------------------------------------------------------------------------------------------------------------