import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_image
import tools_IO
import tools_calibrate
import tools_pr_geom
import tools_render_CV
import tools_GL3D
import tools_wavefront
# ----------------------------------------------------------------------------------------------------------------------
class Calibrator(object):
    def __init__(self,folder_out=None,scale_factor_gps = 100000.0):
        self.folder_out = folder_out
        self.scale_factor_gps = scale_factor_gps

        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_points(self,filename_in):
        X = tools_IO.load_mat_pd(filename_in)
        IDs = numpy.array(X[:, 0],dtype=numpy.float32).astype(numpy.int)
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
    def save_obj_file(self, filename_out,flat_obj=(-1, -1, 0, +1, +1, 0)):

        object = tools_wavefront.ObjLoader()
        cuboid_3d = self.construct_cuboid_v0(flat_obj)
        idx_vertex = numpy.array([[0,1,3],[3,2,0],[6,7,5],[5,4,6],[0,1,7],[6,7,0],[3,2,5],[5,4,2],[0,2,6],[4,6,2],[1,3,7],[5,3,7]])
        object.export_mesh(filename_out, cuboid_3d,idx_vertex=idx_vertex)

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
    def stage_points(self,filename_out,points_2d_all, points_gps_all):

        csv_header = ['ID', 'col', 'row', 'lat', 'long', 'height']
        tools_IO.append_CSV([-1, 0, 0, 0, 0, 0], filename_out, csv_header=csv_header, delim='\t')

        cnt=0
        for p1, p2 in zip(points_2d_all, points_gps_all):
            if (not numpy.any(numpy.isnan(p1))) and (not numpy.any(numpy.isnan(p2))):
                record = ['%d'%cnt, p1[0], p1[1], p2[0], p2[1], p2[2]]
                tools_IO.append_CSV(record, filename_out, csv_header=None, delim='\t')
            cnt += 1

        return
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_fov(self, filename_image, filename_points, a_min=0.4, a_max=0.41, list_of_R = (1, 10), virt_obj=None, do_debug = False):


        base_name = filename_image.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_image)
        W,H = image.shape[1], image.shape[0]
        gray = tools_image.desaturate(image)
        points_2d_all, points_gps_all,IDs = self.load_points(filename_points)
        points_xyz = self.shift_origin(points_gps_all, points_gps_all[0])
        points_xyz, points_2d = points_xyz[1:], points_2d_all[1:]

        err, rvecs, tvecs = [],[],[]
        a_fovs = numpy.arange(a_min, a_max, 0.005)

        for a_fov in a_fovs:
            camera_matrix = tools_pr_geom.compose_projection_mat_3x3(W, H, a_fov, a_fov)
            rvec, tvec, points_2d_check = tools_pr_geom.fit_pnp(points_xyz, points_2d, camera_matrix, numpy.zeros(5))
            err.append(numpy.sqrt(((points_2d_check-points_2d)**2).sum()/len(points_2d)))
            rvecs.append(rvec.flatten())
            tvecs.append(tvec.flatten())

        idx_best = numpy.argmin(numpy.array(err))

        if do_debug:
            tools_IO.remove_files(self.folder_out,'*.png')
            for i in range(len(a_fovs)):
                color_markup = (159, 206, 255)
                if i==idx_best:color_markup = (0,128,255)
                camera_matrix = tools_pr_geom.compose_projection_mat_3x3(W, H, a_fovs[i], a_fovs[i])
                points_2d_check, jac = cv2.projectPoints(points_xyz, rvecs[i], tvecs[i], camera_matrix, numpy.zeros(5))
                image_AR = tools_draw_numpy.draw_ellipses(gray, [((p[0], p[1]), (25, 25), 0) for p in points_2d],color=(0, 0, 190), w=4)
                image_AR = tools_draw_numpy.draw_points(image_AR, points_2d_check.reshape((-1,2)), color=(0, 128, 255), w=8)
                for rad in list_of_R:
                    image_AR = tools_render_CV.draw_compass(image_AR, camera_matrix, numpy.zeros(5), rvecs[i],tvecs[i], rad, color=color_markup)
                if virt_obj is not None:
                    for p in points_xyz:
                        image_AR = tools_draw_numpy.draw_cuboid(image_AR,tools_pr_geom.project_points(p+self.construct_cuboid_v0(virt_obj), rvecs[i], tvecs[i], camera_matrix, numpy.zeros(5))[0])

                cv2.imwrite(self.folder_out + base_name + '_%05d' % (a_fovs[i] * 1000) + '.png', image_AR)

        return a_fovs[idx_best], numpy.array(rvecs)[idx_best], numpy.array(tvecs)[idx_best]
# ----------------------------------------------------------------------------------------------------------------------
    def get_pretty_model_rotation(self,mat_model):

        rvec, tvec = tools_pr_geom.decompose_to_rvec_tvec(mat_model)
        a_pitch_deg, a_yaw_deg, a_roll_deg =  rvec[[0,1,2]]*180/numpy.pi

        #check
        rvec_check = numpy.array((a_pitch_deg, a_yaw_deg, a_roll_deg)) * numpy.pi / 180
        mat_model_check3 = tools_pr_geom.compose_RT_mat(rvec_check, (0,0,0), do_rodriges=False, do_flip=False)



        return a_pitch_deg, a_yaw_deg, a_roll_deg
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_matrices_GL(self, image, rvec, tvec, a_fov, virt_obj=None, do_debug=False):

        if numpy.any(numpy.isnan(rvec)) or numpy.any(numpy.isnan(tvec)):return None, None, None
        H,W = image.shape[:2]

        cuboid_3d = self.construct_cuboid_v0(virt_obj)
        mRT = tools_pr_geom.compose_RT_mat(rvec, tvec, do_rodriges=True, do_flip=True, GL_style=True)
        mR = tools_pr_geom.compose_RT_mat(rvec,  (0,0,0), do_rodriges=True, do_flip=True, GL_style=True)
        imR = numpy.linalg.inv(mR)
        mat_projection = tools_pr_geom.compose_projection_mat_4x4_GL(W, H, a_fov, a_fov)
        T = numpy.matmul(mRT, imR)

        if do_debug:

            gray = tools_image.desaturate(cv2.resize(image, (W, H)))

            filename_obj = self.folder_out + 'temp.obj'
            self.save_obj_file(filename_obj, virt_obj)
            tools_IO.remove_file(filename_obj)

            R = tools_GL3D.render_GL3D(filename_obj=filename_obj, W=W, H=H, do_normalize_model_file=False,is_visible=False, projection_type='P', textured=False)
            cv2.imwrite(self.folder_out + 'AR0_GL_m1.png', R.get_image_perspective(rvec, tvec, a_fov, a_fov, mat_view_to_1=False, do_debug=True))
            cv2.imwrite(self.folder_out + 'AR0_GL_v1.png', R.get_image_perspective(rvec, tvec, a_fov, a_fov, mat_view_to_1=True, do_debug=True))
            cv2.imwrite(self.folder_out + 'AR0_CV.png',tools_render_CV.draw_cube_numpy_MVP_GL(gray, mat_projection, numpy.eye(4), mRT,numpy.eye(4), points_3d=cuboid_3d))
            cv2.imwrite(self.folder_out + 'AR1_CV.png',tools_render_CV.draw_cube_numpy_MVP_GL(gray, mat_projection, mR, numpy.eye(4), T,points_3d=cuboid_3d))
            cv2.imwrite(self.folder_out + 'AR1_GL.png',R.get_image(mat_view=numpy.eye(4),mat_model=mR,mat_trans=T,do_debug=True))

        return numpy.eye(4),mR, T
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
        points_gnss = numpy.array([(52.228391, 21.01075), (52.227676, 21.000511), (52.227384, 21.001689), (52.227674, 21.001706)],numpy.float32)
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