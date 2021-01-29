import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_image
import tools_IO
import tools_calibrate
import tools_pr_geom
import tools_render_CV
# ----------------------------------------------------------------------------------------------------------------------
class Calibrator(object):
    def __init__(self,folder_out=None,scale_factor_gps=1000*40075.0/360):
        self.folder_out = folder_out
        self.scale_factor_gps = scale_factor_gps
        self.colormap_circles = 'jet'
        self.colormap_objects = 'rainbow'
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
    def save_points(self, filename_out, IDs, points_2d_all, points_gps_all,W,H):

        tools_IO.remove_file(filename_out)
        csv_header = ['ID', 'col', 'row', 'lat', 'long', 'height']
        tools_IO.append_CSV([-1, 0, 0, 0, 0, 0], filename_out, csv_header=csv_header, delim='\t')

        for pid, p1, p2 in zip(IDs, points_2d_all, points_gps_all):
            if (numpy.any(numpy.isnan(p1))):continue
            if (numpy.any(numpy.isnan(p2))):continue
            if p1[0]<0 or p1[0]>W: continue
            if p1[1]<0 or p1[1]>H: continue
            record = ['%d' % pid, p1[0], p1[1], p2[0], p2[1], p2[2]]
            tools_IO.append_CSV(record, filename_out, csv_header=None, delim='\t')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def shift_scale(self, points_gnss_3d, origin_xy, orignin_z=0, scale_factor_xy=None):
        points_gnss_3d_normed = points_gnss_3d.copy()
        if scale_factor_xy is None: scale_factor_xy = self.scale_factor_gps

        points_gnss_3d_normed[:,[0,1]]-=origin_xy[[0, 1]]
        points_gnss_3d_normed[:,[0,1]]*=scale_factor_xy

        points_gnss_3d_normed[:,2]=orignin_z

        return points_gnss_3d_normed
# ----------------------------------------------------------------------------------------------------------------------
    def get_adjustment_params(self,marker_xy,target_W,target_H):
        shift1, shift2 = marker_xy[:, 0].min(), marker_xy[:, 1].min()
        scale = max((marker_xy[:, 0].max() - marker_xy[:, 0].min()) / target_W,(marker_xy[:, 1].max() - marker_xy[:, 1].min()) / target_W)

        xy = marker_xy.copy()
        xy[:, 0] -= shift1
        xy[:, 0] /= scale
        xy[:, 1] -= shift2
        xy[:, 1] /= scale

        if (xy[0, 0] > xy[1:, 0].min() and xy[0, 0] < xy[1:, 0].max()):
            shift_h = +xy[0, 0]
        else:
            shift_h = None

        do_flip_v =marker_xy[0, 1] < marker_xy[1:, 1].mean()

        return scale,shift1,shift2,do_flip_v,shift_h
# ----------------------------------------------------------------------------------------------------------------------
    def adjust_2d(self, xy, target_W, target_H, scale, shift1, shift2, do_flip_v, shift_h=None):

        xy[:, 0] -= shift1
        xy[:, 0] /= scale
        xy[:, 1] -= shift2
        xy[:, 1] /= scale

        if shift_h is not None:
            xy[:, 0]+=target_W/2-shift_h

        if do_flip_v:
            xy[:, 1] = target_H - xy[:, 1]
            xy[:, 0] = target_W - xy[:, 0]

        return xy
# ----------------------------------------------------------------------------------------------------------------------
    def AR_points(self, image, filename_points, camera_matrix_init,dist,do_shift_scale, list_of_R =None,virt_obj=None):

        gray = tools_image.desaturate(image)
        points_2d, points_gps, IDs = self.load_points(filename_points)
        if do_shift_scale:
            points_xyz = self.shift_scale(points_gps, points_gps[0])
        else:
            points_xyz = points_gps.copy()
        IDs, points_xyz,points_2d =IDs[1:],points_xyz[1:],points_2d[1:]
        image_AR = gray.copy()

        camera_matrix = numpy.array(camera_matrix_init,dtype=numpy.float32)
        rvecs, tvecs, points_2d_check = tools_pr_geom.fit_pnp(points_xyz, points_2d,camera_matrix, dist)
        rvecs, tvecs = numpy.array(rvecs).flatten(), numpy.array(tvecs).flatten()

        if list_of_R is not None:
            RR = -numpy.sort(-numpy.array(list_of_R).flatten())
            colors_ln = tools_draw_numpy.get_colors(len(list_of_R), colormap=self.colormap_circles)[::-1]
            for rad,color_ln in zip(RR,colors_ln):
                image_AR = tools_render_CV.draw_compass(image_AR, camera_matrix, numpy.zeros(5), rvecs,tvecs, rad,color=color_ln.tolist())

        if virt_obj is not None:
            for p in points_xyz:
                cuboid_3d = p + self.construct_cuboid_v0(virt_obj)
                M = tools_pr_geom.compose_RT_mat(rvecs,tvecs,do_rodriges=True,do_flip=False,GL_style=False)
                P = tools_pr_geom.compose_projection_mat_4x4(camera_matrix[0, 0], camera_matrix[1, 1],camera_matrix[0, 2] / camera_matrix[0, 0],camera_matrix[1, 2] / camera_matrix[1, 1])
                image_AR = tools_draw_numpy.draw_cuboid(image_AR,tools_pr_geom.project_points_p3x4(cuboid_3d, numpy.matmul(P,M)))

        labels = ['ID %02d: %2.1f,%2.1f'%(pid, p[0], p[1]) for pid, p in zip(IDs,points_xyz)]
        image_AR = tools_draw_numpy.draw_ellipses(image_AR, [((p[0], p[1]), (25, 25), 0) for p in points_2d],color=(0, 0, 190), w=4,labels=labels)
        image_AR = tools_draw_numpy.draw_points(image_AR, points_2d_check, color=(0, 128, 255), w=8)

        return image_AR
# ----------------------------------------------------------------------------------------------------------------------
    def BEV_points(self, image, filename_points, do_shift_scale, target_W, target_H, dots_pr_meter =None,draw_points = True, list_of_R=None,cuboids_3d =None, points_2d = None):

        factor = 1.2
        gray = tools_image.desaturate(image,level=0.5)
        empty = numpy.full((target_H, target_W, 3), (255, 255, 255), dtype=numpy.uint8)
        image_R = empty.copy()

        marker_2d, marker_xy, IDs = self.load_points(filename_points)
        if do_shift_scale: marker_xy = self.shift_scale(marker_xy, marker_xy[0])

        scale, shift1, shift2, do_flip_v, shift_h = self.get_adjustment_params(marker_xy[:, :2], target_W, target_H)
        if dots_pr_meter is not None:
            scale=1/dots_pr_meter
        marker_xy = self.adjust_2d(marker_xy[:, :2], target_W, target_H, scale, shift1, shift2, do_flip_v, shift_h)
        marker_xy_ext = tools_draw_numpy.extend_view(marker_xy, target_H, target_W, factor)

        cuboids_xy_ext, cuboid_IDs = [], []
        if cuboids_3d is not None:
            for i, each in enumerate(cuboids_3d.XYZs):
                cuboid_3d = each.reshape((-1,3))[:4,[0,2,1]]
                cuboid_3d[:,0]*=-1
                cuboid_xy = self.adjust_2d(cuboid_3d[:, :2], target_W, target_H, scale, shift1, shift2, do_flip_v,shift_h)
                cuboids_xy_ext.append(tools_draw_numpy.extend_view(cuboid_xy, target_H, target_W, factor))
                cuboid_IDs.append(i)

            cuboid_IDs = numpy.array(cuboid_IDs)
            cuboids_xy_ext = numpy.array(cuboids_xy_ext)

        if list_of_R is not None and len(list_of_R) > 0:
            colors_ln = tools_draw_numpy.get_colors(len(list_of_R), colormap=self.colormap_circles)[::-1]
            RR = -numpy.sort(-numpy.array(list_of_R).flatten())
            center_ext = tools_draw_numpy.extend_view(marker_xy[0], target_H, target_W, factor)[0]
            colors_GB = numpy.arange(180, 255, (255 - 180) / len(list_of_R))[::-1]

            for rad, color_bg, color_ln in zip(RR, colors_GB, colors_ln):
                image_R = tools_draw_numpy.draw_circle(image_R, center_ext[1], center_ext[0], rad / scale / factor,color_bg, alpha_transp=0)
                ellipse = ((center_ext[0], center_ext[1]), ((2 * rad / scale / factor), 2 * rad / scale / factor), 0)
                image_R = tools_draw_numpy.draw_ellipses(image_R, [ellipse], color=color_ln.tolist(), w=1)
        else:
            colors_ln = tools_draw_numpy.get_colors(1, colormap=self.colormap_circles)[::-1]

        H = self.derive_homography(marker_2d[1:], marker_xy[1:])
        image_warped = cv2.warpPerspective(gray, H, (target_W, target_H), borderValue=(255, 255, 255))
        image_warped = tools_draw_numpy.extend_view_from_image(image_warped, factor=factor,color_bg=(255, 255, 255))
        image_BEV = tools_image.put_layer_on_image(image_R, image_warped, (255, 255, 255))

        points_2d_ext, points_IDs = [], []
        if points_2d is not None:
            for i, each in enumerate(points_2d.XYs):
                point_xy = cv2.perspectiveTransform(each.reshape((-1, 1, 2)).astype(numpy.float32), H).reshape((-1, 2))
                points_2d_ext.append(tools_draw_numpy.extend_view(point_xy, target_H, target_W, factor))
                points_IDs.append(i)

            points_IDs = numpy.array(points_IDs)
            points_2d_ext = numpy.array(points_2d_ext)


        if draw_points:
            image_BEV = tools_draw_numpy.draw_points(image_BEV, marker_xy_ext[:1], color=colors_ln[-1].tolist(), w=8)

        if len(cuboid_IDs) > 0:
            col_obj = tools_draw_numpy.get_colors(len(numpy.unique(cuboid_IDs)), colormap=self.colormap_objects)
            for i, id in enumerate(numpy.unique(cuboid_IDs)):
                p = cuboids_xy_ext[cuboid_IDs == id][0]
                image_BEV = tools_draw_numpy.draw_contours(image_BEV, p, color_fill=col_obj[i].tolist(),color_outline=col_obj[i].tolist(),transp_fill=0.3,transp_outline=1.0)
                image_BEV = tools_draw_numpy.draw_lines(image_BEV, numpy.array([[p[0,0],p[0,1],p[1,0],p[1,1]]]),color=col_obj[i].tolist(),w=5)

        if len(points_IDs) > 0:
            col_obj = tools_draw_numpy.get_colors(len(numpy.unique(points_IDs)), colormap=self.colormap_objects)
            for i, id in enumerate(numpy.unique(points_IDs)):
                p = points_2d_ext[points_IDs == id][0]
                if len(p)==1:
                    image_BEV = tools_draw_numpy.draw_points(image_BEV, p,col_obj[i].tolist(),w=12)
                else:
                    image_BEV = tools_draw_numpy.draw_contours(image_BEV, p, color_fill=col_obj[i].tolist(),color_outline=col_obj[i].tolist(), transp_fill=0.3,transp_outline=1.0)


        if draw_points:
            labels = ['ID %02d: %2.1f,%2.1f' % (pid, p[0], p[1]) for pid, p in zip(IDs, marker_xy)]
            image_BEV = tools_draw_numpy.draw_ellipses(image_BEV,[((p[0], p[1]), (25, 25), 0) for p in marker_xy_ext[1:]],color=(0, 0, 190), w=4, labels=labels[1:])

        return image_BEV
# ----------------------------------------------------------------------------------------------------------------------
    def save_obj_file(self, filename_out,flat_obj=(-1, -1, 0, +1, +1, 0)):
        import tools_wavefront
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
    def construct_cuboid(self, dim,rvec,tvec):
        d0, d1, d2 = dim[0], dim[1], dim[2]

        #x_corners = [+d0/2, +d0/2, -d0/2, -d0/2, +d0/2, +d0/2, -d0/2, -d0/2]
        #y_corners = [+d1/2, +d1/2, +d1/2, +d1/2, -d1/2, -d1/2, -d1/2, -d1/2]
        #z_corners = [+d2/2, -d2/2, -d2/2, +d2/2, +d2/2, -d2/2, -d2/2, +d2/2]

        x_corners = [-d0/2, -d0/2, +d0/2, +d0/2, +d0/2, +d0/2, -d0/2, -d0/2]
        y_corners = [-d1/2, +d1/2, -d1/2, +d1/2, -d1/2, +d1/2, -d1/2, +d1/2]
        z_corners = [-d2/2, -d2/2, -d2/2, -d2/2, +d2/2, +d2/2, +d2/2, +d2/2]

        X = numpy.array([x_corners, y_corners, z_corners], dtype=numpy.float32).T

        RT = tools_pr_geom.compose_RT_mat(rvec,tvec,do_rodriges=True,do_flip=False,GL_style=False)
        Xt = tools_pr_geom.apply_matrix_GL(RT, X)[:, :3]


        uu=0
        return Xt
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_fov(self, filename_image, filename_points, a_min=0.4, a_max=0.41, do_shift=True, zero_tvec = False, list_of_R = [], virt_obj=None, do_debug = False):

        base_name = filename_image.split('/')[-1].split('.')[0]
        image = cv2.imread(filename_image)
        W,H = image.shape[1], image.shape[0]
        gray = tools_image.desaturate(image)
        points_2d_all, points_xyz,IDs = self.load_points(filename_points)
        if do_shift:
            points_xyz = self.shift_scale(points_xyz, points_xyz[0])
        points_xyz, points_2d = points_xyz[1:], points_2d_all[1:]
        if len(points_xyz)<=3:
            return numpy.nan,numpy.full(3,numpy.nan),numpy.full(3,numpy.nan)

        err, rvecs, tvecs = [],[],[]
        a_fovs = numpy.arange(a_min, a_max, 0.005)

        for a_fov in a_fovs:
            camera_matrix = tools_pr_geom.compose_projection_mat_3x3(W, H, a_fov, a_fov)
            rvec, tvec, points_2d_check = tools_pr_geom.fit_pnp(points_xyz, points_2d, camera_matrix, numpy.zeros(5))

            if zero_tvec:
                rvec,tvec,points_2d_check = tools_pr_geom.fit_R(points_xyz,points_2d,camera_matrix,rvec,tvec)

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
                points_2d_check, jac = tools_pr_geom.project_points(points_xyz, rvecs[i], tvecs[i], camera_matrix, numpy.zeros(5))
                image_AR = tools_draw_numpy.draw_ellipses(gray, [((p[0], p[1]), (25, 25), 0) for p in points_2d],color=(0, 0, 190), w=4)
                image_AR = tools_draw_numpy.draw_points(image_AR, points_2d_check.reshape((-1,2)), color=color_markup, w=8)
                for rad in list_of_R:
                    image_AR = tools_render_CV.draw_compass(image_AR, camera_matrix, numpy.zeros(5), rvecs[i],tvecs[i], rad, color=color_markup)
                if virt_obj is not None:
                    for p in points_xyz:
                        image_AR = tools_draw_numpy.draw_cuboid(image_AR,tools_pr_geom.project_points(p+self.construct_cuboid_v0(virt_obj), rvecs[i], tvecs[i], camera_matrix, numpy.zeros(5))[0])

                cv2.imwrite(self.folder_out + base_name + '_%05d' % (a_fovs[i] * 1000) + '.png', image_AR)

        return a_fovs[idx_best], numpy.array(rvecs)[idx_best], numpy.array(tvecs)[idx_best]
# ----------------------------------------------------------------------------------------------------------------------
    def get_pretty_model_rotation(self,mat_model):

        if mat_model is None:return numpy.nan,numpy.nan,numpy.nan

        rvec, tvec = tools_pr_geom.decompose_to_rvec_tvec(mat_model)
        a_pitch_deg, a_yaw_deg, a_roll_deg =  rvec[[0,1,2]]*180/numpy.pi

        #check
        rvec_check = numpy.array((a_pitch_deg, a_yaw_deg, a_roll_deg)) * numpy.pi / 180
        mat_model_check3 = tools_pr_geom.compose_RT_mat(rvec_check, (0,0,0), do_rodriges=False, do_flip=False)



        return a_pitch_deg, a_yaw_deg, a_roll_deg
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_matrices_GL(self, image, rvec, tvec, a_fov, points_3d=None,virt_obj=None, do_debug=False):

        if numpy.any(numpy.isnan(rvec)) or numpy.any(numpy.isnan(tvec)):return numpy.full((4,4),numpy.nan), numpy.full((4,4),numpy.nan), numpy.full((4,4),numpy.nan)
        H,W = image.shape[:2]

        mR = tools_pr_geom.compose_RT_mat(rvec, (0, 0, 0), do_rodriges=True, do_flip=True, GL_style=True)
        mRT = tools_pr_geom.compose_RT_mat(rvec, tvec, do_rodriges=True, do_flip=True, GL_style=True)


        imR = numpy.linalg.inv(mR)
        mat_projection = tools_pr_geom.compose_projection_mat_4x4_GL(W, H, a_fov, a_fov)
        T = numpy.matmul(mRT, imR)

        if do_debug:
            import tools_GL3D
            cuboid_3d = self.construct_cuboid_v0(virt_obj)
            gray = tools_image.desaturate(cv2.resize(image, (W, H)))

            filename_obj = self.folder_out + 'temp.obj'
            self.save_obj_file(filename_obj, virt_obj)
            R = tools_GL3D.render_GL3D(filename_obj=filename_obj, W=W, H=H, do_normalize_model_file=False,is_visible=False, projection_type='P', textured=False)
            tools_IO.remove_file(filename_obj)

            cv2.imwrite(self.folder_out + 'AR0_GL_m1.png', R.get_image_perspective(rvec, tvec, a_fov, a_fov, mat_view_to_1=False, do_debug=True))
            cv2.imwrite(self.folder_out + 'AR0_GL_v1.png', R.get_image_perspective(rvec, tvec, a_fov, a_fov, mat_view_to_1=True, do_debug=True))
            cv2.imwrite(self.folder_out + 'AR0_CV_cube.png',tools_render_CV.draw_cube_numpy_MVP_GL(gray, mat_projection, numpy.eye(4), mRT,numpy.eye(4), points_3d=cuboid_3d))
            cv2.imwrite(self.folder_out + 'AR0_CV_pnts.png',tools_render_CV.draw_points_numpy_MVP_GL(points_3d,gray, mat_projection, numpy.eye(4), mRT, numpy.eye(4),w = 8))

            #cv2.imwrite(self.folder_out + 'AR1_CV_pnts.png',tools_render_CV.draw_points_numpy_MVP_GL(points_3dt, gray, mat_projection, numpy.eye(4), mR,numpy.eye(4), w=8))
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
    def derive_cuboid_3d(self, mat_camera_3x3, r_vec, t_vec, points_2D,LWH, base_name='', do_debug=False):

        target_centr_top = numpy.mean(points_2D[[0, 2, 4, 6]], axis=0)
        target_centr_bottom = numpy.mean(points_2D[[1, 3, 5, 7]], axis=0)
        target_centr_bottom_3D = tools_pr_geom.reverce_project_points_Z0([target_centr_bottom], r_vec, t_vec, mat_camera_3x3,numpy.zeros(5))[0]

        diffs, Hs = [], numpy.arange(0, 10, 0.1)
        for h in Hs:
            centr_top_2D, _ = tools_pr_geom.project_points(target_centr_bottom_3D - numpy.array((0, 0, h)), r_vec,t_vec, mat_camera_3x3, numpy.zeros(5))
            diffs.append(numpy.abs(centr_top_2D[0][1] - target_centr_top[1]))

        best_H = Hs[numpy.argmin(diffs)]

        points_3D = tools_pr_geom.reverce_project_points_Z0(points_2D[[1, 3, 5, 7]], r_vec, t_vec, mat_camera_3x3,numpy.zeros(5))

        d = numpy.array([numpy.linalg.norm(p) for p in points_3D])


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
    def centralize_points(self,filename_image, filename_points_in,filename_points_out, mat_camera, rvec, tvec,do_debug=False):

        image = cv2.imread(filename_image)
        H, W = image.shape[:2]

        points_2d_all, points_gps_all, IDs = self.load_points(filename_points_in)
        points_xyz = self.shift_scale(points_gps_all, points_gps_all[0])

        M    = tools_pr_geom.compose_RT_mat(rvec,tvec,do_rodriges=True,do_flip=False,GL_style=False)
        M_GL = tools_pr_geom.compose_RT_mat(rvec, tvec, do_rodriges=True, do_flip=True, GL_style=True)
        P    = tools_pr_geom.compose_projection_mat_4x4(mat_camera[0, 0], mat_camera[1, 1],mat_camera[0, 2] / mat_camera[0, 0],mat_camera[1, 2] / mat_camera[1, 1])

        loss,a_yaws = [],numpy.arange(0,360,1)
        cuboid_3d_centred = self.construct_cuboid_v0((-1,+30,0,+1,+31,1))

        for a_yaw in a_yaws:
            R_CV = tools_pr_geom.compose_RT_mat((0, 0, a_yaw * numpy.pi / 180), (0, 0, 0), do_rodriges=True,do_flip=False, GL_style=False)
            cuboid_r_CV = tools_pr_geom.apply_matrix_GL(R_CV.T, cuboid_3d_centred)
            points_2d = tools_pr_geom.project_points_p3x4(cuboid_r_CV, numpy.matmul(P, M),check_reverce=True)
            if not numpy.isnan(points_2d[:, 0].mean()):loss.append(numpy.abs(points_2d[:, 0].mean()-W/2))
            else:loss.append(numpy.inf)
            #if do_debug:cv2.imwrite(self.folder_out + 'AR2_GL_%03d.png' % (a_yaw),tools_draw_numpy.draw_cuboid(gray,points_2d,color=(0, 0, 255)))

        a_yaw = a_yaws[numpy.argmin(loss)]



        R_CV = tools_pr_geom.compose_RT_mat((0, 0, -a_yaw * numpy.pi / 180), (0, 0, 0), do_rodriges=True, do_flip=True,GL_style=False)
        R_GL = tools_pr_geom.compose_RT_mat((0, 0, -a_yaw * numpy.pi / 180), (0, 0, 0), do_rodriges=True, do_flip=True,GL_style=True)
        iR_CV = numpy.linalg.inv(R_CV)
        iR_GL = numpy.linalg.inv(R_GL)

        points_xyz_t_CV = tools_pr_geom.apply_matrix_GL(R_CV.T, points_xyz)
        points_xyz_t_GL = tools_pr_geom.apply_matrix_GL(R_GL, points_xyz)
        points_xyz_t_CV[:,1]*=-1
        points_xyz_t_GL[:,1]*=-1

        flip = numpy.identity(4)
        flip[1][1] = -1

        M_new_CV = numpy.matmul(M, iR_CV)
        M_new_CV = numpy.matmul(M_new_CV, flip)
        M_new_GL = numpy.matmul(iR_GL, M_GL)
        M_new_GL = numpy.matmul(flip,M_new_GL)

        #check
        gray = tools_image.desaturate(image)
        fov = mat_camera[0, 2] / mat_camera[0, 0]
        mat_projection = tools_pr_geom.compose_projection_mat_4x4_GL(W, H, fov, fov)
        # cv2.imwrite(self.folder_out + 'GL_0.png',tools_render_CV.draw_points_numpy_MVP_GL(points_xyz, gray, mat_projection, M_GL, numpy.eye(4),numpy.eye(4),w=8))
        # cv2.imwrite(self.folder_out + 'GL_1.png',tools_render_CV.draw_points_numpy_MVP_GL(points_xyz_t_GL, gray, mat_projection, M_new_GL, numpy.eye(4),numpy.eye(4),w=8))
        # cv2.imwrite(self.folder_out + 'CV_0.png',tools_draw_numpy.draw_points(gray,tools_pr_geom.project_points_p3x4(points_xyz, numpy.matmul(P, M)),color=(0, 0, 255),w=10))
        # cv2.imwrite(self.folder_out + 'CV_1.png',tools_draw_numpy.draw_points(gray,tools_pr_geom.project_points_p3x4(points_xyz_t_CV, numpy.matmul(P, M_new_CV)),color=(0, 0, 255), w=10))

        self.save_points(filename_points_out, IDs[1:],points_2d_all[1:], points_xyz_t_CV[1:],W,H)

        mat_camera_new = numpy.matmul(P, M_new_CV)

        return mat_camera_new
# ----------------------------------------------------------------------------------------------------------------------
