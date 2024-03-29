import cv2
import numpy
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_image
import tools_IO
from CV import tools_calibrate
from CV import tools_pr_geom
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
    def load_df_points(self, filename_in,do_shift_scale=False):
        df = pd.read_csv(filename_in,sep='\t')
        if do_shift_scale:
            df.iloc[:,3:6] = self.shift_scale(df.iloc[:,3:6].values, df.iloc[0,3:6].values)
            #df=df.iloc[1:,:]

        return df
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
    def shift_scale(self, points_gnss_3d, origin_xy, orignin_z=0.0, scale_factor_xy=None):
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
    def adjust_2d(self, xy0, target_W, target_H, scale, shift1, shift2, do_flip_v, shift_h=None):

        xy = xy0.copy()

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
    def AR_points(self, image, df_points, camera_matrix,M,list_of_R =None,virt_obj=None):

        points_xyz = df_points.iloc[:,3:].values.astype(numpy.float32)
        points_2d = df_points.iloc[:,1:3].values.astype(numpy.float32)
        #points_2d_check = tools_pr_geom.project_points(points_xyz, rvec, tvec, camera_matrix)[0]
        points_2d_check = tools_pr_geom.project_points_M(points_xyz, M, camera_matrix)
        IDs = df_points.iloc[:,0].values
        image_AR = tools_image.desaturate(image)

        if list_of_R is not None:
            RR = -numpy.sort(-numpy.array(list_of_R).flatten())
            colors_ln = tools_draw_numpy.get_colors(len(list_of_R), colormap=self.colormap_circles)[::-1]
            for rad,color_ln in zip(RR,colors_ln):
                image_AR = tools_render_CV.draw_compass(image_AR, camera_matrix, M, rad,color=color_ln.tolist())

        if virt_obj is not None:
            for p in points_xyz:
                cuboid_3d = p + self.construct_cuboid_v0(virt_obj)
                #M = tools_pr_geom.compose_RT_mat(rvec,tvec,do_rodriges=True,do_flip=False,GL_style=False)
                P = tools_pr_geom.compose_projection_mat_4x4(camera_matrix[0, 0], camera_matrix[1, 1],camera_matrix[0, 2] / camera_matrix[0, 0],camera_matrix[1, 2] / camera_matrix[1, 1])
                image_AR = tools_draw_numpy.draw_cuboid(image_AR,tools_pr_geom.project_points_p3x4(cuboid_3d, numpy.matmul(P,M)))

        labels = ['ID %02d: %2.1f,%2.1f'%(pid, p[0], p[1]) for pid, p in zip(IDs,points_xyz)]
        image_AR = tools_draw_numpy.draw_ellipses(image_AR, [((p[0], p[1]), (25, 25), 0) for p in points_2d],color=(0, 0, 190), w=4,labels=labels)
        image_AR = tools_draw_numpy.draw_points(image_AR, points_2d_check, color=(0, 128, 255), w=8)

        return image_AR
# ----------------------------------------------------------------------------------------------------------------------
    def draw_ticks(self,image_R,marker_xy,major_ticks,minor_ticks,target_H, target_W,factor,scale):

        if major_ticks is not None and len(major_ticks) > 0:
            RR = -numpy.sort(-numpy.array(major_ticks).flatten())
            center_ext = tools_draw_numpy.extend_view(marker_xy[0], target_H, target_W, factor)[0]
            colors_GB = numpy.arange(180, 255, (255 - 180) / len(major_ticks))[::-1]

            for rad, color_bg in zip(RR, colors_GB):
                image_R = tools_draw_numpy.draw_circle(image_R, center_ext[1], center_ext[0], rad / scale / factor,color_bg, alpha_transp=0)

        # if minor_ticks is not None and len(minor_ticks) > 0:
        #     RR = -numpy.sort(-numpy.array(minor_ticks).flatten())
        #     center_ext = tools_draw_numpy.extend_view(marker_xy[0], target_H, target_W, factor)[0]
        #     colors_GB = numpy.arange(180, 255, (255 - 180) / len(minor_ticks))[::-1]
        #
        #     for rad, color_bg in zip(RR, colors_GB):
        #         ellipse = ((center_ext[0], center_ext[1]), ((2 * rad / scale / factor), 2 * rad / scale / factor), 0)
        #         image_R = tools_draw_numpy.draw_ellipses(image_R, [ellipse], color=color_bg, w=1)


        if major_ticks is not None and len(major_ticks) > 0:
            RR = -numpy.sort(-numpy.array(major_ticks).flatten())
            center_ext = tools_draw_numpy.extend_view(marker_xy[0], target_H, target_W, factor)[0]

            for rad in RR:
                ellipse = ((center_ext[0], center_ext[1]), ((2 * rad / scale / factor), 2 * rad / scale / factor), 0)
                image_R = tools_draw_numpy.draw_ellipses(image_R, [ellipse], color=(128,128,128), w=1)

        return image_R
# ----------------------------------------------------------------------------------------------------------------------
    def prepare_assets(self, marker_2d,marker_xy, target_W, target_H, dots_pr_meter, factor, major_ticks=None, minor_ticks=None, cuboids_3d=None, points_2d=None):

        image_R = numpy.full((target_H, target_W, 3), 255, dtype=numpy.uint8)

        #if do_shift_scale:
        #marker_xy = self.shift_scale(marker_xy, marker_xy[0])

        scale, shift1, shift2, do_flip_v, shift_h = self.get_adjustment_params(marker_xy[:, :2], target_W, target_H)
        if dots_pr_meter is not None:
            scale = 1 / dots_pr_meter


        marker_xy = self.adjust_2d(marker_xy[:, :2], target_W, target_H, scale, shift1, shift2, do_flip_v, shift_h)
        marker_xy_ext = tools_draw_numpy.extend_view(marker_xy, target_H, target_W, factor)

        H = self.derive_homography(marker_2d[1:], marker_xy[1:])
        image_R = self.draw_ticks(image_R, marker_xy, major_ticks, minor_ticks, target_H, target_W, factor, scale)

        cuboids_xy_ext  = []
        if cuboids_3d is not None:
            for i, each in enumerate(cuboids_3d.XYZs):
                cuboid_3d = each.reshape((-1, 3))[:4, [0, 2, 1]]
                cuboid_3d[:, 0] *= -1
                cuboid_xy = self.adjust_2d(cuboid_3d[:, :2], target_W, target_H, scale, shift1, shift2, do_flip_v,shift_h)
                cuboids_xy_ext.append(tools_draw_numpy.extend_view(cuboid_xy, target_H, target_W, factor))

            cuboids_xy_ext = numpy.array(cuboids_xy_ext)

        points_2d_ext  = []
        if points_2d is not None:
            for i, each in enumerate(points_2d.XYs):
                point_xy = cv2.perspectiveTransform(each.reshape((-1, 1, 2)).astype(numpy.float32), H).reshape((-1, 2))
                points_2d_ext.append(tools_draw_numpy.extend_view(point_xy, target_H, target_W, factor))

            points_2d_ext = numpy.array(points_2d_ext)

        return marker_xy,marker_xy_ext, cuboids_xy_ext, points_2d_ext, image_R, H
# ----------------------------------------------------------------------------------------------------------------------
    def BEV_points(self, image, df_points, target_W, target_H,
                   dots_pr_meter =None, draw_points = True, major_ticks=None,minor_ticks=None, cuboids_3d =None, points_2d = None,
                   draw_hits=False,col_hit=(128, 255, 0),col_miss=(0,64,255),iou=0.3):

        factor = 1.2
        gray = tools_image.desaturate(image,level=0.5)

        IDs = df_points.iloc[:, 0].values
        marker_xy_original = df_points.iloc[:, 3:].values.astype(numpy.float32)
        marker_2d = df_points.iloc[:, 1:3].values.astype(numpy.float32)

        marker_xy000, marker_xy_ext, cuboids_xy_ext, points_2d_ext, image_R, H = \
            self.prepare_assets(marker_2d,marker_xy_original, target_W, target_H, dots_pr_meter, factor, major_ticks, minor_ticks, cuboids_3d, points_2d)

        numpy.set_printoptions(precision=4)

        image_warped = cv2.warpPerspective(gray, H, (target_W, target_H), borderValue=(255, 255, 255))


        image_warped = tools_draw_numpy.extend_view_from_image(image_warped, factor=factor,color_bg=(255, 255, 255))
        image_BEV = tools_image.put_layer_on_image(image_R, image_warped, (255, 255, 255))

        if draw_points:
            colors_ln = tools_draw_numpy.get_colors(1, colormap=self.colormap_circles)[::-1]
            image_BEV = tools_draw_numpy.draw_points(image_BEV, marker_xy_ext[:1], color=colors_ln[-1].tolist(), w=8)

        if len(cuboids_xy_ext) > 0:
            col_obj = tools_draw_numpy.get_colors(len(cuboids_xy_ext), colormap=self.colormap_objects)
            for p,clr,metainfo in zip(cuboids_xy_ext,col_obj,cuboids_3d.metainfo):
                if draw_hits and metainfo is not None:
                    clr = numpy.array(col_hit) if float(metainfo) >= iou else numpy.array(col_miss)

                image_BEV = tools_draw_numpy.draw_contours(image_BEV, p, color_fill=clr.tolist(),color_outline=clr.tolist(),transp_fill=0.3,transp_outline=1.0)
                image_BEV = tools_draw_numpy.draw_lines(image_BEV, numpy.array([[p[0,0],p[0,1],p[1,0],p[1,1]]]),color=clr.tolist(),w=5)

        if len(points_2d_ext) > 0:
            col_obj = tools_draw_numpy.get_colors(len(points_2d_ext), colormap=self.colormap_objects)
            for p,clr in zip(points_2d_ext,col_obj):
                if len(p)==1:
                    image_BEV = tools_draw_numpy.draw_points(image_BEV, p,clr.tolist(),w=12)
                else:
                    image_BEV = tools_draw_numpy.draw_contours(image_BEV, p, color_fill=clr.tolist(),color_outline=clr.tolist(), transp_fill=0.3,transp_outline=1.0)

        if draw_points:
            labels = ['ID %02d: %2.1f,%2.1f' % (pid, p[0], p[1]) for pid, p in zip(IDs, marker_xy_original)]
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
    def evaluate_fov(self, image,df_points,a_min=0.4, a_max=0.41, zero_tvec = False, list_of_R = [], virt_obj=None, do_debug = False):

        H,W = image.shape[:2]
        gray = tools_image.desaturate(image)
        err, rvecs, tvecs = [],[],[]
        a_fovs = numpy.arange(a_min, a_max, 0.005) if a_min!=a_max else [a_min]
        points_xyz = df_points.iloc[:,3:].values.astype(numpy.float32)
        points_2d = df_points.iloc[:,1:3].values.astype(numpy.float32)

        for a_fov in a_fovs:
            camera_matrix = tools_pr_geom.compose_projection_mat_3x3(W, H, a_fov, a_fov)
            rvec, tvec, points_2d_check = tools_pr_geom.fit_pnp(points_xyz, points_2d, camera_matrix, numpy.zeros(5))

            if zero_tvec:
                rvec,tvec,points_2d_check = tools_pr_geom.fit_R(points_xyz,points_2d,camera_matrix,rvec)

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

                cv2.imwrite(self.folder_out + 'evaluate_fov_%05d' % (a_fovs[i] * 1000) + '.png', image_AR)

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
    def centralize_points(self,image, df_points,filename_points_out, mat_camera, M_GL,M_CV,do_debug=False):

        H, W = image.shape[:2]
        gray = tools_image.desaturate(image)
        points_xyz = df_points.iloc[:, 3:].values.astype(numpy.float32)
        P    = tools_pr_geom.compose_projection_mat_4x4(mat_camera[0, 0], mat_camera[1, 1],mat_camera[0, 2] / mat_camera[0, 0],mat_camera[1, 2] / mat_camera[1, 1])
        cuboid_3d_centred = self.construct_cuboid_v0((-1, +29, 0, +1, +31, 0))
        flip = numpy.identity(4)
        flip[1][1] = -1

        loss_g = numpy.inf
        a_yaw_g = None
        for a_yaw in numpy.arange(0,360,1):
            R_GL_cand = tools_pr_geom.compose_RT_mat((0, 0,  a_yaw * numpy.pi / 180), (0, 0, 0), do_rodriges=True,do_flip=True , GL_style=True)
            R_CV_cand = tools_pr_geom.compose_RT_mat((0, 0,  a_yaw * numpy.pi / 180), (0, 0, 0), do_rodriges=True,do_flip=False, GL_style=False)
            cuboid_r_GL = tools_pr_geom.apply_matrix_GL(R_GL_cand.T, cuboid_3d_centred)
            cuboid_r_CV = tools_pr_geom.apply_matrix_GL(R_CV_cand  , cuboid_3d_centred)
            points_2d_cv = tools_pr_geom.project_points_p3x4(cuboid_r_CV, numpy.matmul(P, M_CV),check_reverce=True)
            if not numpy.isnan(points_2d_cv[:, 0].mean()):
                loss = numpy.abs(points_2d_cv[:, 0].mean()-W/2)
                if loss<loss_g:
                    loss_g,R_CV,R_GL = loss,R_CV_cand,R_GL_cand
                    a_yaw_g = a_yaw
                #if do_debug:cv2.imwrite(self.folder_out + 'AR2_CV_%03d.png' % (a_yaw),tools_draw_numpy.draw_cuboid(gray,points_2d_cv,color=(0, 0, 255)))

        I_GL = tools_pr_geom.compose_RT_mat((0, 0,  a_yaw_g * numpy.pi / 180), (0, 0, 0), do_rodriges=True,do_flip=True, GL_style=True)
        I_CV = tools_pr_geom.compose_RT_mat((0, 0, -a_yaw_g * numpy.pi / 180), (0, 0, 0), do_rodriges=True,do_flip=False, GL_style=False)

        M_GL_new = numpy.matmul(R_GL, M_GL)
        #M_GL_new = numpy.matmul(flip, M_GL_new)
        M_CV_new = numpy.matmul(M_CV, R_CV)
        #M_CV_new = numpy.matmul(M_CV_new, flip)

        points_xyz_t_GL = tools_pr_geom.apply_matrix_GL(I_GL.T  , points_xyz)
        points_xyz_t_CV = tools_pr_geom.apply_matrix_GL(I_CV    , points_xyz)

        #check
        # fov = mat_camera[0, 2] / mat_camera[0, 0]
        # mat_projection = tools_pr_geom.compose_projection_mat_4x4_GL(W, H, fov, fov)
        # cv2.imwrite(self.folder_out + 'GL_0.png',tools_render_CV.draw_points_numpy_MVP_GL(points_xyz     , gray, mat_projection, M_GL    , numpy.eye(4),numpy.eye(4),w=8))
        # cv2.imwrite(self.folder_out + 'GL_1.png',tools_render_CV.draw_points_numpy_MVP_GL(points_xyz_t_GL, gray, mat_projection, M_GL_new, numpy.eye(4),numpy.eye(4),w=8))
        # cv2.imwrite(self.folder_out + 'CV_0.png',tools_draw_numpy.draw_points(gray,tools_pr_geom.project_points_p3x4(points_xyz, numpy.matmul(P, M_CV)),color=(0, 0, 255),w=10))
        # cv2.imwrite(self.folder_out + 'CV_1.png',tools_draw_numpy.draw_points(gray,tools_pr_geom.project_points_p3x4(points_xyz_t_CV, numpy.matmul(P, M_CV_new)),color=(0, 0, 255), w=10))
        # cv2.imwrite(self.folder_out + 'CV_2.png',tools_draw_numpy.draw_points(gray, tools_pr_geom.project_points_M  (points_xyz_t_CV,M_CV_new,mat_camera), color=(0, 0, 255), w=10))
        self.save_points(filename_points_out, df_points.iloc[:, 0].values,df_points.iloc[:, 1:3].values, points_xyz_t_CV,W,H)

        return mat_camera,M_CV_new
# ----------------------------------------------------------------------------------------------------------------------
