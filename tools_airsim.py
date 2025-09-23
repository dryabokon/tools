#https://github.com/Cosys-Lab/Cosys-AirSim
# ---------------------------------------------------------------------------------------------------------------------
import cv2
import math
import json
import numpy
import socket
# ---------------------------------------------------------------------------------------------------------------------
import airsim
# ---------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_render_GL
import tools_draw_numpy
import tools_GL3D_light
# ---------------------------------------------------------------------------------------------------------------------
class tools_airsim:
    def __init__(self,ip_airsim,camera_rad_yaw_pitch_roll,fov,height_abs_BEV=100):
        self.fov = fov

        if ip_airsim is None:
            self.is_available = False
            self.W, self.H = 640, 480
            return

        if not self.check_is_available(ip_airsim, 41451):
            print(f'AirSim not available at {ip_airsim}')
            self.is_available = False
            self.W, self.H = 640, 480
            return

        self.is_available = True

        self.airsim_client = airsim.MultirotorClient(ip=ip_airsim)
        self.airsim_client.confirmConnection()

        self.dct_settings = self.get_settings()
        self.vehicle_name = self.airsim_client.listVehicles()[0]
        self.virtual_camera_name = self.get_virtual_camera_name()

        self.set_fov(self.fov)

        self.W = int(self.dct_settings['CameraDefaults']['CaptureSettings'][0]['Width'])
        self.H = int(self.dct_settings['CameraDefaults']['CaptureSettings'][0]['Height'])

        self.init_renderer(self.W, self.H, self.fov)
        self.height_abs_BEV = height_abs_BEV
        self.init_mat(90, self.height_abs_BEV)

        self.camera_rad_yaw_pitch_roll = camera_rad_yaw_pitch_roll
        self.new_setup(camera_rad_yaw_pitch_roll)

        return
    # ---------------------------------------------------------------------------------------------------------------------
    def check_is_available(self,ip_airsim,port=41451):
        try:
            sock = socket.create_connection((ip_airsim, port), timeout=2)
            sock.close()
            return True
        except socket.error as e:
            return False
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def new_setup(self,camera_rad_yaw_pitch_roll):

        yaw, pitch,roll = camera_rad_yaw_pitch_roll

        self.airsim_client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(+0.30, 0.0, -0.35),airsim.to_quaternion(pitch, 0.0, 0.0)))
        #self.airsim_client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(+0.0, 0.0, -0.0),airsim.to_quaternion(pitch, 0.0, 0.0)))
        self.airsim_client.simSetCameraPose("1", airsim.Pose(airsim.Vector3r(0.0, 0.0, -2.5),airsim.to_quaternion(+numpy.pi / 8, 0.0, 0)),vehicle_name=self.virtual_camera_name)
        self.airsim_client.simSetCameraPose("2", airsim.Pose(airsim.Vector3r(0.0, 0.0, -self.height_abs_BEV),airsim.to_quaternion(-numpy.pi / 2, 0, 0.0)))

        self.image_request_ego    = airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        self.image_request_ground = airsim.ImageRequest("1", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        self.image_request_BEV    = airsim.ImageRequest("2", airsim.ImageType.Scene, pixels_as_float=False, compress=False)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def get_settings(self):
        return json.loads(self.airsim_client.client.call('getSettingsString'))
# ---------------------------------------------------------------------------------------------------------------------
    def get_virtual_camera_name(self):
        if 'Vehicles' not in self.dct_settings: return ''

        for v in self.airsim_client.listVehicles():
            if v in ['PX4_2','Cam','Camera']:
                return v

        return ''
    # ---------------------------------------------------------------------------------------------------------------------
    def get_airsim_camera_pose(self):
        xyz = self.get_drone_position()
        yaw, pitch, roll = self.get_drone_orientation()
        orientation = airsim.to_quaternion(pitch, roll, yaw)
        pose_constructed = airsim.Pose(airsim.Vector3r(xyz[0], xyz[1], xyz[2]),orientation)
        return pose_constructed
# ---------------------------------------------------------------------------------------------------------------------
    def quaternion_to_euler_angles(self, w, x, y, z):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        pitch = math.asin(t2)
        roll = math.atan2(t0, t1)
        return [yaw, pitch, roll]
# ---------------------------------------------------------------------------------------------------------------------
    def euler_angles_to_quaternion(self, yaw, pitch, roll):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        return [w, x, y, z]
# ---------------------------------------------------------------------------------------------------------------------
    def get_fov(self):
        return self.airsim_client.simGetCameraInfo('').fov
    # ---------------------------------------------------------------------------------------------------------------------
    def set_fov(self, fov, camera_id="0"):

        if not isinstance(fov, (int, float)) or not (0 < fov <= 180):
            raise ValueError("Field of view (fov) must be a number between 0 and 180 degrees.")

        if self.airsim_client is not None and self.is_available:
            try:
                self.airsim_client.simSetCameraFov(camera_id, fov)
                self.fov = fov
            except RuntimeError as e:
                pass
                #print(f"RuntimeError occurred in simSetCameraFov: {e}")
            except BufferError as be:
                pass
                #print(f"BufferError during AirSim RPC communication: {be}")
            except Exception as ex:
                pass
                #print(f"An unexpected error occurred: {ex}")
        else:
            pass
            #print("AirSim client is not available or not connected.")
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def update_position_camera_based(self,position,roll_pitch_yaw):

        if len(roll_pitch_yaw)==3:
            roll, pitch, yaw = roll_pitch_yaw
            pose = airsim.Pose(airsim.Vector3r(position[0], position[1], position[2]), airsim.to_quaternion(pitch,roll,yaw))
        else:
            q = airsim.Quaternionr(roll_pitch_yaw[0], roll_pitch_yaw[1], roll_pitch_yaw[2], roll_pitch_yaw[3])
            pose = airsim.Pose(airsim.Vector3r(position[0], position[1], position[2]), q)

        self.airsim_client.simSetCameraPose("0", pose)
        self.image_request_ego = airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)

        return
    # ---------------------------------------------------------------------------------------------------------------------
    def get_drone_position(self):
        pos = self.airsim_client.simGetObjectPose(self.vehicle_name).position
        return numpy.array([pos.x_val, pos.y_val, pos.z_val])
    # ---------------------------------------------------------------------------------------------------------------------
    def get_drone_orientation(self):
        orientation = self.airsim_client.simGetObjectPose(self.airsim_client.listVehicles()[0]).orientation
        return self.quaternion_to_euler_angles(orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val)
    # ----------------------------------------------------------------------------------------------------------------------
    def get_image(self, cam='ego'):
        try:
            if   cam=='ego'   :response = self.airsim_client.simGetImages([self.image_request_ego])[0]
            elif cam=='ground':response = self.airsim_client.simGetImages([self.image_request_ground],vehicle_name=self.virtual_camera_name)[0]
            elif cam=='BEV'   :response = self.airsim_client.simGetImages([self.image_request_BEV])[0]
            else              :response = self.airsim_client.simGetImages([self.image_request_ego])[0]
            image = numpy.frombuffer(response.image_data_uint8, dtype=numpy.uint8).reshape(response.height, response.width,3)
        except:
            image = numpy.full((self.H, self.W, 3), 128, dtype=numpy.uint8) + numpy.random.randint(0, 32,(self.H, self.W, 3),dtype=numpy.uint8) - 32
            print(f'Error:get_image')

        return image
    # ----------------------------------------------------------------------------------------------------------------------
    def init_renderer(self, W, H, fov):
        self.R = tools_GL3D_light.render_GL3D(filename_obj=None, W=W, H=H, do_normalize_model_file=False, textured=True,
                                        is_visible=False, projection_type='P', cam_fov_deg=fov, scale=(1, 1, 1),
                                        eye=(0, 0, 0), target=(0, -1, 0), up=(-1, 0, 0))
        return

# ---------------------------------------------------------------------------------------------------------------------
    def init_mat(self, fov, Z):
        self.pose_BEV_AR_deg = numpy.array((0, -90, 0))

        self.get_AR_image(fov, numpy.array((0.0, 0.0, -Z)), self.pose_BEV_AR_deg * numpy.pi / 180.0)
        self.mat_projection_BEV = self.R.mat_projection
        self.mat_view_BEV = self.R.mat_view
        self.mat_model_BEV = self.R.mat_model
        self.mat_trns_BEV = self.R.mat_trns
        return

# ---------------------------------------------------------------------------------------------------------------------
    def airsim_to_GL(self, cam_translation, cam_rotation):
        cam_translation_GL = numpy.array((-cam_translation[1], cam_translation[2], cam_translation[0]))
        cam_rotation_GL = numpy.array((cam_rotation[1], cam_rotation[2], -cam_rotation[0]))
        return cam_translation_GL, cam_rotation_GL

# ---------------------------------------------------------------------------------------------------------------------
    def get_AR_image(self, cam_fov, cam_translation_airsim, cam_rotation_airsim):
        cam_translation_GL, cam_rotation_GL = self.airsim_to_GL(cam_translation_airsim, cam_rotation_airsim)

        self.R.cam_fov_deg = cam_fov
        self.R.reset_view()
        self.R.translate_view(cam_translation_GL)
        self.R.rotate_view(cam_rotation_GL)
        image = self.R.get_image(do_debug=True)
        return image
# ---------------------------------------------------------------------------------------------------------------------
    def draw_camera_on_BEV(self,image,ground_points3d,shift,color):
        H, W = image.shape[:2]
        transperency = 0.8

        if len(ground_points3d)>0:
            points_3d = numpy.array(ground_points3d).reshape((-1, 3))
            if shift is not None:
                points_3d -= numpy.array(shift).reshape((1, 3))

            points_3d[:, 1] *= -1
            points_3d[:, 2] = 0
            points_2d = tools_render_GL.project_points_MVP_GL(points_3d, W, H, self.mat_projection_BEV, self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV).reshape((-1, 2))
            image = tools_draw_numpy.draw_contours_cv(image, points_2d[[2,1,3,0,2]], color=color, w=-1, transperency=transperency)

        return image
# ---------------------------------------------------------------------------------------------------------------------
    def find_ray_intersection(self,i1, j1, i2, j2, W, H):
        def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:return None
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return px, py

        def is_point_in_direction(p1, p2, p_intersection):
            direction = numpy.array([p2[0] - p1[0], p2[1] - p1[1]])
            to_intersection = numpy.array([p_intersection[0] - p1[0], p_intersection[1] - p1[1]])
            return numpy.dot(direction, to_intersection) > 0

        bbox_edges = [(0, 0, W, 0), (W, 0, W, H),(W, H, 0, H),(0, H, 0, 0)]
        intersection_points = []
        for edge in bbox_edges:
            intersection = line_intersection(i1, j1, i2, j2, edge[0], edge[1], edge[2], edge[3])
            if intersection is not None:
                x, y = intersection
                if 0 <= x <= W and 0 <= y <= H and is_point_in_direction((i1, j1), (i2, j2), (x, y)):
                    intersection_points.append(intersection)

        return intersection_points
    # ---------------------------------------------------------------------------------------------------------------------
    def draw_contour_on_BEV(self, image,points_3d,color):
        if points_3d is None:return image
        points_3d = numpy.array(points_3d)
        points_3d[:, 1] *= -1

        H, W = image.shape[:2]
        transperency = 0.8
        points_2d = tools_render_GL.project_points_MVP_GL(points_3d.reshape((-1, 3)), W, H, self.mat_projection_BEV,self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV)
        image = tools_draw_numpy.draw_contours(image, points_2d.reshape((-1, 2)), color=color, w=-1,transperency=transperency)
        return image
# ---------------------------------------------------------------------------------------------------------------------
    def draw_lines_on_BEV(self, image, points_3d_start,points_3d_end, shift=None, color=(0, 0, 0), w=1):

        if points_3d_start is None:return image
        points_3d_start = numpy.array(points_3d_start).reshape((-1, 3))
        if points_3d_start.shape[0] == 0: return image

        if shift is not None:
            points_3d_start-=numpy.array(shift).reshape((1,3))

        points_3d_start[:, 1] *= -1

        if points_3d_end is None:return image
        points_3d_end = numpy.array(points_3d_end).reshape((-1, 3))
        if points_3d_end.shape[0] == 0: return image

        if shift is not None:
            points_3d_end-=numpy.array(shift).reshape((1,3))

        points_3d_end[:, 1] *= -1

        H, W = image.shape[:2]
        points_2d_start = tools_render_GL.project_points_MVP_GL(points_3d_start.reshape((-1, 3)), W, H, self.mat_projection_BEV,self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV)
        points_2d_end   = tools_render_GL.project_points_MVP_GL(points_3d_end.reshape((-1, 3))  , W, H, self.mat_projection_BEV,self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV)

        for p1,p2 in zip(points_2d_start,points_2d_end):
            p = self.find_ray_intersection(p1[0], p1[1], p2[0], p2[1], W, H)
            if len(p)>0:
                image = tools_draw_numpy.draw_line_fast(image, p1[1], p1[0], p[0][1], p[0][0], color, w=w)

        return image
# ---------------------------------------------------------------------------------------------------------------------
    def draw_points_on_BEV(self, image,points_3d,shift=None,color=(0,0,0),w=4,do_lines=False):
        if points_3d is None:return image
        points_3d = numpy.array(points_3d).reshape((-1, 3))
        if points_3d.shape[0] == 0: return image

        if shift is not None:
            points_3d-=numpy.array(shift).reshape((1,3))

        points_3d[:, 1] *= -1
        points_3d[:, 2] = 0

        H, W = image.shape[:2]
        points_2d = tools_render_GL.project_points_MVP_GL(points_3d.reshape((-1, 3)), W, H, self.mat_projection_BEV,self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV)
        image = tools_draw_numpy.draw_points_fast(image, points_2d,color=color,w=w)

        if do_lines:
            for i in range(len(points_2d)-1):
                image = tools_draw_numpy.draw_line_fast(image, points_2d[i][1], points_2d[i][0], points_2d[i+1][1], points_2d[i+1][0], color,w=1)

        return image
# ---------------------------------------------------------------------------------------------------------------------
    def draw_grid_on_BEV(self, image,origin, shift=None,color=(64,64,64),w=1):
        L = 10
        step = 1
        H, W = image.shape[:2]

        points_start = numpy.array([(x, -L, 0) for x in range(-L, L+step,step)]).astype(numpy.float32) + numpy.array(origin).astype(numpy.float32).reshape((1,3))
        points_end   = numpy.array([(x, +L, 0) for x in range(-L, L+step,step)]).astype(numpy.float32) + numpy.array(origin).astype(numpy.float32).reshape((1,3))
        if shift is not None:points_start-=numpy.array(shift).reshape((1,3))
        if shift is not None:points_end  -=numpy.array(shift).reshape((1,3))
        points_start[:, 1] *= -1
        points_end[:, 1] *= -1

        for p1,p2 in zip(points_start,points_end):
            points_2da = tools_render_GL.project_points_MVP_GL(p1.reshape((-1, 3)), W, H, self.mat_projection_BEV,self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV)
            points_2db = tools_render_GL.project_points_MVP_GL(p2.reshape((-1, 3)), W, H, self.mat_projection_BEV,self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV)
            image = tools_draw_numpy.draw_line_fast(image, points_2da[0][1], points_2da[0][0], points_2db[0][1], points_2db[0][0], color,w=w)

        points_start = numpy.array([(-L,y, 0) for y in range(-L, L + step, step)]).astype(numpy.float32) + numpy.array(origin).astype(numpy.float32).reshape((1,3))
        points_end   = numpy.array([(+L,y, 0) for y in range(-L, L + step, step)]).astype(numpy.float32) + numpy.array(origin).astype(numpy.float32).reshape((1,3))
        if shift is not None:points_start-=numpy.array(shift).reshape((1,3))
        if shift is not None:points_end  -=numpy.array(shift).reshape((1,3))
        points_start[:, 1] *= -1
        points_end[:, 1] *= -1

        for p1,p2 in zip(points_start,points_end):
            points_2da = tools_render_GL.project_points_MVP_GL(p1.reshape((-1, 3)), W, H, self.mat_projection_BEV,self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV)
            points_2db = tools_render_GL.project_points_MVP_GL(p2.reshape((-1, 3)), W, H, self.mat_projection_BEV,self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV)
            image = tools_draw_numpy.draw_line_fast(image, points_2da[0][1], points_2da[0][0], points_2db[0][1], points_2db[0][0], color,w=w)

        return image
# ---------------------------------------------------------------------------------------------------------------------
    def draw_circles_on_BEV(self, image, origin, shift=None, color=(64, 64, 64)):


        Pos = [1,5,10,50,100,500,1000]

        if origin is None: return image
        origin = numpy.array(origin).reshape((-1, 3))
        if origin.shape[0] == 0: return image
        if shift is not None:
            origin -= numpy.array(shift).reshape((1, 3))

        H, W = image.shape[:2]
        origin[:,1] *= -1
        origin[:,2] = 0

        origin_2d = tools_render_GL.project_points_MVP_GL(origin.reshape((-1, 3)), W, H, self.mat_projection_BEV, self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV)

        for pos in Pos:
            point_2d = tools_render_GL.project_points_MVP_GL((origin + numpy.array((pos, 0, 0))).reshape((-1, 3)), W, H, self.mat_projection_BEV, self.mat_view_BEV, self.mat_model_BEV, self.mat_trns_BEV)
            radius = numpy.linalg.norm(point_2d - origin_2d[0])
            image = cv2.circle(image, (int(origin_2d[0][0]), int(origin_2d[0][1])), radius=int(radius), color=color, thickness=1)

        return image
# ---------------------------------------------------------------------------------------------------------------------

