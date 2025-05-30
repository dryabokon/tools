#ISSUE: flask reload !!
# ----------------------------------------------------------------------------------------------------------------------
import os
import cv2
import numpy
import pyvista as pv
from scipy.spatial.transform import Rotation
import scipy.spatial.transform as transform
from scipy.linalg import orth
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class render_pyvista(object):

    def __init__(self, filename_obj, W=1080, H=1080+144-56,cam_fov_deg=90,eye=None,target=(0,0,0),up=(0,0,1)):
        print('render_pyvista')
        self.plotter = pv.Plotter(off_screen=True)
        self.actor = self.plotter.add_mesh(pv.read(filename_obj)if filename_obj is not None else pv.Cube(), color="white")

        self.W, self.H = W, H
        self.cam_fov_deg = cam_fov_deg
        self.bg_color = numpy.array([231, 244, 255, 1]) / 255
        self.fg_color = numpy.array([0, 0, 0, 1]) / 255

        if (eye is None):
            self.eye_default, self.target_default, self.up_default = self.get_default_ETU()
        else:
            self.eye_default,self.target_default,self.up_default   = eye,target,up

        self.plotter.camera_position = [self.eye_default, self.target_default, self.up_default]
        self.plotter.camera.view_angle = self.cam_fov_deg
        self.plotter.background_color = self.bg_color
        self.plotter.window_size = (self.W, self.H)
        self.plotter.camera.parallel_projection = False

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def get_default_ETU(self):
        eye = numpy.array([10, 1, 0])
        target = numpy.array([9, 1, 0])
        up = numpy.array([0, 1, 0])
        return eye, target, up
    # ----------------------------------------------------------------------------------------------------------------------
    def from_euler(self,roll, pitch, yaw):return Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
    def from_euler_as_mat(self, roll, pitch, yaw):return Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    def to_euler(self,q):return Rotation.from_quat(q).as_euler('xyz')
    def to_euler_as_mat(self,q):return Rotation.from_quat(q).as_matrix()
    # ----------------------------------------------------------------------------------------------------------------------
    def get_matrices(self):
        vtk_projection = self.plotter.camera.GetProjectionTransformMatrix(1, 0, 1)
        vtk_trns = self.plotter.camera.GetModelTransformMatrix()
        vtk_model = self.actor.GetMatrix()
        vtk_view = self.plotter.camera.GetViewTransformMatrix()

        self.mat_projection = numpy.array([[vtk_projection.GetElement(i, j) for j in range(4)] for i in range(4)])
        self.mat_trns = numpy.array([[vtk_trns.GetElement(i, j) for j in range(4)] for i in range(4)])
        self.mat_model = numpy.array([[vtk_model.GetElement(i, j) for j in range(4)] for i in range(4)])
        self.mat_view = numpy.array([[vtk_view.GetElement(i, j) for j in range(4)] for i in range(4)])
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def print_debug_info(self,image,font_size=24):
        clr_bg = (192, 192, 192) if image[:200, :200].mean() > 192 else (64, 64, 64)
        color_fg = (32, 32, 32) if image[:200, :200].mean() > 192 else (192, 192, 192)
        space = font_size + 2

        eye_drone2, target, up_vector = self.plotter.camera_position
        fov = self.plotter.camera.view_angle

        image = tools_draw_numpy.draw_text(image,'eye=(%.1f, %.1f, %.1f)' % (eye_drone2[0], eye_drone2[1], eye_drone2[2]),(0, space * 9), color_fg=color_fg, clr_bg=clr_bg, font_size=font_size)
        image = tools_draw_numpy.draw_text(image, 'trg=(%.1f, %.1f, %.1f)' % (target[0], target[1], target[2]),(0, space * 10), color_fg=color_fg, clr_bg=clr_bg, font_size=font_size)
        image = tools_draw_numpy.draw_text(image, 'up =(%.1f, %.1f, %.1f)' % (up_vector[0], up_vector[1], up_vector[2]),(0, space * 11), color_fg=color_fg, clr_bg=clr_bg, font_size=font_size)
        image = tools_draw_numpy.draw_text(image, 'fov=%.1f' % fov, (0, space * 15), color_fg=color_fg, clr_bg=clr_bg, font_size=font_size)

        return image
    # ----------------------------------------------------------------------------------------------------------------------
    def update_position(self,eye, target, up_vector):
        self.plotter.camera_position = [eye, target, up_vector]
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_image(self,do_debug=False):
        self.plotter.render()
        image = numpy.array(self.plotter.screenshot(return_img=True))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if do_debug:
            image = self.print_debug_info(image)
        return image
    # ----------------------------------------------------------------------------------------------------------------------
    def stage_data(self, folder_out,do_debug=True):
        name = 'obj'
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)

        filenames = tools_IO.get_filenames(folder_out, name+'*.png')
        ids = [(f.split('.')[0]).split('_')[1] for f in filenames]
        if len(ids) > 0:
            i = 1 + numpy.array(ids, dtype=int).max()
        else:
            i = 0

        image = self.get_image()
        if do_debug:
            image = self.print_debug_info(image)
        cv2.imwrite(folder_out + name+'_%03d.png' % i, image)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def quaternion_from_ETU(self, eye, target, up):
        v_forward = numpy.array(target) - numpy.array(eye)
        v_forward /= numpy.linalg.norm(v_forward)
        v_right = numpy.cross(up, v_forward)
        v_right /= numpy.linalg.norm(v_right)
        v_up = numpy.cross(v_forward, v_right)
        v_up /= numpy.linalg.norm(v_up)
        #rotation_matrix = numpy.column_stack((v_right, v_up, v_forward))
        rotation_matrix = numpy.column_stack((v_right, v_up, v_forward)).T
        rotation_matrix = orth(rotation_matrix)
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

        # check
        # rotation_matrix2 = Rotation.from_quat(quaternion).as_matrix()[[2, 1, 0]]
        # v_right2 = rotation_matrix[:, 0]
        # v_up2 = rotation_matrix[:, 1]
        # v_forward2 = rotation_matrix[:, 2]
        return quaternion
    # ----------------------------------------------------------------------------------------------------------------------
    def ETU_from_quaternion(self, eye, quaternion):
        rotation_matrix2 = Rotation.from_quat(quaternion).as_matrix().T[[2, 1, 0]]
        v_right2 = rotation_matrix2[:, 0]
        v_up2 = rotation_matrix2[:, 1]
        v_forward2 = rotation_matrix2[:, 2]
        v_target2 = numpy.array(eye) + v_forward2

        return eye, v_target2, v_up2
    # ----------------------------------------------------------------------------------------------------------------------
    def get_camera_quaternion_v1(self):
        eye = numpy.array(self.plotter.camera_position[0])
        target = numpy.array(self.plotter.camera_position[1])
        forward = target - eye
        up = numpy.array(self.plotter.camera_position[2])
        orientation_q = self.quaternion_from_ETU(eye, target, up)
        return orientation_q
    # ----------------------------------------------------------------------------------------------------------------------
    def get_camera_quaternion_v2(self):
        vtk_matrix = self.plotter.camera.GetModelViewTransformMatrix()
        rotation_matrix = numpy.array([[vtk_matrix.GetElement(i, j) for j in range(3)] for i in range(3)])
        orientation_q = transform.Rotation.from_matrix(rotation_matrix).as_quat()
        return orientation_q
    # ----------------------------------------------------------------------------------------------------------------------
    def set_camera_quaternion(self, eye, quaternion):
        eye, target, up = self.ETU_from_quaternion(eye, quaternion)
        self.update_position(eye, target, up)
        return
   # ----------------------------------------------------------------------------------------------------------------------
    def move_up_down(self, movement_scalar):
        position = numpy.array(self.plotter.camera_position[0])
        target = numpy.array(self.plotter.camera_position[1])
        up = numpy.array(self.plotter.camera_position[2])
        position[1] += movement_scalar
        target[1] += movement_scalar
        self.update_position(position, target, up)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def move_back_forwrd(self, step):
        position = numpy.array(self.plotter.camera_position[0])
        target = numpy.array(self.plotter.camera_position[1])
        up = numpy.array(self.plotter.camera_position[2])
        forward = target - position
        forward /= numpy.linalg.norm(forward)
        target += step * forward
        position += step * forward
        self.update_position(position, target, up)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def move_left_right(self, step):
        position = numpy.array(self.plotter.camera_position[0])
        target = numpy.array(self.plotter.camera_position[1])
        up = numpy.array(self.plotter.camera_position[2])
        forward = target - position
        forward /= numpy.linalg.norm(forward)
        right = numpy.cross(forward, up)
        right /= numpy.linalg.norm(right)
        target+= step * right
        position += step * right
        self.update_position(position, target, up)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    # def rotate_model_yaw(self, delta_yaw):
    #     self.mesh.rotate_y(delta_yaw, inplace=True)
    #     return
# ----------------------------------------------------------------------------------------------------------------------
#     def rotate_model_pitch(self, delta_pitch):
#         self.mesh.rotate_z(delta_pitch, inplace=True)
#         return
# ----------------------------------------------------------------------------------------------------------------------
    def rotate_view_yaw(self,delta_yaw):
        eye = numpy.array(self.plotter.camera_position[0])
        target = numpy.array(self.plotter.camera_position[1])
        v_up = numpy.array(self.plotter.camera_position[2])
        v_forward = target - eye
        v_right = numpy.cross(v_up, v_forward)

        mat_R = Rotation.from_rotvec([0,delta_yaw,0]).as_matrix()
        v_forward2 = mat_R @ v_forward
        v_up2 = mat_R @ v_up
        target2 = eye + v_forward2

        self.update_position(eye, target2, v_up2)
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def rotate_view_pitch(self,delta_pitch):
        eye = numpy.array(self.plotter.camera_position[0])
        target = numpy.array(self.plotter.camera_position[1])
        v_up = numpy.array(self.plotter.camera_position[2])
        v_forward = target - eye
        v_right = numpy.cross(v_up, v_forward)

        cos_theta = numpy.cos(delta_pitch)
        sin_theta = numpy.sin(delta_pitch)
        R = numpy.array([
            [cos_theta + (1 - cos_theta) * v_right[0] ** 2,
             (1 - cos_theta) * v_right[0] * v_right[1] - sin_theta * v_right[2],
             (1 - cos_theta) * v_right[0] * v_right[2] + sin_theta * v_right[1]],

            [(1 - cos_theta) * v_right[1] * v_right[0] + sin_theta * v_right[2],
             cos_theta + (1 - cos_theta) * v_right[1] ** 2,
             (1 - cos_theta) * v_right[1] * v_right[2] - sin_theta * v_right[0]],

            [(1 - cos_theta) * v_right[2] * v_right[0] - sin_theta * v_right[1],
             (1 - cos_theta) * v_right[2] * v_right[1] + sin_theta * v_right[0],
             cos_theta + (1 - cos_theta) * v_right[2] ** 2]
        ])

        v_forward2 = R @ v_forward
        v_up2 = R @ v_up
        target2 = eye + v_forward2
        self.update_position(eye, target2, v_up2)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def key_control_mode_1(self, key):
        delta = 10 * numpy.pi * 1 / 180
        if key=='PageUp': self.move_up_down(1)
        if key=='PageDown': self.move_up_down(-1)
        if key=='Up': self.move_back_forwrd(1)
        if key=='Down': self.move_back_forwrd(-1)
        if key=='Left': self.move_left_right(-1)
        if key=='Right': self.move_left_right(1)
        if key == 'Clear': self.update_position(self.eye_default, self.target_default, self.up_default)

        if key=='a': self.rotate_view_yaw(+delta)
        if key=='d': self.rotate_view_yaw(-delta)
        if key=='w': self.rotate_view_pitch(-delta)
        if key=='s': self.rotate_view_pitch(+delta)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def key_control_mode_2(self, key):
        if key=='PageUp': self.move_up_down(1)
        if key=='PageDown': self.move_up_down(-1)
        if key=='Up': self.move_back_forwrd(1)
        if key=='Down': self.move_back_forwrd(-1)
        if key=='Left': self.move_left_right(-1)
        if key=='Right': self.move_left_right(1)
        if key=='Clear': self.update_position(self.eye_default,self.target_default,self.up_default)

        if key=='a': self.move_left_right(-1)
        if key=='d': self.move_left_right(+1)
        if key=='w': self.move_back_forwrd(+1)
        if key=='s': self.move_back_forwrd(-1)
        if key=='q': self.move_up_down(-1)
        if key=='e': self.move_up_down(1)
        return