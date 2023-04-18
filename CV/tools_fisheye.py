#https://kaustubh-sadekar.github.io/OmniCV-Lib/index.html#output-gallery
import cv2
import numpy

class Converter:
    def __init__(self,param_file_path=None):
        self.Hd = None
        self.Wd = None
        self.map_x = None
        self.map_y = None
        self.singleLens = False
        self.filePath = param_file_path

    def rmat(self,alpha,beta,gamma):
        rx = numpy.array(
            [
                [1, 0, 0],
                [0, numpy.cos(alpha * numpy.pi / 180), -numpy.sin(alpha * numpy.pi / 180)],
                [0, numpy.sin(alpha * numpy.pi / 180), numpy.cos(alpha * numpy.pi / 180)],
            ]
        )
        ry = numpy.array(
            [
                [numpy.cos(beta * numpy.pi / 180), 0, numpy.sin(beta * numpy.pi / 180)],
                [0, 1, 0],
                [-numpy.sin(beta * numpy.pi / 180), 0, numpy.cos(beta * numpy.pi / 180)],
            ]
        )
        rz = numpy.array(
            [
                [numpy.cos(gamma * numpy.pi / 180), -numpy.sin(gamma * numpy.pi / 180), 0],
                [numpy.sin(gamma * numpy.pi / 180), numpy.cos(gamma * numpy.pi / 180), 0],
                [0, 0, 1],
            ]
        )

        return numpy.matmul(rz, numpy.matmul(ry, rx))
# ----------------------------------------------------------------------------------------------------------------------
    def apply_fisheye_effect(self,img, distortion=2):

        img = cv2.resize(img, (img.shape[0], img.shape[0]))
        H, W = img.shape[:2]

        x, y = numpy.arange(0, 1 * H), numpy.arange(0, 1 * W)
        xnd = ((x - 0.5 * W) / W)
        ynd = ((y - 0.5 * H) / H)

        gg1, gg2 = numpy.meshgrid(xnd, ynd)
        rd = gg1 ** 2 + gg2 ** 2

        ddd = (1 - (distortion * rd))
        ddd[ddd == 0] = 1e-4

        xdu = gg1 / ddd
        ydu = gg2 / ddd

        map_x = (((xdu + 1.0) * W) / 2).astype('float32')
        map_y = (((ydu + 1.0) * H) / 2).astype('float32')

        dstimg = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        dstimg[map_x < 0] = 0
        dstimg[map_y < 0] = 0
        dstimg[map_x > W] = 0
        dstimg[map_y > H] = 0

        return dstimg
# ----------------------------------------------------------------------------------------------------------------------
    def fisheye2panoram(self,srcFrame,outShape,offset_x=0,x_min=0,x_max=1.0,y_min=0.0,y_max=0.0):
        Cx = srcFrame.shape[0] / 2
        Cy = srcFrame.shape[1] / 2
        R = Cx

        Hd,Wd = outShape

        Ry = numpy.array([R*(y_min + y*(y_max-y_min)/Hd) for y in range(Hd)])
        theta_x = numpy.array([(((x_min + x * (x_max - x_min)) - offset_x) / (Wd)) * 2.0 * numpy.pi for x in range(Wd)])

        X, Y = numpy.meshgrid(theta_x, Ry)
        map_x = Cx + numpy.array([numpy.sin(X.flatten()) * Y.flatten()]).reshape((Ry.shape[0], theta_x.shape[0])).astype('float32')
        map_y = Cy + numpy.array([numpy.cos(X.flatten()) * Y.flatten()]).reshape((Ry.shape[0], theta_x.shape[0])).astype('float32')

        result = cv2.remap(srcFrame, map_x, map_y, cv2.INTER_LINEAR)[::-1, ::-1]
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def fisheye2equirect(self,srcFrame,outShape,aperture=0,delx=0,dely=0,radius=0,edit_mode=False):

        inShape = srcFrame.shape[:2]
        self.Hs = inShape[0]
        self.Ws = inShape[1]
        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.map_x = numpy.zeros((self.Hd, self.Wd), numpy.float32)
        self.map_y = numpy.zeros((self.Hd, self.Wd), numpy.float32)

        self.Cx = (
                self.Ws // 2 - delx
        )  # This value needs to be tuned using the GUI for every new camera
        self.Cy = (
                self.Hs // 2 - dely
        )  # This value needs to be tuned using the GUI for every new camera

        if not radius:
            self.radius = min(inShape)
        else:
            self.radius = radius

        if not aperture:
            self.aperture = 385  # This value is determined using the GUI
        else:
            self.aperture = aperture

        i, j = numpy.meshgrid(numpy.arange(0, int(self.Hd)), numpy.arange(0, int(self.Wd)))
        x = self.radius * numpy.cos((i * 1.0 / self.Hd - 0.5) * numpy.pi) * numpy.cos(
            (j * 1.0 / self.Hd - 0.5) * numpy.pi)
        y = self.radius * numpy.cos((i * 1.0 / self.Hd - 0.5) * numpy.pi) * numpy.sin(
            (j * 1.0 / self.Hd - 0.5) * numpy.pi)
        z = self.radius * numpy.sin((i * 1.0 / self.Hd - 0.5) * numpy.pi)
        r = 2 * numpy.arctan2(numpy.sqrt(x ** 2 + z ** 2), y) / numpy.pi * 180 / self.aperture * self.radius
        theta = numpy.arctan2(z, x)

        self.map_x = numpy.multiply(r, numpy.cos(theta)).T.astype(numpy.float32) + self.Cx
        self.map_y = numpy.multiply(r, numpy.sin(theta)).T.astype(numpy.float32) + self.Cy
        return cv2.remap(srcFrame, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    def equirect2cubemap(self,srcFrame,side=256,modif=False,dice=False):

        self.dice = dice
        self.side = side

        inShape = srcFrame.shape[:2]
        mesh = numpy.stack(
            numpy.meshgrid(
                numpy.linspace(-0.5, 0.5, num=side, dtype=numpy.float32),
                -numpy.linspace(-0.5, 0.5, num=side, dtype=numpy.float32),
            ),
            -1,
        )

        # Creating a matrix that contains x,y,z values of all 6 faces
        facesXYZ = numpy.zeros((side, side * 6, 3), numpy.float32)

        if modif:
            # Front face (z = 0.5)
            facesXYZ[:, 0 * side: 1 * side, [0, 2]] = mesh
            facesXYZ[:, 0 * side: 1 * side, 1] = -0.5

            # Right face (x = 0.5)
            facesXYZ[:, 1 * side: 2 * side, [1, 2]] = numpy.flip(mesh, axis=1)
            facesXYZ[:, 1 * side: 2 * side, 0] = 0.5

            # Back face (z = -0.5)
            facesXYZ[:, 2 * side: 3 * side, [0, 2]] = mesh
            facesXYZ[:, 2 * side: 3 * side, 1] = 0.5

            # Left face (x = -0.5)
            facesXYZ[:, 3 * side: 4 * side, [1, 2]] = numpy.flip(mesh, axis=1)
            facesXYZ[:, 3 * side: 4 * side, 0] = -0.5

            # Up face (y = 0.5)
            facesXYZ[:, 4 * side: 5 * side, [0, 1]] = mesh[::-1]
            facesXYZ[:, 4 * side: 5 * side, 2] = 0.5

            # Down face (y = -0.5)
            facesXYZ[:, 5 * side: 6 * side, [0, 1]] = mesh
            facesXYZ[:, 5 * side: 6 * side, 2] = -0.5

        else:
            # Front face (z = 0.5)
            facesXYZ[:, 0 * side: 1 * side, [0, 1]] = mesh
            facesXYZ[:, 0 * side: 1 * side, 2] = 0.5

            # Right face (x = 0.5)
            facesXYZ[:, 1 * side: 2 * side, [2, 1]] = mesh
            facesXYZ[:, 1 * side: 2 * side, 0] = 0.5

            # Back face (z = -0.5)
            facesXYZ[:, 2 * side: 3 * side, [0, 1]] = mesh
            facesXYZ[:, 2 * side: 3 * side, 2] = -0.5

            # Left face (x = -0.5)
            facesXYZ[:, 3 * side: 4 * side, [2, 1]] = mesh
            facesXYZ[:, 3 * side: 4 * side, 0] = -0.5

            # Up face (y = 0.5)
            facesXYZ[:, 4 * side: 5 * side, [0, 2]] = mesh
            facesXYZ[:, 4 * side: 5 * side, 1] = 0.5

            # Down face (y = -0.5)
            facesXYZ[:, 5 * side: 6 * side, [0, 2]] = mesh
            facesXYZ[:, 5 * side: 6 * side, 1] = -0.5

        # Calculating the spherical coordinates phi and theta for given XYZ
        # coordinate of a cube face
        x, y, z = numpy.split(facesXYZ, 3, axis=-1)
        # phi = tan^-1(x/z)
        phi = numpy.arctan2(x, z)
        # theta = tan^-1(y/||(x,y)||)
        theta = numpy.arctan2(y, numpy.sqrt(x ** 2 + z ** 2))

        h, w = inShape
        # Calculating corresponding coordinate points in
        # the equirectangular image
        eqrec_x = (phi / (2 * numpy.pi) + 0.5) * w
        eqrec_y = (-theta / numpy.pi + 0.5) * h
        # Note: we have considered equirectangular image to
        # be mapped to a normalised form and then to the scale of (pi,2pi)

        self.map_x = eqrec_x
        self.map_y = eqrec_y

        dstFrame = cv2.remap(srcFrame,
                             self.map_x,
                             self.map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)

        if self.dice:
            line1 = numpy.hstack(
                (
                    dstFrame[:, 4 * side: 5 * side, :] * 0,
                    cv2.flip(dstFrame[:, 4 * side: 5 * side, :], 0),
                    dstFrame[:, 4 * side: 5 * side, :] * 0,
                    dstFrame[:, 4 * side: 5 * side, :] * 0,
                )
            )
            line2 = numpy.hstack(
                (
                    dstFrame[:, 3 * side: 4 * side, :],
                    dstFrame[:, 0 * side: 1 * side, :],
                    cv2.flip(dstFrame[:, 1 * side: 2 * side, :], 1),
                    cv2.flip(dstFrame[:, 2 * side: 3 * side, :], 1),
                )
            )
            line3 = numpy.hstack(
                (
                    dstFrame[:, 5 * side: 6 * side, :] * 0,
                    dstFrame[:, 5 * side: 6 * side, :],
                    dstFrame[:, 5 * side: 6 * side, :] * 0,
                    dstFrame[:, 5 * side: 6 * side, :] * 0,
                )
            )
            dstFrame = numpy.vstack((line1, line2, line3))

        return dstFrame
    def cubemap2equirect(self,srcFrame,outShape):

        h, w = srcFrame.shape[:2]

        if h / w == 3 / 4:
            l1, l2, l3 = numpy.split(srcFrame, 3, axis=0)
            _, pY, _, _ = numpy.split(l1, 4, axis=1)
            nX, pZ, pX, nZ = numpy.split(l2, 4, axis=1)
            _, nY, _, _ = numpy.split(l3, 4, axis=1)

            srcFrame = numpy.hstack(
                (pZ, cv2.flip(pX, 1), cv2.flip(nZ, 1), nX, cv2.flip(pY, 0), nY)
            )

        inShape = srcFrame.shape[:2]
        self.Hd = outShape[0]
        self.Wd = outShape[1]
        h = self.Hd
        w = self.Wd
        face_w = inShape[0]

        phi = numpy.linspace(-numpy.pi, numpy.pi, num=self.Wd, dtype=numpy.float32)
        theta = numpy.linspace(numpy.pi, -numpy.pi, num=self.Hd, dtype=numpy.float32) / 2

        phi, theta = numpy.meshgrid(phi, theta)

        tp = numpy.zeros((h, w), dtype=numpy.int32)
        tp[:, : w // 8] = 2
        tp[:, w // 8: 3 * w // 8] = 3
        tp[:, 3 * w // 8: 5 * w // 8] = 0
        tp[:, 5 * w // 8: 7 * w // 8] = 1
        tp[:, 7 * w // 8:] = 2

        # Prepare ceil mask
        mask = numpy.zeros((h, w // 4), bool)
        idx = numpy.linspace(-numpy.pi, numpy.pi, w // 4) / 4
        idx = h // 2 - numpy.round(numpy.arctan(numpy.cos(idx)) * h / numpy.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1

        mask = numpy.roll(mask, w // 8, 1)

        mask = numpy.concatenate([mask] * 4, 1)

        tp[mask] = 4
        tp[numpy.flip(mask, 0)] = 5

        tp = tp.astype(numpy.int32)

        coor_x = numpy.zeros((h, w))
        coor_y = numpy.zeros((h, w))

        for i in range(4):
            mask = tp == i
            coor_x[mask] = 0.5 * numpy.tan(phi[mask] - numpy.pi * i / 2)
            coor_y[mask] = (
                    -0.5 * numpy.tan(theta[mask]) / numpy.cos(phi[mask] - numpy.pi * i / 2)
            )

        mask = tp == 4
        c = 0.5 * numpy.tan(numpy.pi / 2 - theta[mask])
        coor_x[mask] = c * numpy.sin(phi[mask])
        coor_y[mask] = c * numpy.cos(phi[mask])

        mask = tp == 5
        c = 0.5 * numpy.tan(numpy.pi / 2 - numpy.abs(theta[mask]))
        coor_x[mask] = c * numpy.sin(phi[mask])
        coor_y[mask] = -c * numpy.cos(phi[mask])

        # Final renormalize
        coor_x = (numpy.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
        coor_y = (numpy.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

        self.map_x = coor_x.astype(numpy.float32)
        self.map_y = coor_y.astype(numpy.float32)

        dstFrame = 0
        cube_faces = numpy.stack(numpy.split(srcFrame, 6, 1), 0)
        cube_faces[1] = numpy.flip(cube_faces[1], 1)
        cube_faces[2] = numpy.flip(cube_faces[2], 1)
        cube_faces[4] = numpy.flip(cube_faces[4], 0)
        self.tp = tp
        for i in range(6):
            mask = self.tp == i
            dstFrame1 = cv2.remap(
                cube_faces[i],
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            # We use this border mode to avoid small black lines

            dstFrame += cv2.bitwise_and(
                dstFrame1, dstFrame1, mask=mask.astype(numpy.uint8)
            )

        return dstFrame
    def eqruirect2persp(self,img,FOV,THETA,PHI,Hd,Wd):

        # THETA is left/right angle, PHI is up/down angle, both in degree
        equ_h, equ_w = img.shape[:2]

        equ_cx = (equ_w) / 2.0
        equ_cy = (equ_h) / 2.0

        wFOV = FOV
        hFOV = float(Hd) / Wd * wFOV

        c_x = (Wd) / 2.0
        c_y = (Hd) / 2.0

        w_len = 2 * numpy.tan(numpy.radians(wFOV / 2.0))
        w_interval = w_len / (Wd)

        h_len = 2 * numpy.tan(numpy.radians(hFOV / 2.0))
        h_interval = h_len / (Hd)

        x_map = numpy.zeros([Hd, Wd], numpy.float32) + 1
        y_map = numpy.tile((numpy.arange(0, Wd) - c_x) * w_interval, [Hd, 1])
        z_map = -numpy.tile((numpy.arange(0, Hd) - c_y) * h_interval, [Wd, 1]).T
        D = numpy.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)

        xyz = numpy.zeros([Hd, Wd, 3], numpy.float32)
        xyz[:, :, 0] = (x_map / D)[:, :]
        xyz[:, :, 1] = (y_map / D)[:, :]
        xyz[:, :, 2] = (z_map / D)[:, :]

        y_axis = numpy.array([0.0, 1.0, 0.0], numpy.float32)
        z_axis = numpy.array([0.0, 0.0, 1.0], numpy.float32)
        [R1, _] = cv2.Rodrigues(z_axis * numpy.radians(THETA))
        [R2, _] = cv2.Rodrigues(numpy.dot(R1, y_axis) * numpy.radians(-PHI))

        xyz = xyz.reshape([Hd * Wd, 3]).T
        xyz = numpy.dot(R1, xyz)
        xyz = numpy.dot(R2, xyz).T
        lat = numpy.arcsin(xyz[:, 2] / 1)
        lon = numpy.zeros([Hd * Wd], numpy.float32)
        theta = numpy.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + numpy.pi
        lon[idx4] = theta[idx4] - numpy.pi

        lon = lon.reshape([Hd, Wd]) / numpy.pi * 180
        lat = -lat.reshape([Hd, Wd]) / numpy.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        self.map_x = lon.astype(numpy.float32)
        self.map_y = lat.astype(numpy.float32)

        persp = cv2.remap(img,
                          lon.astype(numpy.float32),
                          lat.astype(numpy.float32),
                          cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)

        return persp
    def cubemap2persp(self,img,FOV,THETA,PHI,Hd,Wd):

        # THETA is left/right angle, PHI is up/down angle, both in degree

        img = self.cubemap2equirect(img, [2 * Hd, 4 * Hd])

        equ_h, equ_w = img.shape[:2]

        equ_cx = (equ_w) / 2.0
        equ_cy = (equ_h) / 2.0

        wFOV = FOV
        hFOV = float(Hd) / Wd * wFOV

        c_x = (Wd) / 2.0
        c_y = (Hd) / 2.0

        w_len = 2 * 1 * numpy.sin(
            numpy.radians(wFOV / 2.0)) / numpy.cos(numpy.radians(wFOV / 2.0))
        w_interval = w_len / (Wd)

        h_len = 2 * 1 * numpy.sin(
            numpy.radians(hFOV / 2.0)) / numpy.cos(numpy.radians(hFOV / 2.0))
        h_interval = h_len / (Hd)

        x_map = numpy.zeros([Hd, Wd], numpy.float32) + 1
        y_map = numpy.tile((numpy.arange(0, Wd) - c_x) * w_interval, [Hd, 1])
        z_map = -numpy.tile((numpy.arange(0, Hd) - c_y) * h_interval, [Wd, 1]).T
        D = numpy.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = numpy.zeros([Hd, Wd, 3], numpy.float)
        xyz[:, :, 0] = (1 / D * x_map)[:, :]
        xyz[:, :, 1] = (1 / D * y_map)[:, :]
        xyz[:, :, 2] = (1 / D * z_map)[:, :]

        y_axis = numpy.array([0.0, 1.0, 0.0], numpy.float32)
        z_axis = numpy.array([0.0, 0.0, 1.0], numpy.float32)
        [R1, _] = cv2.Rodrigues(z_axis * numpy.radians(THETA))
        [R2, _] = cv2.Rodrigues(numpy.dot(R1, y_axis) * numpy.radians(-PHI))

        xyz = xyz.reshape([Hd * Wd, 3]).T
        xyz = numpy.dot(R1, xyz)
        xyz = numpy.dot(R2, xyz).T
        lat = numpy.arcsin(xyz[:, 2] / 1)
        lon = numpy.zeros([Hd * Wd], numpy.float)
        theta = numpy.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(numpy.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(numpy.bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + numpy.pi
        lon[idx4] = theta[idx4] - numpy.pi

        lon = lon.reshape([Hd, Wd]) / numpy.pi * 180
        lat = -lat.reshape([Hd, Wd]) / numpy.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        self.map_x = lon.astype(numpy.float32)
        self.map_y = lat.astype(numpy.float32)

        persp = cv2.remap(
            img,
            lon.astype(numpy.float32),
            lat.astype(numpy.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        return persp


    def equirect2Fisheye(self,img,outShape,f=50,xi=1.2,angles=[0, 0, 0]):

        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.f = f
        self.xi = xi

        Hs, Ws = img.shape[:2]

        self.Cx = self.Wd / 2.0
        self.Cy = self.Hd / 2.0

        x = numpy.linspace(0, self.Wd - 1, num=self.Wd, dtype=numpy.float32)
        y = numpy.linspace(0, self.Hd - 1, num=self.Hd, dtype=numpy.float32)

        x, y = numpy.meshgrid(range(self.Wd), range(self.Hd))
        xref = 1
        yref = 1

        self.fmin = (
                numpy.sqrt(
                    -(1 - self.xi ** 2) * ((xref - self.Cx) ** 2 +
                                           (yref - self.Cy) ** 2)
                )
                * 1.0001
        )

        x_hat = (x - self.Cx) / self.f
        y_hat = (y - self.Cy) / self.f

        x2_y2_hat = x_hat ** 2 + y_hat ** 2

        omega = numpy.real(
            self.xi + numpy.lib.scimath.sqrt(1 + (1 - self.xi ** 2) * x2_y2_hat)
        ) / (x2_y2_hat + 1)
        # print(np.max(x2_y2_hat))

        Ps_x = omega * x_hat
        Ps_y = omega * y_hat
        Ps_z = omega - self.xi

        self.alpha = angles[0]
        self.beta = angles[1]
        self.gamma = angles[2]

        R = numpy.matmul(
            self.rmat(self.alpha, self.beta, self.gamma),
            numpy.matmul(self.rmat(0, -90, 45), self.rmat(0, 90, 90)),
        )

        Ps = numpy.stack((Ps_x, Ps_y, Ps_z), -1)
        Ps = numpy.matmul(Ps, R.T)

        Ps_x, Ps_y, Ps_z = numpy.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = numpy.arctan2(Ps_y, Ps_x)
        phi = numpy.arctan2(Ps_z, numpy.sqrt(Ps_x ** 2 + Ps_y ** 2))

        a = 2 * numpy.pi / (Ws - 1)
        b = numpy.pi - a * (Ws - 1)
        self.map_x = (1.0 / a) * (theta - b)

        a = -numpy.pi / (Hs - 1)
        b = numpy.pi / 2
        self.map_y = (1.0 / a) * (phi - b)

        output = cv2.remap(
            img,
            self.map_x.astype(numpy.float32),
            self.map_y.astype(numpy.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        if self.f < self.fmin:
            r = numpy.sqrt(-(self.f ** 2) / (1 - self.xi ** 2))
            mask = numpy.zeros_like(output[:, :, 0])
            mask = cv2.circle(
                mask, (int(self.Cx), int(self.Cy)), int(r), (255, 255, 255), -1
            )
            output = cv2.bitwise_and(output, output, mask=mask)

        return output
    def equirect2Fisheye_UCM(self,img,outShape,f=50,xi=1.2,angles=[0, 0, 0]):

        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.f = f
        self.xi = xi

        Hs, Ws = img.shape[:2]

        self.Cx = self.Wd / 2.0
        self.Cy = self.Hd / 2.0

        x = numpy.linspace(0, self.Wd - 1, num=self.Wd, dtype=numpy.float32)
        y = numpy.linspace(0, self.Hd - 1, num=self.Hd, dtype=numpy.float32)

        x, y = numpy.meshgrid(range(self.Wd), range(self.Hd))
        xref = 1
        yref = 1

        self.fmin = (
                numpy.lib.scimath.sqrt(
                    -(1 - self.xi ** 2) *
                    ((xref - self.Cx) ** 2 + (yref - self.Cy) ** 2)
                )
                * 1.0001
        )

        if self.xi ** 2 >= 1:
            self.fmin = numpy.real(self.fmin)
        else:
            self.fmin = numpy.imag(self.fmin)

        x_hat = (x - self.Cx) / self.f
        y_hat = (y - self.Cy) / self.f

        x2_y2_hat = x_hat ** 2 + y_hat ** 2

        omega = numpy.real(
            self.xi + numpy.lib.scimath.sqrt(1 + (1 - self.xi ** 2) * x2_y2_hat)
        ) / (x2_y2_hat + 1)

        Ps_x = omega * x_hat
        Ps_y = omega * y_hat
        Ps_z = omega - self.xi

        self.alpha = angles[0]
        self.beta = angles[1]
        self.gamma = angles[2]

        R = numpy.matmul(
            self.rmat(self.alpha, self.beta, self.gamma),
            numpy.matmul(self.rmat(0, -90, 45), self.rmat(0, 90, 90)),
        )

        Ps = numpy.stack((Ps_x, Ps_y, Ps_z), -1)
        Ps = numpy.matmul(Ps, R.T)

        Ps_x, Ps_y, Ps_z = numpy.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = numpy.arctan2(Ps_y, Ps_x)
        phi = numpy.arctan2(Ps_z, numpy.sqrt(Ps_x ** 2 + Ps_y ** 2))

        a = 2 * numpy.pi / (Ws - 1)
        b = numpy.pi - a * (Ws - 1)
        self.map_x = (1.0 / a) * (theta - b)

        a = -numpy.pi / (Hs - 1)
        b = numpy.pi / 2
        self.map_y = (1.0 / a) * (phi - b)

        output = cv2.remap(
            img,
            self.map_x.astype(numpy.float32),
            self.map_y.astype(numpy.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        if self.f < self.fmin:
            r = numpy.sqrt(numpy.abs(-(self.f ** 2) / (1 - self.xi ** 2)))
            mask = numpy.zeros_like(output[:, :, 0])
            mask = cv2.circle(
                mask, (int(self.Cx), int(self.Cy)), int(r), (255, 255, 255), -1
            )
            output = cv2.bitwise_and(output, output, mask=mask)

        return output
    def equirect2Fisheye_EUCM(self,img,outShape,f=50,a_=0.5,b_=0.5,angles=[0, 0, 0]):

        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.f = f
        self.a_ = a_
        self.b_ = b_

        Hs, Ws = img.shape[:2]

        self.Cx = self.Wd / 2.0
        self.Cy = self.Hd / 2.0

        x = numpy.linspace(0, self.Wd - 1, num=self.Wd, dtype=numpy.float32)
        y = numpy.linspace(0, self.Hd - 1, num=self.Hd, dtype=numpy.float32)

        x, y = numpy.meshgrid(range(self.Wd), range(self.Hd))
        xref = 1
        yref = 1

        self.fmin = (
                numpy.lib.scimath.sqrt(
                    self.b_
                    * (2 * self.a_ - 1)
                    * ((xref - self.Cx) ** 2 + (yref - self.Cy) ** 2)
                )
                * 1.0001
        )
        # print(self.fmin)
        if numpy.real(self.fmin) <= 0:
            self.fmin = numpy.imag(self.fmin)

        # print(self.f)
        # print(self.fmin)

        mx = (x - self.Cx) / self.f
        my = (y - self.Cy) / self.f

        r_2 = mx ** 2 + my ** 2

        mz = numpy.real(
            (1 - self.b_ * self.a_ * self.a_ * r_2)
            / (
                    self.a_ * numpy.lib.scimath.sqrt(1 - (2 * self.a_ - 1) *
                                                     self.b_ * r_2)
                    + (1 - self.a_)
            )
        )

        coef = 1 / numpy.sqrt(mx ** 2 + my ** 2 + mz ** 2)

        Ps_x = mx * coef
        Ps_y = my * coef
        Ps_z = mz * coef

        self.alpha = angles[0]
        self.beta = angles[1]
        self.gamma = angles[2]

        R = numpy.matmul(
            self.rmat(self.alpha, self.beta, self.gamma),
            numpy.matmul(self.rmat(0, -90, 45), self.rmat(0, 90, 90)),
        )

        Ps = numpy.stack((Ps_x, Ps_y, Ps_z), -1)
        Ps = numpy.matmul(Ps, R.T)

        Ps_x, Ps_y, Ps_z = numpy.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = numpy.arctan2(Ps_y, Ps_x)
        phi = numpy.arctan2(Ps_z, numpy.sqrt(Ps_x ** 2 + Ps_y ** 2))

        a = 2 * numpy.pi / (Ws - 1)
        b = numpy.pi - a * (Ws - 1)
        self.map_x = (1.0 / a) * (theta - b)

        a = -numpy.pi / (Hs - 1)
        b = numpy.pi / 2
        self.map_y = (1.0 / a) * (phi - b)

        output = cv2.remap(
            img,
            self.map_x.astype(numpy.float32),
            self.map_y.astype(numpy.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        if self.f < self.fmin:
            r = numpy.sqrt(numpy.abs((self.f ** 2) / (self.b_ * (2 * self.a_ - 1))))
            mask = numpy.zeros_like(output[:, :, 0])
            mask = cv2.circle(
                mask, (int(self.Cx), int(self.Cy)), int(r), (255, 255, 255), -1
            )
            output = cv2.bitwise_and(output, output, mask=mask)

        return output
    def equirect2Fisheye_FOV(self,img,outShape,f=50,w_=0.5,angles=[0, 0, 0]):

        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.f = f
        self.w_ = w_

        Hs, Ws = img.shape[:2]

        self.Cx = self.Wd / 2.0
        self.Cy = self.Hd / 2.0

        x = numpy.linspace(0, self.Wd - 1, num=self.Wd, dtype=numpy.float32)
        y = numpy.linspace(0, self.Hd - 1, num=self.Hd, dtype=numpy.float32)

        x, y = numpy.meshgrid(range(self.Wd), range(self.Hd))

        mx = (x - self.Cx) / self.f
        my = (y - self.Cy) / self.f

        rd = numpy.sqrt(mx ** 2 + my ** 2)

        Ps_x = mx * numpy.sin(rd * self.w_) / (2 * rd * numpy.tan(self.w_ / 2))
        Ps_y = my * numpy.sin(rd * self.w_) / (2 * rd * numpy.tan(self.w_ / 2))
        Ps_z = numpy.cos(rd * self.w_)

        self.alpha = angles[0]
        self.beta = angles[1]
        self.gamma = angles[2]

        R = numpy.matmul(
            self.rmat(self.alpha, self.beta, self.gamma),
            numpy.matmul(self.rmat(0, -90, 45), self.rmat(0, 90, 90)),
        )

        Ps = numpy.stack((Ps_x, Ps_y, Ps_z), -1)
        Ps = numpy.matmul(Ps, R.T)

        Ps_x, Ps_y, Ps_z = numpy.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = numpy.arctan2(Ps_y, Ps_x)
        phi = numpy.arctan2(Ps_z, numpy.sqrt(Ps_x ** 2 + Ps_y ** 2))

        a = 2 * numpy.pi / (Ws - 1)
        b = numpy.pi - a * (Ws - 1)
        self.map_x = (1.0 / a) * (theta - b)

        a = -numpy.pi / (Hs - 1)
        b = numpy.pi / 2
        self.map_y = (1.0 / a) * (phi - b)

        output = cv2.remap(
            img,
            self.map_x.astype(numpy.float32),
            self.map_y.astype(numpy.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        return output
    def equirect2Fisheye_DS(self,img,outShape,f=50,a_=0.5,xi_=0.5,angles=[0, 0, 0]):

        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.f = f
        self.a_ = a_
        self.xi_ = xi_

        Hs, Ws = img.shape[:2]

        self.Cx = self.Wd / 2.0
        self.Cy = self.Hd / 2.0

        x = numpy.linspace(0, self.Wd - 1, num=self.Wd, dtype=numpy.float32)
        y = numpy.linspace(0, self.Hd - 1, num=self.Hd, dtype=numpy.float32)

        x, y = numpy.meshgrid(range(self.Wd), range(self.Hd))
        xref = 1
        yref = 1

        self.fmin = numpy.sqrt(numpy.abs((2 * self.a_ - 1) *
                                         ((xref - self.Cx) ** 2 + (yref - self.Cy) ** 2))
                               )

        mx = (x - self.Cx) / self.f
        my = (y - self.Cy) / self.f

        r_2 = mx ** 2 + my ** 2

        mz = numpy.real(
            (1 - self.a_ * self.a_ * r_2)
            / (self.a_ * numpy.lib.scimath.sqrt(1 - (2 * self.a_ - 1) * r_2) +
               1 - self.a_)
        )

        omega = numpy.real(
            (mz * self.xi_ + numpy.lib.scimath.sqrt(mz ** 2 +
                                                    (1 - self.xi_ ** 2) * r_2))
            / (mz ** 2 + r_2)
        )

        Ps_x = omega * mx
        Ps_y = omega * my
        Ps_z = omega * mz - self.xi_

        self.alpha = angles[0]
        self.beta = angles[1]
        self.gamma = angles[2]

        R = numpy.matmul(
            self.rmat(self.alpha, self.beta, self.gamma),
            numpy.matmul(self.rmat(0, -90, 45), self.rmat(0, 90, 90)),
        )

        Ps = numpy.stack((Ps_x, Ps_y, Ps_z), -1)
        Ps = numpy.matmul(Ps, R.T)

        Ps_x, Ps_y, Ps_z = numpy.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = numpy.arctan2(Ps_y, Ps_x)
        phi = numpy.arctan2(Ps_z, numpy.sqrt(Ps_x ** 2 + Ps_y ** 2))

        a = 2 * numpy.pi / (Ws - 1)
        b = numpy.pi - a * (Ws - 1)
        self.map_x = (1.0 / a) * (theta - b)

        a = -numpy.pi / (Hs - 1)
        b = numpy.pi / 2
        self.map_y = (1.0 / a) * (phi - b)

        output = cv2.remap(
            img,
            self.map_x.astype(numpy.float32),
            self.map_y.astype(numpy.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        if self.f < self.fmin:
            r = numpy.sqrt(numpy.abs((self.f ** 2) / (2 * self.a_ - 1)))
            mask = numpy.zeros_like(output[:, :, 0])
            mask = cv2.circle(
                mask, (int(self.Cx), int(self.Cy)), int(r), (255, 255, 255), -1
            )
            output = cv2.bitwise_and(output, output, mask=mask)

        return output

    def applyMap(self,map,srcFrame):

        if map == 0:
            return cv2.remap(
                srcFrame,
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        if map == 1:
            dstFrame = cv2.remap(
                srcFrame,
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            if self.dice:
                line1 = numpy.hstack(
                    (
                        dstFrame[:, 4 * self.side: 5 * self.side, :] * 0,
                        cv2.flip(dstFrame[:, 4 *
                                             self.side: 5 * self.side, :], 0),
                        dstFrame[:, 4 * self.side: 5 * self.side, :] * 0,
                        dstFrame[:, 4 * self.side: 5 * self.side, :] * 0,
                    )
                )
                line2 = numpy.hstack(
                    (
                        dstFrame[:, 3 * self.side: 4 * self.side, :],
                        dstFrame[:, 0 * self.side: 1 * self.side, :],
                        cv2.flip(dstFrame[:, 1 *
                                             self.side: 2 * self.side, :], 1),
                        cv2.flip(dstFrame[:, 2 *
                                             self.side: 3 * self.side, :], 1),
                    )
                )
                line3 = numpy.hstack(
                    (
                        dstFrame[:, 5 * self.side: 6 * self.side, :] * 0,
                        dstFrame[:, 5 * self.side: 6 * self.side, :],
                        dstFrame[:, 5 * self.side: 6 * self.side, :] * 0,
                        dstFrame[:, 5 * self.side: 6 * self.side, :] * 0,
                    )
                )
                dstFrame = numpy.vstack((line1, line2, line3))
            return dstFrame

        if map == 2:
            h, w = srcFrame.shape[:2]
            if h / w == 3 / 4:
                l1, l2, l3 = numpy.split(srcFrame, 3, axis=0)
                _, pY, _, _ = numpy.split(l1, 4, axis=1)
                nX, pZ, pX, nZ = numpy.split(l2, 4, axis=1)
                _, nY, _, _ = numpy.split(l3, 4, axis=1)
                srcFrame = numpy.hstack(
                    (pZ, cv2.flip(pX, 1), cv2.flip(nZ, 1),
                     nX, cv2.flip(pY, 0), nY)
                )

            dstFrame = 0
            cube_faces = numpy.stack(numpy.split(srcFrame, 6, 1), 0)
            cube_faces[1] = numpy.flip(cube_faces[1], 1)
            cube_faces[2] = numpy.flip(cube_faces[2], 1)
            cube_faces[4] = numpy.flip(cube_faces[4], 0)

            for i in range(6):
                mask = self.tp == i
                dstFrame1 = cv2.remap(
                    cube_faces[i],
                    self.map_x,
                    self.map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                # We use this border mode to avoid small black lines

                dstFrame += cv2.bitwise_and(
                    dstFrame1, dstFrame1, mask=mask.astype(numpy.uint8)
                )

            return dstFrame

        if map == 3:

            dstFrame = cv2.remap(
                srcFrame,
                self.map_x.astype(numpy.float32),
                self.map_y.astype(numpy.float32),
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_WRAP,
            )

            if self.f < self.fmin:
                r = numpy.sqrt(-(self.f ** 2) / (1 - self.xi ** 2))
                mask = numpy.zeros_like(dstFrame[:, :, 0])
                mask = cv2.circle(
                    mask, (int(self.Cx), int(self.Cy)),
                    int(r), (255, 255, 255), -1
                )
                dstFrame = cv2.bitwise_and(dstFrame, dstFrame, mask=mask)

            return dstFrame

        if map == 4:
            return cv2.remap(
                srcFrame,
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        else:
            return print("WRONG MAP ENTERED")