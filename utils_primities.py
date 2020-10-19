import cv2
import os
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_render_CV
import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
class Lines(object):
    def __init__(self):
        self.filename_IDs = []
        self.xyxy = []
        self.ids = []

        return
# ----------------------------------------------------------------------------------------------------------------------
    def make_arrays(self):
        self.filename_IDs = numpy.array(self.filename_IDs).flatten()
        self.xyxy = numpy.array(self.xyxy)
        self.ids = numpy.array(self.ids)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_from_file(self, filename_in,dict_filenames):
        self.__init__()

        if not os.path.exists(filename_in):
            self.make_arrays()
            return

        data = tools_IO.load_mat_pd(filename_in, delim=' ', dtype=numpy.chararray)
        for each in data:
            key = each[0]
            if key in dict_filenames:
                self.filename_IDs.append(dict_filenames[key])
                each = numpy.array(each[1:], dtype=numpy.float)
                self.xyxy.append([int(each[0]), int(each[1]), int(each[2]), int(each[3])])
                self.ids.append(int(each[4]))

        self.make_arrays()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_to_file(self,filename_out,filenames):
        f = open(filename_out, 'w')
        f.write('filename x1 y1 x2 y2 class_ID\n')
        f.close()

        for filename_ID,line,id in zip(self.filename_IDs,self.xyxy,self.ids):
            if numpy.any(numpy.isnan(line)): continue
            tools_IO.save_raw_vec([filenames[int(filename_ID)], int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(id)], filename_out, mode=(os.O_RDWR | os.O_APPEND), fmt='%s', delim=' ')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def append_to_file(self, filename_out, imagenames, lines, IDs):
        if not os.path.isfile(filename_out):
            f = open(filename_out, 'a')
            f.write('filename x1 y1 x2 y2 class_ID\n')
            f.close()

        for imagename,line,id in zip(imagenames,lines,IDs):
            if numpy.any(numpy.isnan(line)): continue
            tools_IO.save_raw_vec([imagename, int(line[0]), int(line[1]), int(line[2]), int(line[3]), id], filename_out, mode=(os.O_RDWR | os.O_APPEND), fmt='%s', delim=' ')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get(self,filename_id):
        idx = (self.filename_IDs == filename_id)
        res_lines = Lines()
        if 1*idx.sum()>0:
            res_lines.filename_IDs = self.filename_IDs[idx]
            res_lines.xyxy = self.xyxy[idx]
            res_lines.ids = self.ids[idx]
        return res_lines
# ----------------------------------------------------------------------------------------------------------------------
    def standartize(self,xyxy):
        res = xyxy
        if xyxy[1]<xyxy[3]:
            res = (xyxy[2],xyxy[3],xyxy[0],xyxy[1])
        elif xyxy[1]==xyxy[3]:
            if xyxy[0]>xyxy[2]:
                res = (xyxy[2], xyxy[3], xyxy[0], xyxy[1])

        return res
# ----------------------------------------------------------------------------------------------------------------------
    def add(self,filename_ID,xyxy,classID,do_standartize=True):
        self.filename_IDs =  numpy.insert(self.filename_IDs,len(self.filename_IDs),filename_ID)
        self.filename_IDs = self.filename_IDs.astype(numpy.int)

        if do_standartize:
            xyxy = self.standartize(xyxy)
        if len(self.xyxy)>0:
            self.xyxy = numpy.vstack((self.xyxy,numpy.array([[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]])))
        else:
            self.xyxy = numpy.array([xyxy])

        self.ids = numpy.insert(self.ids,len(self.ids),int(classID))
        self.ids = self.ids.astype(numpy.int)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def remove_by_cut(self,filename_ID,cut):
        i=0
        while i < len(self.filename_IDs):
            if self.filename_IDs[i]==filename_ID:
                ln = (self.xyxy[i][0], self.xyxy[i][1], self.xyxy[i][2], self.xyxy[i][3])
                if tools_render_CV.do_lines_intersect(ln,cut):
                    self.filename_IDs= numpy.delete(self.filename_IDs,i,axis=0)
                    self.xyxy = numpy.delete(self.xyxy, i, axis=0)
                    self.ids = numpy.delete(self.ids, i, axis=0)
            i+=1

        return
# ----------------------------------------------------------------------------------------------------------------------
    def remove_last(self):
        L = len(self.filename_IDs)
        if L>0:
            self.filename_IDs = numpy.delete(self.filename_IDs,L-1, axis=0)
            self.xyxy = numpy.delete(self.xyxy, L-1, axis=0)
            self.ids = numpy.delete(self.ids, L-1, axis=0)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def remove_by_fileID(self,filename_ID):

        idx = numpy.where(self.filename_IDs==filename_ID)
        if len(idx[0]) > 0:
            self.filename_IDs = numpy.delete(self.filename_IDs, idx, axis=0)
            self.xyxy = numpy.delete(self.xyxy, idx, axis=0)
            self.ids = numpy.delete(self.ids, idx, axis=0)
        return
# ----------------------------------------------------------------------------------------------------------------------
class Landmarks(object):
    def __init__(self):
        self.filename_IDs = []
        self.xy = []
        self.ids = []
        return
# ----------------------------------------------------------------------------------------------------------------------
    def make_arrays(self):
        self.filename_IDs = numpy.array(self.filename_IDs).flatten()
        self.xy = numpy.array(self.xy)
        self.ids = numpy.array(self.ids,dtype=numpy.int)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def copy(self):
        copy = Landmarks()
        copy.filename_IDs = self.filename_IDs.copy()
        copy.ids = self.xy.copy()
        copy.xy = self.ids.copy()
        return copy
# ----------------------------------------------------------------------------------------------------------------------
    def add(self, filename_ID, xy, classID):
        self.filename_IDs = numpy.insert(self.filename_IDs, len(self.filename_IDs), filename_ID).astype(numpy.int)

        if len(self.xy) > 0:
            if len(xy.shape) == 1:
                self.xy = numpy.vstack((self.xy, numpy.array([[xy[0], xy[1]]])))
            else:
                self.xy = numpy.vstack((self.xy, xy))
        else:
            if len(xy.shape)==1:
                self.xy = numpy.array([xy])
            else:
                self.xy = numpy.array(xy)

        self.ids = numpy.insert(self.ids, len(self.ids), classID).astype(numpy.int)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_from_file(self, filename_in,dict_filenames):
        self.__init__()

        if not os.path.isfile(filename_in):return

        for each in tools_IO.load_mat_pd(filename_in, delim=' ',dtype=numpy.chararray):
            key = each[0]
            if key in dict_filenames:
                self.filename_IDs.append(dict_filenames[key])
                each = numpy.array(each[1:], dtype=numpy.float)
                self.xy.append(each[0:-1])
                self.ids.append(each[-1])

        self.make_arrays()
        return
# ----------------------------------------------------------------------------------------------------------------------
class Homographies(object):
    def __init__(self,filename_in=None,dict_filenames = None):
        self.filename_IDs = []
        self.H = []

        if filename_in is not None and dict_filenames is not None:
            self.read_from_file(filename_in,dict_filenames)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def make_arrays(self):
        self.filename_IDs = numpy.array(self.filename_IDs).flatten()
        self.H= numpy.array(self.H)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def append(self, filename_out, imagename, homography):
        if homography is None: return

        if not os.path.isfile(filename_out):
            f = open(filename_out, 'a')
            f.write('filename H H H H H H H H H\n')
            f.close()

        xxx = [imagename] + (homography.flatten()).tolist()
        tools_IO.save_raw_vec(xxx, filename_out, mode=(os.O_RDWR | os.O_APPEND), fmt='%s', delim=' ')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def add(self, filename_ID, h_3x3):
        self.filename_IDs = numpy.insert(self.filename_IDs, len(self.filename_IDs), filename_ID).astype(numpy.int)

        if len(self.H) > 0:
            self.H = numpy.insert(self.H,len(self.H),h_3x3,axis=0)
        else:
            self.H= numpy.array([h_3x3])

        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_from_file(self, filename_in,dict_filenames):
        self.__init__()

        if not os.path.isfile(filename_in):return
        data = tools_IO.load_mat_pd(filename_in, delim=' ', dtype=numpy.chararray)

        if data[:,0].max() in dict_filenames:
            last = dict_filenames[data[:,0].max()]
        else:
            last = len(numpy.unique(data[:,0]))

        for key in dict_filenames:
            if int(dict_filenames[key]) > last:continue

            self.filename_IDs.append(int(dict_filenames[key]))
            i = numpy.where(data[:,0]==key)[0]
            if len(i)==1:
                self.H.append(numpy.array(data[i,1:], dtype=numpy.float).reshape((3,3)))
            else:
                self.H.append(numpy.full((3, 3),numpy.nan))


        self.make_arrays()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_to_file(self, filename_out, filenames):
        f = open(filename_out, 'w')
        f.write('filename H H H H H H H H H\n')
        f.close()
        for filename_ID, h_3x3 in zip(self.filename_IDs, self.H):
            hh = (h_3x3.flatten()).tolist()

            tools_IO.save_raw_vec([filenames[filename_ID]]+hh,filename_out, mode=(os.O_RDWR | os.O_APPEND), fmt='%s', delim=' ')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get(self, filename_id):
        if len(self.filename_IDs)==0:return None
        idx = (self.filename_IDs == filename_id)
        res = None
        if 1*idx.sum() > 0:
            res = self.H[idx][0]

        return res
# ----------------------------------------------------------------------------------------------------------------------
class Ellipses(object):
    def __init__(self):
        self.filename_IDs = []
        self.ellipse = []
        return
# ----------------------------------------------------------------------------------------------------------------------
    def append(self, filename_out, imagename, ellipse):

        if not os.path.isfile(filename_out):
            f = open(filename_out, 'a')
            f.write('filename cx cy ax ay rotation\n')
            f.close()

        if numpy.any(numpy.isnan(ellipse[0])): return
        if numpy.any(numpy.isnan(ellipse[1])): return
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes = (int(ellipse[1][0]/2), int(ellipse[1][1]/2))
        rotation_angle = ellipse[2]
        tools_IO.save_raw_vec([imagename, int(center[0]), int(center[1]), int(axes[0]), int(axes[1]), rotation_angle], filename_out, mode=(os.O_RDWR | os.O_APPEND), fmt='%s', delim=' ')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_from_file(self, filename_in,dict_filenames):
        self.__init__()

        if not os.path.exists(filename_in):
            return

        for each in tools_IO.load_mat_pd(filename_in, delim=' ', dtype=numpy.chararray):
            key = each[0]#.decode("utf-8")
            if key in dict_filenames:
                self.filename_IDs.append(dict_filenames[key])
                each = numpy.array(each[1:],dtype=numpy.float)
                ellipse = [(int(each[0]),int(each[1])),(int(2*each[2]),int(2*each[3])),each[4]]
                self.ellipse.append(ellipse)

        self.make_arrays()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_to_file(self, filename_out, filenames):
        f = open(filename_out, 'w')
        f.write('filename cx cy rx ry ang_deg class_ID\n')
        f.close()

        for filename_ID, ellipse in zip(self.filename_IDs, self.ellipse):
            tools_IO.save_raw_vec([filenames[filename_ID], int(ellipse[0][0]),
                                   int(ellipse[0][1]), int(ellipse[1][0] // 2),
                                   int(ellipse[1][1] // 2), int(ellipse[2]), 0],
                                  filename_out, mode=(os.O_RDWR | os.O_APPEND),fmt='%s', delim=' ')

        return

# ----------------------------------------------------------------------------------------------------------------------
    def make_arrays(self):
        self.filename_IDs = numpy.array(self.filename_IDs).flatten()
        self.ellipse= numpy.array(self.ellipse)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get(self,filename_id):
        if len(self.filename_IDs)==0:
            return None
        idx = (self.filename_IDs == filename_id)
        res = None
        if 1*idx.sum()>0:
            res = self.ellipse[idx]
            if res is not None:
                res.tolist()

        return res
# ----------------------------------------------------------------------------------------------------------------------
    def add(self,filename_ID,e5,classID):
        self.filename_IDs =  numpy.insert(self.filename_IDs,len(self.filename_IDs),filename_ID)

        if len(self.ellipse)>0:
            self.ellipse = numpy.vstack((self.ellipse,e5))
        else:
            self.ellipse= numpy.array([e5])

        return
# ----------------------------------------------------------------------------------------------------------------------
    def remove_by_fileID(self,filename_ID):

        idx = numpy.where(self.filename_IDs==filename_ID)
        if len(idx[0]) > 0:
            self.filename_IDs = numpy.delete(self.filename_IDs, idx, axis=0)
            self.ellipse = numpy.delete(self.ellipse, idx, axis=0)

        return
# ----------------------------------------------------------------------------------------------------------------------
class Balls(object):
    def __init__(self,filename_in = None, dict_filenames=None):
        self.filename_IDs = []
        self.xyxy = []
        self.confidence = []

        if filename_in is not None and dict_filenames is not None:
            self.read_from_file(filename_in,dict_filenames)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def make_arrays(self):
        self.filename_IDs = numpy.array(self.filename_IDs).flatten()
        self.xyxy = numpy.array(self.xyxy)
        self.confidence = numpy.array(self.confidence)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_from_folder(self, folder_in):
        self.__init__()

        if not os.path.exists(folder_in):
            self.make_arrays()
            return

        filenames = tools_IO.get_filenames(folder_in,'*.txt')
        filenames = numpy.array([f.split('.')[0] for f in filenames], dtype=int)
        filename_IDs = numpy.sort(filenames).astype(numpy.chararray)
        filenames = [str(id) +'.txt' for id in filename_IDs]


        for filename, id in zip(filenames,filename_IDs):
            print(filename,id)
            data = tools_IO.load_mat_pd(folder_in+filename,delim=' ')
            if len(data)==0:continue
            for X in data:
                self.xyxy.append(numpy.array(X[2:],dtype=float))
                self.filename_IDs.append(int(id)-1)
                self.confidence.append((int(X[1])))

        self.make_arrays()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_from_folder2(self, folder_in):
        self.__init__()

        if not os.path.exists(folder_in):
            self.make_arrays()
            return

        filenames = tools_IO.get_filenames(folder_in, '*.txt')
        filenames = numpy.array([f.split('.')[0] for f in filenames], dtype=int)
        filename_IDs = numpy.sort(filenames).astype(numpy.chararray)
        filenames = [str(id) + '.txt' for id in filename_IDs]

        for filename, id in zip(filenames, filename_IDs):
            print(filename, id)
            data = tools_IO.load_mat_var_size(folder_in + filename)[1:]
            #data = numpy.array(data)
            if len(data) == 0: continue
            #if len((data))==1:
            #    data = [data]
            for X in data:
                X=X[0].split(' ')
                self.xyxy.append(numpy.array(X[2:], dtype=float))
                self.filename_IDs.append(int(id) - 1)
                self.confidence.append((int(X[1])))

        self.make_arrays()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_from_file(self, filename_in, dict_filenames):
        self.__init__()
        if not os.path.isfile(filename_in): return
        data = tools_IO.load_mat_pd(filename_in, delim=' ', dtype=numpy.chararray)

        mode_filenames = isinstance(data[0][0],numpy.str)

        for each in data:
            key = each[0]
            if not mode_filenames:
                self.filename_IDs.append(key)
                self.xyxy.append(numpy.array(each[1:-1], dtype=numpy.float))
                self.confidence.append(1.0)
            else:
                if key in dict_filenames:
                    self.filename_IDs.append(dict_filenames[key])
                    self.xyxy.append(numpy.array(each[1:-1], dtype=numpy.float))
                    self.confidence.append(1.0)



        self.make_arrays()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def standartize(self, xyxy):
        res = xyxy
        if xyxy[1] < xyxy[3]:
            res = (xyxy[2], xyxy[3], xyxy[0], xyxy[1])
        elif xyxy[1] == xyxy[3]:
            if xyxy[0] > xyxy[2]:
                res = (xyxy[2], xyxy[3], xyxy[0], xyxy[1])

        return res
# ----------------------------------------------------------------------------------------------------------------------
    def add(self,filename_ID,xyxy,classID):
        self.filename_IDs =  numpy.insert(self.filename_IDs,len(self.filename_IDs),filename_ID)
        xyxy = self.standartize(xyxy)
        if len(self.xyxy)>0:
            self.xyxy = numpy.vstack((self.xyxy,numpy.array([[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]])))
        else:
            self.xyxy = numpy.array([xyxy])

        self.ids = numpy.insert(self.ids,len(self.ids),int(classID))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def append(self, filename_out, imagename, xy):
        if xy is None: return

        if not os.path.isfile(filename_out):
            f = open(filename_out, 'a')
            f.write('filename x y x y c\n')
            f.close()

        xxx = [imagename] + (xy.flatten()).tolist() + [1]
        tools_IO.save_raw_vec(xxx, filename_out, mode=(os.O_RDWR | os.O_APPEND), fmt='%s', delim=' ')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_to_file(self, filename_out, filenames):
        f = open(filename_out, 'w')
        f.write('filename x y x y c\n')
        f.close()

        for filename_ID, xyxy, cnf in zip(self.filename_IDs, self.xyxy, self.confidence):
            if numpy.any(numpy.isnan(xyxy)): continue
            tools_IO.save_raw_vec(
                [filenames[filename_ID], int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), cnf], filename_out,mode=(os.O_RDWR | os.O_APPEND), fmt='%s', delim=' ')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get(self,filename_id):
        idx = (self.filename_IDs == filename_id)
        res = Balls()
        if 1*idx.sum()>0:
            res.filename_IDs = self.filename_IDs[idx]
            res.xyxy = self.xyxy[idx]
            res.confidence= self.confidence[idx]
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def transform(self,h_3x3):
        if self.xyxy is not None and len(self.xyxy)>0:
            self.xyxy = tools_pr_geom.perspective_transform(numpy.array(self.xyxy).reshape((-1, 1, 2)).astype(numpy.float32), h_3x3).reshape((-1,4))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def erode(self,R):
        if self.xyxy is not None and len(self.xyxy)>0:
            for i in range(len(self.xyxy)):
                self.xyxy[i] = numpy.array((self.xyxy[i][0]-R,self.xyxy[i][1]-R,self.xyxy[i][2]+R,self.xyxy[i][3]+R))
        return
# ----------------------------------------------------------------------------------------------------------------------
class Players(object):
    def __init__(self,filename_in = None, dict_filenames=None):
        self.filename_IDs = []
        self.xyxy = []
        self.ids = []


        if filename_in is not None and dict_filenames is not None:
            self.read_from_file(filename_in,dict_filenames)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def make_arrays(self):
        self.filename_IDs = numpy.array(self.filename_IDs).flatten()
        self.xyxy = numpy.array(self.xyxy)
        self.ids = numpy.array(self.ids)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_from_folder(self, folder_in):
        self.__init__()

        if not os.path.exists(folder_in):
            self.make_arrays()
            return

        filenames = tools_IO.get_filenames(folder_in,'*.txt')
        filenames = numpy.array([f.split('.')[0] for f in filenames], dtype=int)
        filename_IDs = numpy.sort(filenames).astype(numpy.chararray)
        filenames = [str(id) +'.txt' for id in filename_IDs]


        for filename, id in zip(filenames,filename_IDs):
            data = tools_IO.load_mat_pd(folder_in+filename,delim=' ')
            if len(data)==0:continue
            for X in data:
                self.xyxy.append(numpy.array(X[2:],dtype=float))
                self.filename_IDs.append(int(id)-1)
                self.confidence.append((int(X[1])))

        self.make_arrays()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_from_file(self, filename_in, dict_filenames):
        self.__init__()
        if not os.path.isfile(filename_in): return
        data = tools_IO.load_mat_pd(filename_in, delim=' ', dtype=numpy.chararray)

        mode_filenames = isinstance(data[0][0],numpy.str)

        for each in data:
            key = each[0]
            if not mode_filenames:
                self.filename_IDs.append(key)
                self.xyxy.append(numpy.array(each[1:-1], dtype=numpy.float))
                self.ids.append(each[-1])
            else:
                if key in dict_filenames:
                    self.filename_IDs.append(dict_filenames[key])
                    self.xyxy.append(numpy.array(each[1:-1], dtype=numpy.float))
                    self.ids.append(int(each[-1]))



        self.make_arrays()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_to_file(self, filename_out, filenames):

        data = [['filename','x1','y1','x2','y2','c']]

        for filename_ID, box, id in zip(self.filename_IDs, self.xyxy, self.ids):
            if numpy.any(numpy.isnan(box)): continue
            vec = [filenames[filename_ID], int(box[0]), int(box[1]), int(box[2]), int(box[3]), id]
            data.append(vec)

        data = numpy.array(data)
        tools_IO.save_mat(data, filename_out,fmt='%s',delim=' ')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def append(self, filename_out, imagename, xy):
        if xy is None: return

        if not os.path.isfile(filename_out):
            f = open(filename_out, 'a')
            f.write('filename x y x y c\n')
            f.close()

        xxx = [imagename] + (xy.flatten()).tolist() + [1]
        tools_IO.save_raw_vec(xxx, filename_out, mode=(os.O_RDWR | os.O_APPEND), fmt='%s', delim=' ')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def standartize(self, xyxy):
        res = xyxy
        if xyxy[1] < xyxy[3]:
            res = (xyxy[2], xyxy[3], xyxy[0], xyxy[1])
        elif xyxy[1] == xyxy[3]:
            if xyxy[0] > xyxy[2]:
                res = (xyxy[2], xyxy[3], xyxy[0], xyxy[1])

        return res
# ----------------------------------------------------------------------------------------------------------------------
    def add(self, filename_ID, xyxy, classID):
        self.filename_IDs = numpy.insert(self.filename_IDs, len(self.filename_IDs), int(filename_ID)).astype(numpy.int)
        xyxy = self.standartize(xyxy)
        if len(self.xyxy) > 0:
            self.xyxy = numpy.vstack(
                (self.xyxy, numpy.array([[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]])))
        else:
            self.xyxy = numpy.array([xyxy])

        self.ids = numpy.insert(self.ids, len(self.ids), int(classID)).astype(numpy.int)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def remove_by_fileID(self,filename_ID):

        idx = numpy.where(self.filename_IDs==filename_ID)
        if len(idx[0]) > 0:
            self.filename_IDs = numpy.delete(self.filename_IDs, idx, axis=0)
            self.xyxy = numpy.delete(self.xyxy, idx, axis=0)
            self.ids = numpy.delete(self.ids, idx, axis=0)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get(self,filename_id):
        idx = (self.filename_IDs == filename_id)
        res = Players()
        if 1*idx.sum()>0:
            res.filename_IDs = self.filename_IDs[idx]
            res.xyxy = self.xyxy[idx]
            res.ids= self.ids[idx]
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def remove_by_cut(self,filename_ID,cut_line):
        i=0
        while i < len(self.filename_IDs):
            if self.filename_IDs[i]==filename_ID:
                box = (self.xyxy[i][0], self.xyxy[i][1], self.xyxy[i][2], self.xyxy[i][3])
                res = tools_render_CV.trim_line_by_box(cut_line,box)

                if not numpy.any(numpy.isnan(res)):
                    self.filename_IDs= numpy.delete(self.filename_IDs,i,axis=0)
                    self.xyxy = numpy.delete(self.xyxy, i, axis=0)
                    self.ids = numpy.delete(self.ids, i, axis=0)
            i+=1

        return
# ----------------------------------------------------------------------------------------------------------------------
    def remove_by_point(self,filename_ID,coord):
        i=0
        while i < len(self.filename_IDs):
            if self.filename_IDs[i]==filename_ID:
                #box = (self.xyxy[i][0], self.xyxy[i][1], self.xyxy[i][2], self.xyxy[i][3])
                left = min(self.xyxy[i][0],self.xyxy[i][2])
                right = max(self.xyxy[i][0], self.xyxy[i][2])
                top = min(self.xyxy[i][1], self.xyxy[i][3])
                bottom = max(self.xyxy[i][1], self.xyxy[i][3])

                if left <= coord[0] and coord[0] <= right and top <= coord[1] and coord[1] <= bottom:
                    self.filename_IDs= numpy.delete(self.filename_IDs,i,axis=0)
                    self.xyxy = numpy.delete(self.xyxy, i, axis=0)
                    self.ids = numpy.delete(self.ids, i, axis=0)
            i+=1

        return
# ----------------------------------------------------------------------------------------------------------------------
class Polygons(object):
    def __init__(self,filename_in = None, dict_filenames=None):
        self.filename_IDs = []
        self.xy = []
        self.ids = []


        if filename_in is not None and dict_filenames is not None:
            self.read_from_file(filename_in,dict_filenames)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def make_arrays(self):
        self.filename_IDs = numpy.array(self.filename_IDs).flatten()
        self.xy = numpy.array(self.xy)
        self.ids = numpy.array(self.ids)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_from_file(self, filename_in, dict_filenames):
        self.__init__()
        if not os.path.isfile(filename_in): return
        #data = tools_IO.load_mat_pd(filename_in, delim=' ', dtype=numpy.chararray)
        data = tools_IO.load_mat_var_size(filename_in,dtype=numpy.chararray,delim=' ')[1:]

        mode_filenames = True

        for each in data:
            key = each[0]
            if key in dict_filenames:
                self.filename_IDs.append(dict_filenames[key])
                #self.xyxy.append(numpy.array(each[1:-1], dtype=numpy.float))
                #self.ids.append(int(each[-1]))


        self.make_arrays()
        return
# ----------------------------------------------------------------------------------------------------------------------