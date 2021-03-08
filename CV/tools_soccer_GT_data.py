import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_render_CV
# ----------------------------------------------------------------------------------------------------------------------
# lines
# 0,1,2,3  - outer bounds
# 4,5,6    - left  penalty area
# 7,8,9    - left  goal area
# 10,11,12 - right penalty area
# 13,14,15 - right goal area
# ----------------------------------------------------------------------------------------------------------------------
class Soccer_Field_GT_data(object):
    def __init__(self):
        self.w_fild = 100.0
        self.h_fild = 75.0
        self.w_penl = 18.0
        self.h_penl = 44.0
        self.w_goal = 6.0
        self.h_goal = 20.0
        self.dict_landmark_ID = {}
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_list_of_lines(self):
        list3 = [[3, 0], [3, 2], [3, 4], [3, 6], [3, 7], [3, 9]]
        list5 = [[5, 0], [5, 2], [5, 4], [5, 6], [5, 7], [5, 9]]
        list8 = [[8, 0], [8, 2], [8, 4], [8, 6], [8, 7], [8, 9]]

        list1  = [[1 , 0], [ 1, 2], [ 1, 10], [ 1, 12], [ 1, 13], [ 1, 15]]
        list11 = [[11, 0], [11, 2], [11, 10], [11, 12], [11, 13], [11, 15]]
        list14 = [[14, 0], [14, 2], [14, 10], [14, 12], [14, 13], [14, 15]]
        list_of_lines = list3 + list5 + list8 + list1 + list11 + list14
        return list_of_lines
# ----------------------------------------------------------------------------------------------------------------------
    def get_list_of_X_crosses(self):
        res = [0,1,8,9,16,17,18,19,26,27,34,35]
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def get_list_of_T_crosses(self):
        res = [2,3,4,5,20,21,22,23]
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def get_GT(self):
        scale_factor = 20
        w_fild = self.w_fild
        h_fild = self.h_fild
        w_penl = self.w_penl
        h_penl = self.h_penl
        w_goal = self.w_goal
        h_goal = self.h_goal

        lines = [[-w_fild/2, -h_fild/2, +w_fild/2,          -h_fild/2], [+w_fild/2,          -h_fild/2, +w_fild/2         , +h_fild/2],[+w_fild/2         , +h_fild/2, -w_fild/2, +h_fild/2],[-w_fild/2,+h_fild/2,-w_fild/2,-h_fild/2],
                 [-w_fild/2, -h_penl/2, -w_fild/2 + w_penl, -h_penl/2], [-w_fild/2 + w_penl, -h_penl/2, -w_fild/2 + w_penl, +h_penl/2],[-w_fild/2 + w_penl, +h_penl/2, -w_fild/2, +h_penl/2],
                 [-w_fild/2, -h_goal/2, -w_fild/2 + w_goal, -h_goal/2], [-w_fild/2 + w_goal, -h_goal/2, -w_fild/2 + w_goal, +h_goal/2],[-w_fild/2 + w_goal, +h_goal/2, -w_fild/2, +h_goal/2],
                 [+w_fild/2, -h_penl/2, +w_fild/2 - w_penl, -h_penl/2], [+w_fild/2 - w_penl, -h_penl/2, +w_fild/2 - w_penl, +h_penl/2],[+w_fild/2 - w_penl, +h_penl/2, +w_fild/2, +h_penl/2],
                 [+w_fild/2, -h_goal/2, +w_fild/2 - w_goal, -h_goal/2], [+w_fild/2 - w_goal, -h_goal/2, +w_fild/2 - w_goal, +h_goal/2],[+w_fild/2 - w_goal, +h_goal/2, +w_fild/2, +h_goal/2]]

        lines = numpy.array(lines)

        lines[:,0]+=w_fild/2
        lines[:,1]+=h_fild/2
        lines[:,2]+=w_fild/2
        lines[:,3]+=h_fild/2

        lines = scale_factor*numpy.array(lines)

        landmarks = []
        for l1, l2 in self.get_list_of_lines():
            landmarks.append(tools_render_CV.line_intersection(lines[l1],lines[l2]))

        landmarks = numpy.array(landmarks)
        return landmarks, lines
# ----------------------------------------------------------------------------------------------------------------------
    def get_landmark_ID(self,idx_v, idx_h):
        if tuple((idx_v, idx_h)) not in self.dict_landmark_ID:
            res = None
            for c, (l1, l2) in enumerate(self.get_list_of_lines()):
                if l1==idx_v and l2==idx_h:
                    res=c
            self.dict_landmark_ID[tuple((idx_v, idx_h))] = res
            return res
        else:
            res = self.dict_landmark_ID[tuple((idx_v, idx_h))]
        return res
# ----------------------------------------------------------------------------------------------------------------------