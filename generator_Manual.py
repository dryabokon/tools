import numpy
import tools_IO as IO
import matplotlib.pyplot as plt
# --------------------------------------------------------------------------------------------------------------------
class generator_Manual(object):
    def __init__(self, dim=2):
        self.X0 = []
        self.X1 = []
        self.filename_train = []
        self.filename_test = []
# ----------------------------------------------------------------------------------------------------------------------
    def create_train_test_samples(self,filename_train,filename_test):

        def OnClose(event):

            print(event)
            self.X0 = numpy.array(self.X0)
            self.X1 = numpy.array(self.X1)

            Y0 = numpy.full(self.X0.shape[0], -1).astype('float32')
            Y1 = numpy.full(self.X1.shape[0], +1).astype('float32')

            data_train = numpy.vstack((self.X0, self.X1))
            target_train = numpy.hstack((Y0, Y1))
            #IO.save_data_to_feature_file_float(self.filename_train, data_train, target_train)
            #IO.save_data_to_feature_file_float(self.filename_test,  data_train, target_train)

            return

        def OnClick(event):

            print((event.x, event.y))
            print(event)
            if(event.button==1):
                self.X0.append((event.x, event.y))
                plt.plot(event.x, event.y, 'ro', color='red', alpha=0.4)
            else:
                self.X1.append((event.x, event.y))
                plt.plot(event.x, event.y, 'ro', color='blue', alpha=0.4)

            plt.show()
            return

        self.filename_train = filename_train
        self.filename_test = filename_test

        fig = plt.gcf()
        size = fig.get_size_inches() * fig.dpi
        axes = plt.gca()

        axes.set_xlim([0, size[0]])
        axes.set_ylim([0, size[1]])
        axes.set_position([0,0,1,1])

        cid = fig.canvas.mpl_connect('button_press_event', OnClick)
        #fig.canvas.mpl_connect('close_event', OnClose)



        return
# ----------------------------------------------------------------------------------------------------------------------
