#see github.com/qqwweee/keras-yolo3
# pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
# ----------------------------------------------------------------------------------------------------------------------
import numpy
import cv2
import os
from os import listdir
import fnmatch
import time
# ----------------------------------------------------------------------------------------------------------------------
import detector_YOLO3_core
import tools_YOLO
import tools_image
import tools_IO
import tools_HDF5
# ----------------------------------------------------------------------------------------------------------------------
from keras import backend as K
from keras.layers import Lambda, Input, Concatenate
from keras.callbacks import EarlyStopping
from keras.models import Model, load_model
# ----------------------------------------------------------------------------------------------------------------------
class Logger(object):
    def __init__(self):
        self.data_source = None
        self.time_train = None
        self.time_validate = None
        self.time_test = None
        self.AP_train = None
        self.AP_test = None
        self.last_layers = None

# ----------------------------------------------------------------------------------------------------------------------
class detector_YOLO3(object):

    def __init__(self, model_weights_h5,filename_metadata,obj_threshold=None):
        if model_weights_h5 is not None and filename_metadata is not None:
            self.__load_model(model_weights_h5,filename_metadata,obj_threshold)

        self.logs = Logger()
# ----------------------------------------------------------------------------------------------------------------------
    def init_tiny(self,default_weights_h5, num_classes):

        self.input_image_size,self.class_names,self.anchors,self.anchor_mask,self.obj_threshold,self.nms_threshold = tools_YOLO.init_default_metadata_tiny(num_classes)
        self.colors = tools_YOLO.generate_colors(len(self.class_names))
        self.input_tensor_shape = K.placeholder(shape=(2,))
        self.sess = K.get_session()
        self.model = detector_YOLO3_core.yolo_body_tiny(Input(shape=(None, None, 3)), 3, num_classes)
        default_model = load_model(default_weights_h5, compile=False)
        detector_YOLO3_core.assign_weights(default_model,self.model)

        self.boxes, self.scores, self.classes = detector_YOLO3_core.get_tensors_box_score_class(self.model.output,self.anchors, self.anchor_mask,len(self.class_names),
                                                                                                    self.input_tensor_shape,score_threshold=self.obj_threshold,iou_threshold=self.nms_threshold)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_model(self,filename_out):
        self.model.save(filename_out)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __load_model(self,model_weights_h5,filename_metadata,obj_threshold=None):

        self.input_image_size,self.class_names,self.anchors,self.anchor_mask, self.obj_threshold,self.nms_threshold = tools_YOLO.load_metadata(filename_metadata)
        if obj_threshold is not None: self.obj_threshold=obj_threshold
        self.colors = tools_YOLO.generate_colors(len(self.class_names))
        self.input_tensor_shape = K.placeholder(shape=(2,))
        self.sess = K.get_session()

        self.model = load_model(model_weights_h5, compile=False)

        self.boxes, self.scores, self.classes = detector_YOLO3_core.get_tensors_box_score_class(self.model.output,self.anchors,self.anchor_mask, len(self.class_names),
                                                                                                    self.input_tensor_shape,score_threshold=self.obj_threshold,iou_threshold=self.nms_threshold)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_image(self, image):

        image_resized = tools_image.smart_resize(image, self.input_image_size[0], self.input_image_size[1])
        image_resized = numpy.expand_dims(image_resized / 255.0, axis=0)

        boxes_yxyx, classes, scores = self.sess.run([self.boxes, self.classes, self.scores],
                                                    feed_dict={self.model.input: image_resized,
                                                               self.input_tensor_shape: [image.shape[0],
                                                                                         image.shape[1]],
                                                               K.learning_phase(): 0})

        return boxes_yxyx, classes, scores
# ----------------------------------------------------------------------------------------------------------------------
    def process_file_debug(self, filename_in, folder_out):
        image = cv2.imread(filename_in)

        image_resized = tools_image.smart_resize(image, self.input_image_size[0], self.input_image_size[1])
        image_resized = numpy.expand_dims(image_resized / 255.0, axis=0)

        u_boxes, u_scores, u_classes = detector_YOLO3_core.get_tensors_box_score_class_unfilter(
            self.model.output, self.anchors, self.anchor_mask, len(self.class_names),
            self.input_tensor_shape, score_threshold=0.01, iou_threshold=self.nms_threshold)

        boxes_yxyx, classes, scores = self.sess.run([u_boxes, u_classes, u_scores],feed_dict={self.model.input: image_resized,self.input_tensor_shape: [image.shape[0],image.shape[1]],K.learning_phase(): 0})

        self.process_file(filename_in, folder_out + filename_in.split('/')[-1])

        total_image = tools_image.desaturate(image)

        for c in list(set(classes)):
            idx = numpy.where(classes == c)
            temp_image  = tools_YOLO.draw_classes_on_image(tools_image.desaturate(image), boxes_yxyx[idx], [1]*len(idx[0]), self.colors[c],draw_score=False)
            total_image = tools_YOLO.draw_classes_on_image(total_image                  , boxes_yxyx[idx], scores[idx]    , self.colors[c],draw_score=True )
            cv2.imwrite(folder_out+'class_%02d-%s-p%02d.png'%(c,self.class_names[c],100*scores[idx].max()),temp_image)

        cv2.imwrite(folder_out + 'all_boxes.png', total_image)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_file(self, filename_in, filename_out):

        image = cv2.imread(filename_in)
        boxes_yxyx, classes, scores  = self.process_image(image)

        tools_YOLO.draw_and_save(filename_out, image, boxes_yxyx, scores, classes,self.colors, self.class_names)
        markup = tools_YOLO.get_markup(filename_in,boxes_yxyx,scores,classes)

        return markup
# ----------------------------------------------------------------------------------------------------------------------
    def process_folder(self, path_input, path_out, mask='*.jpg', limit=1000000,markup_only=False):
        tools_IO.remove_files(path_out)
        start_time = time.time()
        local_filenames = numpy.array(fnmatch.filter(listdir(path_input), mask))[:limit]
        result = [('filename', 'x_right','y_top','x_left','y_bottom','class_ID','confidence')]
        local_filenames = numpy.sort(local_filenames)
        for local_filename in local_filenames:
            filename_out = path_out + local_filename if not markup_only else None
            for each in self.process_file(path_input + local_filename, filename_out):
                result.append(each)
            tools_IO.save_mat(result, path_out + 'markup_res.txt', delim=' ')
        total_time = (time.time() - start_time)
        print('Processing: %s sec in total - %f per image' % (total_time, int(total_time) / len(local_filenames)))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_annotation(self, file_annotations, filename_markup_out_true,filename_markup_out_pred,folder_annotation=None, markup_only=False,limit=1000000):

        start_time = time.time()
        if folder_annotation is None:
            foldername = '/'.join(file_annotations.split('/')[:-1]) + '/'
        else:
            foldername = folder_annotation

        result = [('filename', 'x_right', 'y_top', 'x_left', 'y_bottom', 'class_ID', 'confidence')]
        fact   = [('filename', 'x_right', 'y_top', 'x_left', 'y_bottom', 'class_ID', 'confidence')]

        with open(file_annotations) as f:lines = f.readlines()[1:limit]
        list_filenames = sorted(set([foldername+line.split(' ')[0] for line in lines]))
        for each in lines:
            each = each.split('\n')[0]
            fact.append(each.split(' '))

        tools_IO.save_mat(fact, filename_markup_out_true, delim=' ')


        for local_filename in list_filenames:
            for each in self.process_file(local_filename, None):
                result.append(each)
            tools_IO.save_mat(result,filename_markup_out_pred, delim=' ')




        total_time = (time.time() - start_time)
        print('Processing: %s sec in total - %f per image' % (total_time, int(total_time) / len(list_filenames)))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_bottleneck_features(self, filenames_list, dict_bottlenects):

        images_resized = []
        for filename in filenames_list:
            image_resized = tools_image.smart_resize(cv2.imread(filename), self.input_image_size[0],
                                                     self.input_image_size[1])
            images_resized.append(image_resized / 255.0)

        images_resized = numpy.array(images_resized).astype(numpy.float)
        L = len(self.model.layers)

        outputs = [self.model.layers[L + i].output for i in dict_bottlenects.values()]
        bottlenecks = Model(self.model.input, outputs).predict(images_resized)
        return bottlenecks
# ----------------------------------------------------------------------------------------------------------------------
    def save_bottleneck_features(self, folder_out, filenames_list, dict_bottlenects):

        outputs = [self.model.layers[len(self.model.layers) + i].output for i in dict_bottlenects.values()]

        image_resized = numpy.zeros((1,self.input_image_size[0],self.input_image_size[1],3))
        bottlenecks = Model(self.model.input, outputs).predict(image_resized)
        store0 = tools_HDF5.HDF5_store(filename=folder_out+'bottlenecks_0.hdf5', object_shape=bottlenecks[0][0].shape,dtype=numpy.float32)
        store1 = tools_HDF5.HDF5_store(filename=folder_out+'bottlenecks_1.hdf5', object_shape=bottlenecks[1][0].shape,dtype=numpy.float32)

        for filename in filenames_list:
            image_resized = tools_image.smart_resize(cv2.imread(filename), self.input_image_size[0], self.input_image_size[1])
            image_resized = numpy.expand_dims(image_resized / 255.0, axis=0)
            bottlenecks = Model(self.model.input, outputs).predict(image_resized)
            store0.append(bottlenecks[0][0])
            store1.append(bottlenecks[1][0])

        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_default_tiny_model_metadata(self, default_model_weights, num_classes, filename_model_out,filename_metadata_out):
        out_model = detector_YOLO3_core.yolo_body_tiny(Input(shape=(None, None, 3)), 3, num_classes)
        default_model = load_model(default_model_weights, compile=False)
        detector_YOLO3_core.assign_weights(default_model, out_model)
        out_model.save(filename_model_out)
        input_image_size, class_names, anchors, anchor_mask, obj_threshold, nms_threshold = tools_YOLO.init_default_metadata_tiny(num_classes)
        tools_YOLO.save_metadata(filename_metadata_out, input_image_size, class_names, anchors, anchor_mask,obj_threshold, nms_threshold)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_default_full_model_metadata(self, default_model_weights, num_classes, filename_model_out,filename_metadata_out):
        out_model = detector_YOLO3_core.yolo_body_full(Input(shape=(None, None, 3)), 3, num_classes)
        default_model = load_model(default_model_weights, compile=False)
        detector_YOLO3_core.assign_weights(default_model, out_model)
        out_model.save(filename_model_out)
        input_image_size, class_names, anchors, anchor_mask, obj_threshold, nms_threshold = tools_YOLO.init_default_metadata_full(
            num_classes)
        tools_YOLO.save_metadata(filename_metadata_out, input_image_size, class_names, anchors, anchor_mask,
                                 obj_threshold, nms_threshold)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def construct_last_layers_placeholders(self, L, dict_last_layers, dict_bottlenects):

        placeholder_btlncs = [Input(shape=self.model.layers[L + i].output.shape[1:].as_list(),name='btlncs%d'%n) for n,i in enumerate(dict_bottlenects.values())]

        mid_layer={}

        for key in dict_last_layers.keys():
            values = dict_last_layers[key]
            if numpy.min(values)>=0:
                if len(values)==1:
                    t = placeholder_btlncs[values[0]]
                    mid_layer[key] = self.model.layers[L + key](t)
                else:
                    xxx = []
                    for i in values:xxx.append(placeholder_btlncs[i])
                    mid_layer[key] = Concatenate()(xxx)

        flag = 1
        while flag==1:
            flag=0
            for key in dict_last_layers.keys():
                if key<0 and (key not in mid_layer):
                    values = dict_last_layers[key]
                    if numpy.max(values) < 0:
                        if len(values) == 1 and (values[0] in mid_layer):
                            t = mid_layer[values[0]]
                            mid_layer[key] = self.model.layers[L + key](t)
                            flag = 1
                        else:
                            if numpy.min([1*(each in mid_layer) for each in values])==1:
                                xxx = []
                                for i in values: xxx.append(mid_layer[i])
                                mid_layer[key] = Concatenate()(xxx)
                                flag = 1


        last_layers_outs=[]
        for i in dict_last_layers[0]:
            last_layers_outs.append(mid_layer[i])

        return placeholder_btlncs,last_layers_outs
# ----------------------------------------------------------------------------------------------------------------------
    def __calc_loss(self,tensor_loss,placeholder_btlncs,placeholder_yolo_features,bottlenecks,targets):
        loss = []
        nlyrs = len(targets)

        for i in range(bottlenecks[0].shape[0]):
            dct={placeholder_btlncs[0]: numpy.array([bottlenecks[0][i]]),
                placeholder_btlncs[1]: numpy.array([bottlenecks[1][i]]),
                placeholder_yolo_features[0]: numpy.array([targets[0][i]]),
                placeholder_yolo_features[1]: numpy.array([targets[1][i]])}

            if nlyrs==3:
                dct[placeholder_btlncs[2]]=numpy.array([bottlenecks[2][i]])
                dct[placeholder_yolo_features[2]] = numpy.array(numpy.array([targets[2][i]]))

            lv = self.sess.run(tensor_loss, feed_dict=dct)
            loss.append(lv)

        return loss
# ----------------------------------------------------------------------------------------------------------------------
    def __last_layers2(self):
        dict_last_layers, dict_bottlenects = {}, {}

        dict_last_layers[0] = [-2, -1]
        dict_last_layers[-2], dict_last_layers[-1] = [-4], [-3]
        dict_last_layers[-4], dict_last_layers[-3] = [0], [1]

        dict_bottlenects[0] = -6
        dict_bottlenects[1] = -5
        return dict_last_layers, dict_bottlenects
# ----------------------------------------------------------------------------------------------------------------------
    def __last_layers3(self):
        dict_last_layers, dict_bottlenects = {}, {}

        dict_last_layers[0] = [-2, -1]
        dict_last_layers[-2], dict_last_layers[-1] = [-4], [-3]
        dict_last_layers[-4], dict_last_layers[-3] = [-6], [-5]
        dict_last_layers[-6], dict_last_layers[-5] = [0], [1]

        dict_bottlenects[0] = -8
        dict_bottlenects[1] = -7
        return dict_last_layers, dict_bottlenects
# ----------------------------------------------------------------------------------------------------------------------
    def __last_layers4(self):
        dict_last_layers, dict_bottlenects = {}, {}

        dict_last_layers[0] = [-2, -1]
        dict_last_layers[-2], dict_last_layers[-1] = [-4], [-3]
        dict_last_layers[-4], dict_last_layers[-3] = [-6], [-5]
        dict_last_layers[-6], dict_last_layers[-5] = [-8], [-7]
        dict_last_layers[-8], dict_last_layers[-7] = [0], [1]

        dict_bottlenects[0] = -14
        dict_bottlenects[1] = -9
        return dict_last_layers, dict_bottlenects
# ----------------------------------------------------------------------------------------------------------------------
    def __last_layers5(self):
        dict_last_layers, dict_bottlenects = {}, {}

        dict_last_layers[0] = [-2, -1]
        dict_last_layers[-2], dict_last_layers[-1] = [-4], [-3]
        dict_last_layers[-4], dict_last_layers[-3] = [-6], [-5]
        dict_last_layers[-6], dict_last_layers[-5] = [-8], [-7]
        dict_last_layers[-8], dict_last_layers[-7] = [-14], [-9]
        dict_last_layers[-14], dict_last_layers[-9] = [0], [1, 2]

        dict_bottlenects[0] = -15
        dict_bottlenects[1] = -10
        dict_bottlenects[2] = -25
        return dict_last_layers, dict_bottlenects
# ----------------------------------------------------------------------------------------------------------------------
    def __last_layers14(self):
        dict_last_layers, dict_bottlenects = {}, {}

        dict_last_layers[0] = [-2, -1]
        dict_last_layers[-2], dict_last_layers[-1] = [-4], [-3]
        dict_last_layers[-4], dict_last_layers[-3] = [-6], [-5]
        dict_last_layers[-6], dict_last_layers[-5] = [-8], [-7]
        dict_last_layers[-8], dict_last_layers[-7] = [-14], [-9]
        dict_last_layers[-14], dict_last_layers[-9] = [-15], [-10, -25]
        dict_last_layers[-15], dict_last_layers[-10], dict_last_layers[-25] = [-16], [-11], [-26]
        dict_last_layers[-16], dict_last_layers[-11], dict_last_layers[-26] = [-17], [-12], [-27]
        dict_last_layers[-17], dict_last_layers[-12], dict_last_layers[-27] = [-18], [-13], [-28]
        dict_last_layers[-18], dict_last_layers[-13], dict_last_layers[-28] = [-19], [-14], [-29]
        dict_last_layers[-19], dict_last_layers[-14], dict_last_layers[-29] = [-20], [-15], [-30]
        dict_last_layers[-20], dict_last_layers[-30] = [-21], [-31]
        dict_last_layers[-21], dict_last_layers[-31] = [-22], [-32]
        dict_last_layers[-22], dict_last_layers[-32] = [-23], [-33]
        dict_last_layers[-23], dict_last_layers[-33] = [0], [1]

        dict_bottlenects[0] = -24
        dict_bottlenects[1] = -34

        return dict_last_layers, dict_bottlenects
# ----------------------------------------------------------------------------------------------------------------------
    def data_generator(self,list_store_bottlenecks,list_store_targets,validation_split=0.3,batch_size=32,is_train=True):

        G = list_store_bottlenecks[0].size
        G_val = int(G*validation_split)
        G_train = G - G_val
        g_train,g_val = 0,0
        while True:
            bottlenecks = [[] for i in range(len(list_store_bottlenecks))]
            targets     = [[] for i in range(len(list_store_targets))]

            for b in range(batch_size):
                if is_train ==True:
                    g = g_train
                else:
                    g = G_train + g_val

                for i in range(len(list_store_bottlenecks)):bottlenecks[i].append(list_store_bottlenecks[i].get(g))
                for i in range(len(list_store_targets))    :targets[i].append(list_store_targets[i].get(g))
                if is_train ==True:
                    g_train = (g_train + 1) % G_train
                else:
                    g_val = (g_val + 1) % G_val


            for i in range(len(bottlenecks)):bottlenecks[i]= numpy.array(bottlenecks[i])
            for i in range(len(targets))    :targets[i] = numpy.array(targets[i])
            yield [*bottlenecks, *targets], numpy.zeros((batch_size,1))
# ----------------------------------------------------------------------------------------------------------------------
    def __fit_loss_model(self,model_last_layers,placeholder_btlncs,bottlenecks,targets):

        n_epochs = 100
        lambda_object = Lambda(function=detector_YOLO3_core.yolo_loss, output_shape=(1,), name='yolo_loss',arguments={'anchors': self.anchors,'anchor_mask': self.anchor_mask, 'num_classes': len(self.class_names), 'ignore_thresh': 0.5})
        placeholder_yolo_features = [Input(shape=(13,13,3,5+len(self.class_names))),Input(shape=(2*13,2*13,3,5+len(self.class_names))) ]
        if len(model_last_layers.output)==3:placeholder_yolo_features.append(Input(shape=(52,52,3,5+len(self.class_names))))

        tensor_loss = lambda_object([*model_last_layers.output, *placeholder_yolo_features])

        model_loss = Model(inputs = [*placeholder_btlncs, *placeholder_yolo_features], outputs=tensor_loss)
        model_loss.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        early_stopping_monitor = EarlyStopping(monitor='loss', patience=10)
        print('Learning ...')
        model_loss.fit(x=[*bottlenecks,*targets],y=numpy.zeros((targets[0].shape[0],1)),validation_split=0.3,verbose=2,epochs=n_epochs,callbacks=[early_stopping_monitor])
        return
# ----------------------------------------------------------------------------------------------------------------------
    def __fit_loss_model_gen(self,model_last_layers,placeholder_btlncs,folder_in):

        n_epochs = 100
        lambda_object = Lambda(function=detector_YOLO3_core.yolo_loss, output_shape=(1,), name='yolo_loss',arguments={'anchors': self.anchors,'anchor_mask': self.anchor_mask, 'num_classes': len(self.class_names), 'ignore_thresh': 0.5})
        placeholder_yolo_features = [Input(shape=(13,13,3,5+len(self.class_names))),Input(shape=(2*13,2*13,3,5+len(self.class_names))) ]
        if len(model_last_layers.output)==3:placeholder_yolo_features.append(Input(shape=(52,52,3,5+len(self.class_names))))

        tensor_loss = lambda_object([*model_last_layers.output, *placeholder_yolo_features])

        model_loss = Model(inputs = [*placeholder_btlncs, *placeholder_yolo_features], outputs=tensor_loss)
        model_loss.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        early_stopping_monitor = EarlyStopping(monitor='loss', patience=10)


        store_target0 = tools_HDF5.HDF5_store(filename=folder_in+'target_0.hdf5')
        store_target1 = tools_HDF5.HDF5_store(filename=folder_in+'target_1.hdf5')
        if len(self.anchor_mask) > 2:
            store_target2 = tools_HDF5.HDF5_store(filename=folder_in + 'target_2.hdf5')

        store_bottlenecks0 = tools_HDF5.HDF5_store(filename=folder_in+'bottlenecks_0.hdf5')
        store_bottlenecks1 = tools_HDF5.HDF5_store(filename=folder_in+'bottlenecks_1.hdf5')

        if (store_target0.size<1000):
            bottlenecks = [store_bottlenecks0.get(),store_bottlenecks1.get()]
            targets = [store_target0.get(),store_target1.get()]
            if len(self.anchor_mask) > 2:targets.append(store_target2.get())
            print('Learn fit ...')
            model_loss.fit(x=[*bottlenecks,*targets],y=numpy.zeros((targets[0].shape[0],1)),validation_split=0.3,verbose=2,epochs=n_epochs,callbacks=[early_stopping_monitor])
        else:
            print('Learn fit_generator ...')
            generator_train = self.data_generator([store_bottlenecks0, store_bottlenecks1],[store_target0, store_target1], validation_split=0.3,batch_size=32,is_train=True)
            generator_val   = self.data_generator([store_bottlenecks0, store_bottlenecks1],[store_target0, store_target1], validation_split=0.3,batch_size=32,is_train=False)
            model_loss.fit_generator(generator_train,validation_data=generator_val, epochs=n_epochs,steps_per_epoch=8,validation_steps=1,verbose=2,callbacks=[early_stopping_monitor])

        return
# ----------------------------------------------------------------------------------------------------------------------
    def learn_tiny(self, file_annotations, folder_out, folder_annotation=None, limit=1000000):

        if folder_annotation is not None:
            foldername = folder_annotation
        else:
            foldername = '/'.join(file_annotations.split('/')[:-1]) + '/'
        start_time = time.time()

        with open(file_annotations) as f:lines = f.readlines()[1:limit]
        list_filenames = sorted(set([foldername+line.split(' ')[0] for line in lines]))
        true_boxes = tools_YOLO.get_true_boxes(foldername, file_annotations, delim =' ', smart_resized_target=(416, 416),limit=limit)
        detector_YOLO3_core.save_targets(folder_out, true_boxes, (416, 416), self.anchors, self.anchor_mask, len(self.class_names))

        if len(true_boxes)>6:
            self.anchors = tools_YOLO.annotation_boxes_to_ancors(true_boxes,len(self.anchors),delim=' ')
            self.boxes, self.scores, self.classes = detector_YOLO3_core.get_tensors_box_score_class(self.model.output,
                                                                                                    self.anchors,
                                                                                                    self.anchor_mask,
                                                                                                    len(self.class_names),
                                                                                                    self.input_tensor_shape,
                                                                                                    score_threshold=self.obj_threshold,
                                                                                                    iou_threshold=self.nms_threshold)

        dict_last_layers, dict_bottlenects = self.__last_layers4()

        placeholder_btlncs,last_layers_outs = self.construct_last_layers_placeholders(len(self.model.layers), dict_last_layers, dict_bottlenects)
        self.save_bottleneck_features(folder_out,list_filenames, dict_bottlenects)

        model_last_layers = Model(inputs=[*placeholder_btlncs], outputs=[*last_layers_outs])
        self.__fit_loss_model_gen(model_last_layers, placeholder_btlncs,folder_out)
        total_time = (time.time() - start_time)
        print('Learn: %s sec in total' % (total_time))

        self.logs.time_train = total_time
        self.logs.data_source = folder_annotation
        self.logs.last_layers = 4
        self.save_model(folder_out + 'A_model.h5')
        tools_YOLO.save_metadata(folder_out + 'A_metadata.txt', self.input_image_size, self.class_names, self.anchors, self.anchor_mask, self.obj_threshold, self.nms_threshold)
        self.__load_model(folder_out + 'A_model.h5', folder_out + 'A_metadata.txt', self.obj_threshold)
        return 0
# ----------------------------------------------------------------------------------------------------------------------

