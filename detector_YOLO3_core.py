# ----------------------------------------------------------------------------------------------------------------------
from functools import wraps
import numpy
import cv2
import os
from os import listdir
import fnmatch
import io
from collections import defaultdict
import configparser
# ----------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D,Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from functools import reduce
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
# ----------------------------------------------------------------------------------------------------------------------
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)
# ----------------------------------------------------------------------------------------------------------------------
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    cnv2d = DarknetConv2D(*args, **no_bias_kwargs)
    res = compose(cnv2d, BatchNormalization(), LeakyReLU(alpha=0.1))
    return res
# ----------------------------------------------------------------------------------------------------------------------
def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x
# ----------------------------------------------------------------------------------------------------------------------
def unique_config_sections(config_file):
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream
# ----------------------------------------------------------------------------------------------------------------------
def darknet_to_keras(filename_config, filename_weights, filename_output):

    config_path = filename_config
    weights_path = filename_weights
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)
    assert weights_path.endswith('.weights'), '{} is not a .weights file'.format(weights_path)

    output_path = filename_output
    assert output_path.endswith('.h5'), 'output path {} is not a .h5 file'.format(output_path)
    output_root = os.path.splitext(output_path)[0]

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = numpy.ndarray(
        shape=(3,), dtype='int32', buffer=weights_file.read(12))
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        seen = numpy.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = numpy.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    print('Creating Keras model.')
    input_layer = Input(shape=(None, None, 3))
    prev_layer = input_layer
    all_layers = []

    weight_decay = float(cfg_parser['net_0']['decay']
                         ) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    out_index = []
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)

            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = numpy.product(weights_shape)

            print('conv2d', 'bn'
            if batch_normalize else '  ', activation, weights_shape)

            conv_bias = numpy.ndarray(
                shape=(filters,),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            if batch_normalize:
                bn_weights = numpy.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = numpy.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = numpy.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            # Create Conv2D layer
            if stride > 1:
                # Darknet uses left and top padding instead of 'same' mode
                prev_layer = ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)
            conv_layer = (Conv2D(
                filters, (size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                activation=act_fn,
                padding=padding))(prev_layer)

            if batch_normalize:
                conv_layer = (BatchNormalization(
                    weights=bn_weight_list))(conv_layer)
            prev_layer = conv_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                print('Concatenating route layers:', layers)
                concatenate_layer = Concatenate()(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported.'
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2, 'Only stride=2 supported.'
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('yolo'):
            out_index.append(len(all_layers) - 1)
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            pass

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    if len(out_index) == 0: out_index.append(len(all_layers) - 1)
    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    print(model.summary())

    #model.save_weights('{}'.format(output_path))
    #print('Saved Keras weights to {}'.format(output_path))

    model.save('{}'.format(output_path))
    print('Saved Keras model to {}'.format(output_path))

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count +
                                                       remaining_weights))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))

    return
# ----------------------------------------------------------------------------------------------------------------------
def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x
# ----------------------------------------------------------------------------------------------------------------------
def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y
# ----------------------------------------------------------------------------------------------------------------------
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),[1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),[grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs
# ----------------------------------------------------------------------------------------------------------------------
def yolo_body_tiny(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])
# ----------------------------------------------------------------------------------------------------------------------
def yolo_body_full(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    dn_body = darknet_body(inputs)

    darknet = Model(inputs, dn_body)
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(DarknetConv2D_BN_Leaky(256, (1,1)),UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(DarknetConv2D_BN_Leaky(128, (1,1)),UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])
# ----------------------------------------------------------------------------------------------------------------------
def assign_weights(model_source,model_target):

    if len(model_source.layers)!=len(model_target.layers):
        return

    for i,layer_in in enumerate(model_source.layers):
        layer_out = model_target.layers[i]
        weights_in = layer_in.get_weights()
        weights_out = layer_out.get_weights()
        if len(weights_in) == len(weights_out):
            flag = [w1.shape==w2.shape for w1,w2 in zip(weights_in,weights_out)]
            if sum(flag) == len(flag):
                layer_out.set_weights(weights_in)

        #layer_out = model_target.layers[i]
        #for j, weight_in in enumerate(layer_in.non_trainable_weights): layer_out.non_trainable_weights[j] = weight_in
        #for j, weight_in in enumerate(layer_in.trainable_weights): layer_out.trainable_weights[j] = weight_in
        #for j, weight_in in enumerate(layer_in.weights):layer_out.weights[j]=weight_in

    return
# ----------------------------------------------------------------------------------------------------------------------
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes
# ----------------------------------------------------------------------------------------------------------------------
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores
# ----------------------------------------------------------------------------------------------------------------------
def get_tensors_box_score_class(yolo_outputs,anchors,anchor_mask,num_classes,image_shape,max_boxes=20,score_threshold=.6,iou_threshold=.5):

    num_layers = len(yolo_outputs)

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes, box_scores = [],[]
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):

        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
# ----------------------------------------------------------------------------------------------------------------------
def get_tensors_box_score_class_unfilter(yolo_outputs,anchors,anchor_mask,num_classes,image_shape,max_boxes=20,score_threshold=.6,iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes, box_scores = [],[]
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):

        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        #nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        #class_boxes = K.gather(class_boxes, nms_index)
        #class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
# ----------------------------------------------------------------------------------------------------------------------
def preprocess_true_boxes(true_boxes, input_shape, anchors, anchor_mask,num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting



    true_boxes = numpy.array(true_boxes, dtype='float32')
    input_shape = numpy.array(input_shape, dtype='int32')
    centers_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    wh_xy = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = centers_xy/input_shape[::-1]
    true_boxes[..., 2:4] = wh_xy/input_shape[::-1]

    m = true_boxes.shape[0]     #normaly m=1
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [numpy.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes), dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = numpy.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = wh_xy[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = wh_xy[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = numpy.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = numpy.maximum(box_mins, anchor_mins)
        intersect_maxes = numpy.minimum(box_maxes, anchor_maxes)
        intersect_wh = numpy.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)


        best_anchor = numpy.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = numpy.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = numpy.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true
# ----------------------------------------------------------------------------------------------------------------------
def preprocess_true_boxes_wrapper(list_of_boxes, input_shape, anchors,anchor_mask, num_classes):
    y_true=[[],[],[]]

    for k,boxes in enumerate(list_of_boxes):
        if k%100==0:print('.',end='')
        true_boxes = numpy.array([boxes])
        y = preprocess_true_boxes(true_boxes, input_shape, anchors,anchor_mask, num_classes)
        for i,each in enumerate(y):
            y_true[i].append(each)

    print()

    y_true[0] = numpy.concatenate(y_true[0],axis=0)
    y_true[1] = numpy.concatenate(y_true[1], axis=0)
    if len(y_true[2])!=0:
        numpy.concatenate(y_true[2], axis=0)
    else:
        y_true = y_true[:2]

    return y_true
# ----------------------------------------------------------------------------------------------------------------------
def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou
# ----------------------------------------------------------------------------------------------------------------------
def yolo_loss(args, anchors, anchor_mask,num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]

    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss =   object_mask  * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
                         (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)* ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
# ----------------------------------------------------------------------------------------------------------------------
def generate_colors(N=80):
    numpy.random.seed(42)
    colors = []
    for i in range(0, N):
        color = cv2.cvtColor(
            numpy.array([int(255 * numpy.random.rand()), 255, 225], dtype=numpy.uint8).reshape(1, 1, 3),
            cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(color[0]), int(color[1]), int(color[2])))
    numpy.random.seed(None)

    return colors
# ----------------------------------------------------------------------------------------------------------------------
def draw_objects_on_image(image, boxes_bound, scores, classes, colors, class_names):

    if boxes_bound is None:
        return image

    for box, score, cl in zip(boxes_bound, scores, classes):
        top, left, bottom,right = box

        top     = max(0, numpy.floor(top + 0.5).astype('int32'))
        left    = max(0, numpy.floor(left + 0.5).astype('int32'))
        bottom  = min(image.shape[0], numpy.floor(bottom + 0.5).astype('int32'))
        right   = min(image.shape[1], numpy.floor(right + 0.5).astype('int32'))

        color = colors[cl]

        cv2.rectangle(image, (left,top), (right,bottom), color, 2)
        position = top - 6 if top - 6 > 10 else top + 26
        cv2.putText(image, '{0} {1:.2f}'.format(class_names[cl], score), (left+4, position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return image
# ----------------------------------------------------------------------------------------------------------------------
def get_true_boxes(foldername, filename, delim='\t', smart_resized_target=None,limit=100):

    with open(filename) as f:lines = f.readlines()[1:limit]
    list_filenames = [line.split(' ')[0] for line in lines]
    filenames_dict = sorted(set(list_filenames))

    true_boxes = []

    for filename in filenames_dict:
        local_boxes = []
        for line in lines:
            split = line.split(delim)
            if split[0]==filename:
                class_ID = int(split[5])
                x_min, y_min, x_max, y_max = numpy.array(split[1:5]).astype(numpy.float)

                if smart_resized_target is not None:
                    image = cv2.imread(foldername + filename)
                    x_min, y_min = tools_image.smart_resize_point(x_min, y_min, image.shape[1], image.shape[0],smart_resized_target[1], smart_resized_target[0])
                    x_max, y_max = tools_image.smart_resize_point(x_max, y_max, image.shape[1], image.shape[0],smart_resized_target[1], smart_resized_target[0])

                local_boxes.append([x_min, y_min, x_max, y_max, class_ID])

        true_boxes.append(local_boxes)

    return true_boxes
# ----------------------------------------------------------------------------------------------------------------------
def draw_annotation_boxes(file_annotations, path_out,colors,class_names,delim=' '):

    foldername = '/'.join(file_annotations.split('/')[:-1]) + '/'

    with open(file_annotations) as f:lines = f.readlines()[1:]
    box_xyxy  = numpy.array([line.split(delim)[1:5] for line in lines],dtype=numpy.int)
    filenames = numpy.array([line.split(delim)[0] for line in lines])
    class_IDs = numpy.array([line.split(delim)[5] for line in lines],dtype=numpy.int)

    for filename in set(filenames):
        image = cv2.imread(foldername + filename)
        idx = numpy.where(filenames == filename)
        for box,class_ID in zip(box_xyxy[idx],class_IDs[idx]):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[class_ID],thickness=2)

        cv2.imwrite(path_out + filename, image)
# ----------------------------------------------------------------------------------------------------------------------
def filter_dups(boxes,classes,scores,nms_threshold):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = numpy.where(classes == c)
        b, c, s = boxes[inds], classes[inds], scores[inds]
        keep = nms_boxes(b, s, nms_threshold)
        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return [], [], []

    nboxes = numpy.concatenate(nboxes)
    nclasses = numpy.concatenate(nclasses)
    nscores = numpy.concatenate(nscores)

    #keep = nms_boxes(nboxes, nscores, nms_threshold)
    #nboxes=nboxes[keep]
    #nclasses=nclasses[keep]
    #nscores=nscores[keep]

    return nboxes,nclasses,nscores
# ----------------------------------------------------------------------------------------------------------------------
def nms_boxes(boxes, scores,nms_threshold):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = numpy.maximum(x[i], x[order[1:]])
        yy1 = numpy.maximum(y[i], y[order[1:]])
        xx2 = numpy.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = numpy.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = numpy.maximum(0.0, xx2 - xx1 + 1)
        h1 = numpy.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = numpy.where(ovr <= nms_threshold)[0]
        order = order[inds + 1]

    keep = numpy.array(keep)

    return keep
# ----------------------------------------------------------------------------------------------------------------------
