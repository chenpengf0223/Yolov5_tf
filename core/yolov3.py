#! /usr/bin/env python
# coding=utf-8
import numpy as np
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg

import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


class YOLOV3(object):
    def __init__(self, input_data, trainable, freeze_pb=False):
        self.freeze_pb = freeze_pb
        self.net_type = cfg.YOLO.NET_TYPE
        self.trainable = trainable
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD

        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])


    def __build_nework(self, input_data):
        print('__build_nework self.net_type=', self.net_type)
        assert self.net_type in ['darknet53', 'mobilenetv2', 'mobilenetv3', 'mobilenetv3_small']

        if self.net_type == 'darknet53':
            route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)

            input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv52')
            input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
            input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
            input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
            input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')

            conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
            conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

            input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
            input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

            with tf.variable_scope('route_1'):
                input_data = tf.concat([input_data, route_2], axis=-1)

            input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
            input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
            input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
            input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
            input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

            conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512),  self.trainable, name='conv_mobj_branch' )
            conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
            input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

            with tf.variable_scope('route_2'):
                input_data = tf.concat([input_data, route_1], axis=-1)

            input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
            input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
            input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

            conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
            conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)
            return conv_lbbox, conv_mbbox, conv_sbbox

        
        elif self.net_type == 'mobilenetv2':
            route_1, route_2, input_data = backbone.mobilenetv2(input_data, self.trainable)

            input_data = common.convolutional(input_data, (1, 1, 320, 160), self.trainable, 'conv18')
            input_data = common.convolutional(input_data, (3, 3, 160, 320), self.trainable, 'conv19')
            input_data = common.convolutional(input_data, (1, 1, 320, 160), self.trainable, 'conv20')

            conv_lobj_branch = common.convolutional(input_data, (3, 3, 160, 320), self.trainable, name='conv_lobj_branch')
            conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 320, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)


            input_data = common.convolutional(input_data, (1, 1, 160, 80), self.trainable, 'conv21')
            input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

            with tf.variable_scope('route_1'):
                input_data = tf.concat([input_data, route_2], axis=-1)

            input_data = common.convolutional(input_data, (1, 1, 176, 80), self.trainable, 'conv22')
            input_data = common.convolutional(input_data, (3, 3, 80, 160), self.trainable, 'conv23')
            input_data = common.convolutional(input_data, (1, 1, 160, 80), self.trainable, 'conv24')

            conv_mobj_branch = common.convolutional(input_data, (3, 3, 80, 160),  self.trainable, name='conv_mobj_branch' )
            conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 160, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

            
            input_data = common.convolutional(input_data, (1, 1, 80, 40), self.trainable, 'conv25')
            input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

            with tf.variable_scope('route_2'):
                input_data = tf.concat([input_data, route_1], axis=-1)

            input_data = common.convolutional(input_data, (1, 1, 72, 40), self.trainable, 'conv26')
            input_data = common.convolutional(input_data, (3, 3, 40, 80), self.trainable, 'conv27')
            input_data = common.convolutional(input_data, (1, 1, 80, 40), self.trainable, 'conv28')

            conv_sobj_branch = common.convolutional(input_data, (3, 3, 40, 80), self.trainable, name='conv_sobj_branch')
            conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 80, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)
            return conv_lbbox, conv_mbbox, conv_sbbox


        elif self.net_type == 'mobilenetv3':
            route_1, route_2, input_data = backbone.mobilenetv3(input_data, self.trainable)

            input_data = common.convolutional(input_data, (1, 1, 160, 80), self.trainable, 'conv8')
            input_data = common.convolutional(input_data, (3, 3, 80, 160), self.trainable, 'conv9')
            input_data = common.convolutional(input_data, (1, 1, 160, 80), self.trainable, 'conv10')

            conv_lobj_branch = common.convolutional(input_data, (3, 3, 80, 160), self.trainable,
                                                    name='conv_lobj_branch')
            conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 160, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

            input_data = common.convolutional(input_data, (1, 1, 80, 40), self.trainable, 'conv11')
            input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

            with tf.variable_scope('route_1'):
                input_data = tf.concat([input_data, route_2], axis=-1)

            input_data = common.convolutional(input_data, (1, 1, 152, 40), self.trainable, 'conv12')
            input_data = common.convolutional(input_data, (3, 3, 40, 80), self.trainable, 'conv13')
            input_data = common.convolutional(input_data, (1, 1, 80, 40), self.trainable, 'conv14')

            conv_mobj_branch = common.convolutional(input_data, (3, 3, 40, 80), self.trainable,
                                                    name='conv_mobj_branch')
            conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 80, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

            input_data = common.convolutional(input_data, (1, 1, 40, 20), self.trainable, 'conv15')
            input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

            with tf.variable_scope('route_2'):
                input_data = tf.concat([input_data, route_1], axis=-1)

            input_data = common.convolutional(input_data, (1, 1, 60, 20), self.trainable, 'conv16')
            input_data = common.convolutional(input_data, (3, 3, 20, 40), self.trainable, 'conv17')
            input_data = common.convolutional(input_data, (1, 1, 40, 20), self.trainable, 'conv18')

            conv_sobj_branch = common.convolutional(input_data, (3, 3, 20, 40), self.trainable, name='conv_sobj_branch')
            conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 40, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)
            return conv_lbbox, conv_mbbox, conv_sbbox


        elif self.net_type == 'mobilenetv3_small':
            route_1, route_2, input_data = backbone.mobilenetv3_small(input_data, self.trainable)

            input_data = common.convolutional(input_data, (1, 1, 96, 80), self.trainable, 'conv7')
            input_data = common.convolutional(input_data, (3, 3, 80, 160), self.trainable, 'conv8')
            input_data = common.convolutional(input_data, (1, 1, 160, 80), self.trainable, 'conv9')

            conv_lobj_branch = common.convolutional(input_data, (3, 3, 80, 160), self.trainable,
                                                    name='conv_lobj_branch')
            conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 160, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

            input_data = common.convolutional(input_data, (1, 1, 80, 40), self.trainable, 'conv10')
            input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

            with tf.variable_scope('route_1'):
                input_data = tf.concat([input_data, route_2], axis=-1)

            input_data = common.convolutional(input_data, (1, 1, 88, 40), self.trainable, 'conv11')
            input_data = common.convolutional(input_data, (3, 3, 40, 80), self.trainable, 'conv12')
            input_data = common.convolutional(input_data, (1, 1, 80, 40), self.trainable, 'conv13')

            conv_mobj_branch = common.convolutional(input_data, (3, 3, 40, 80), self.trainable,
                                                    name='conv_mobj_branch')
            conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 80, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

            input_data = common.convolutional(input_data, (1, 1, 40, 20), self.trainable, 'conv14')
            input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

            with tf.variable_scope('route_2'):
                input_data = tf.concat([input_data, route_1], axis=-1)

            input_data = common.convolutional(input_data, (1, 1, 44, 20), self.trainable, 'conv15')
            input_data = common.convolutional(input_data, (3, 3, 20, 40), self.trainable, 'conv16')
            input_data = common.convolutional(input_data, (1, 1, 40, 20), self.trainable, 'conv17')

            conv_sobj_branch = common.convolutional(input_data, (3, 3, 20, 40), self.trainable, name='conv_sobj_branch')
            conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 40, 3 * (self.num_class + 5)),
                                              trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)
            return conv_lbbox, conv_mbbox, conv_sbbox

        else:
            print('self.net_type=%s error' % self.net_type)


    def decode(self, conv_output, anchors, stride):
        """return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
           contains (x, y, w, h, score, probability)"""
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        '''
        pre_dim = [batch_size, output_size, output_size]
        conv_raw_dxdy = tf.slice(conv_output, begin=[0,0,0,0], size=pre_dim + [anchor_per_scale*2])
        conv_raw_dwdh = tf.slice(conv_output, begin=[0,0,0,anchor_per_scale*2], size=pre_dim + [anchor_per_scale*2])
        conv_raw_conf = tf.slice(conv_output, begin=[0,0,0,anchor_per_scale*4], size=pre_dim + [anchor_per_scale])
        conv_raw_prob = tf.slice(conv_output, begin=[0,0,0,anchor_per_scale*5], size=pre_dim + [anchor_per_scale*self.num_class])
        print('sdsd', conv_raw_dwdh)
        input()
        if self.freeze_pb:
            conv_raw_dxdy = tf.reshape(conv_raw_dxdy, (output_size, output_size, anchor_per_scale, 2))
            conv_raw_dwdh = tf.reshape(conv_raw_dwdh, (output_size, output_size, anchor_per_scale, 2))
            conv_raw_conf = tf.reshape(conv_raw_conf, (output_size, output_size, anchor_per_scale, 1))
            conv_raw_prob = tf.reshape(conv_raw_prob, (output_size, output_size, anchor_per_scale, self.num_class))
        else:
            conv_raw_dxdy = tf.reshape(conv_raw_dxdy, (batch_size, output_size, output_size, anchor_per_scale, 2))
            conv_raw_dwdh = tf.reshape(conv_raw_dwdh, (batch_size, output_size, output_size, anchor_per_scale, 2))
            conv_raw_conf = tf.reshape(conv_raw_conf, (batch_size, output_size, output_size, anchor_per_scale, 1))
            conv_raw_prob = tf.reshape(conv_raw_prob, (batch_size, output_size, output_size, anchor_per_scale, self.num_class))
        '''
       
        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))
        conv_raw_dxdy = conv_output[:, :, :, :, 0:2] #strided_slice
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]


        #print(conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob)
        #input()
        #conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, [2, 2, 1, self.num_class], axis=4)
        # pre_dim = [batch_size, output_size, output_size, anchor_per_scale]
        # conv_raw_dxdy = tf.slice(conv_output, begin=[0,0,0,0,0], size=pre_dim + [2])
        # conv_raw_dwdh = tf.slice(conv_output, begin=[0,0,0,0,2], size=pre_dim + [2])
        # conv_raw_conf = tf.slice(conv_output, begin=[0,0,0,0,4], size=pre_dim + [1])
        # conv_raw_prob = tf.slice(conv_output, begin=[0,0,0,0,5], size=pre_dim + [self.num_class])

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)
        if self.freeze_pb:
            xy_grid = tf.reshape(xy_grid, (output_size, output_size, anchor_per_scale, 2))
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        print('anchors', anchors, stride)
        pred_wh = tf.exp(conv_raw_dwdh) * anchors * stride        
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss


    def bbox_giou(self, boxes1, boxes2):
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]), tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]), tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]

        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
        return giou


    def bbox_iou(self, boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        
        iou = 1.0 * inter_area / union_area
        return iou


    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size

        conv = tf.reshape(conv, (batch_size, output_size, output_size, self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)
        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf) + \
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf))
        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
        
        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))
        return giou_loss, conf_loss, prob_loss


    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors = self.anchors[0], stride = self.strides[0])
        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors = self.anchors[1], stride = self.strides[1])
        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.name_scope('iou_loss'):
            iou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]
        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]
        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]
        return iou_loss, conf_loss, prob_loss
