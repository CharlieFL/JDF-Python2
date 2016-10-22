'''CHANGES
1. Conv2dLayer attr "pad" is depricated. replaced with "pad"

'''

import os

import numpy as np
import pandas as p
import theano.tensor as T
import lasagne as nn
from lasagne.layers import dnn
from lasagne.nonlinearities import LeakyRectify

from layers import ApplyNonlinearity
from utils import (oversample_set,
                   get_img_ids_from_dir,
                   softmax,
                   split_data)
from losses import (log_loss,
                    accuracy_loss,
                    quad_kappa_loss,
                    quad_kappa_log_hybrid_loss,
                    quad_kappa_log_hybrid_loss_clipped)

# Main dir used to load files.
#base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
base_dir = 'F:/Research data/'


output_size = 512  # 120
batch_size = 64  # * 4
input_height, input_width = (output_size, output_size)
output_dim = 5
num_channels = 3

config_name = 'local_normal_' + str(output_size)

#model_name = ""

#prefix_train = '/media/user/Extended_ext4/train_ds2_crop/'
#prefix_test = '/media/user/Extended_ext4/test_ds2_crop/'

prefix_train = base_dir + 'train_ds5_crop'
prefix_test = base_dir + 'test_ds5_crop'

# (       image
#  level
#  0      25810
#  1       2443
#  2       5292
#  3        873
#  4        708,           image
#  level
#  0      0.734783
#  1      0.069550
#  2      0.150658
#  3      0.024853
#  4      0.020156)
chunk_size = 128  # * 2
num_chunks_train = 30000 // chunk_size * 200
validate_every = num_chunks_train // 50
output_every = num_chunks_train // 400
save_every = num_chunks_train // 200

buffer_size = 3
num_generators = 3

default_transfo_params = {'rotation': True, 'rotation_range': (0, 360),
                          'contrast': True, 'contrast_range': (0.7, 1.3),
                          'brightness': True, 'brightness_range': (0.7, 1.3),
                          'color': True, 'color_range': (0.7, 1.3),
                          'flip': True, 'flip_prob': 0.5,
                          'crop': True, 'crop_prob': 0.4,
                          'crop_w': 0.03, 'crop_h': 0.04,
                          'keep_aspect_ratio': False,
                          'resize_pad': False,
                          'zoom': True, 'zoom_prob': 0.5,
                          'zoom_range': (0.00, 0.05),
                          'paired_transfos': False,
                          'rotation_expand': False,
                          'crop_height': False,
                          'extra_width_crop': True,
                          'rotation_before_resize': False,
                          'crop_after_rotation': True}

no_transfo_params = {'keep_aspect_ratio':
                     default_transfo_params['keep_aspect_ratio'],
                     'resize_pad':
                         default_transfo_params['resize_pad'],
                     'extra_width_crop':
                         default_transfo_params['extra_width_crop'],
                     'rotation_before_resize':
                         default_transfo_params['rotation_before_resize'],
                     'crop_height':
                         default_transfo_params['crop_height'],
                     }

pixel_based_norm = False
paired_transfos = True

SEED = 1
sample_coefs = [0, 7, 3, 22, 25]
# [0, 7, 3, 22, 25] gives more even [0.25. 0.19. 0.20. 0.19. 0.18] distribution

# Change train dataset to oversample other labels.
# Total sizes:
# (       image
#  level
#  0      25810
#  1       2443
#  2       5292
#  3        873
#  4        708,           image
#  level
#  0      0.734783
#  1      0.069550
#  2      0.150658
#  3      0.024853
#  4      0.020156

switch_chunk = 60 * num_chunks_train // 100

leakiness = 0.5

obj_loss = 'kappalogclipped'
y_pow = 1

# Kappalog
log_scale = 0.50
log_offset = 0.50

# Kappalogclipped
log_cutoff = 0.80

lambda_reg = 0.0002

lr_scale = 6.00

LEARNING_RATE_SCHEDULE = {
    1: 0.0010 * lr_scale,
    num_chunks_train // 100 * 30: 0.00050 * lr_scale,
    num_chunks_train // 100 * 50: 0.00010 * lr_scale,
    num_chunks_train // 100 * 85: 0.00001 * lr_scale,
    num_chunks_train // 100 * 95: 0.000001 * lr_scale,
}

momentum = 0.90


def build_model():
    #model_name = "base model"
    
    layers = []

    l_in_imgdim = nn.layers.InputLayer(
        shape=(batch_size, 2),
        name='imgdim'
    )

    #layer 0
    l_in1 = nn.layers.InputLayer(
        shape=(batch_size, num_channels, input_width, input_height),
        name='images'
    )
    layers.append(l_in1)

    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
    DenseLayer = nn.layers.DenseLayer

    #layer 1
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(7, 7), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    #layer 2
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    #layer 3
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 4
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    #layer 5
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    
    #layer 6
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 7
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2),
                            name='coarse_last_pool')
    layers.append(l_pool)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(DenseLayer(layers[-1],
                             nonlinearity=None,
                             num_units=1024,
                             W=nn.init.Orthogonal(1.0),
                             b=nn.init.Constant(0.1),
                             name='first_fc_0'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_first_repr = layers[-1]

    l_coarse_repr = nn.layers.concat([l_first_repr,
                                      l_in_imgdim])
    layers.append(l_coarse_repr)

    # Combine representations of both eyes.
    layers.append(
        nn.layers.ReshapeLayer(layers[-1], shape=(batch_size // 2, -1)))

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(nn.layers.DenseLayer(layers[-1],
                                       nonlinearity=None,
                                       num_units=1024,
                                       W=nn.init.Orthogonal(1.0),
                                       b=nn.init.Constant(0.1),
                                       name='combine_repr_fc'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_hidden = nn.layers.DenseLayer(nn.layers.DropoutLayer(layers[-1], p=0.5),
                                    num_units=output_dim * 2,
                                    nonlinearity=None,  # No softmax yet!
                                    W=nn.init.Orthogonal(1.0),
                                    b=nn.init.Constant(0.1))
    layers.append(l_hidden)

    # Reshape back to 5.
    layers.append(nn.layers.ReshapeLayer(layers[-1],
                                         shape=(batch_size, 5)))

    # Apply softmax.
    l_out = ApplyNonlinearity(layers[-1],
                              nonlinearity=nn.nonlinearities.softmax)
    layers.append(l_out)

    l_ins = [l_in1, l_in_imgdim]

    return l_out, l_ins

def build_LUCERO_model_1():
    '''
   Number of trainable parameters: 10066986
   0  InputLayer                   (64, 3, 512, 512)  
   1  Conv2DDNNLayer               (64, 32, 256, 256)  5 //2
   2  MaxPool2DDNNLayer            (64, 32, 127, 127)  3 //2
   3  Conv2DDNNLayer               (64, 32, 127, 127)  3 //1
   4  Conv2DDNNLayer               (64, 32, 127, 127)  3 //1
   5  MaxPool2DDNNLayer            (64, 32, 63, 63)  3 //2
   6  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
   7  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
   8  MaxPool2DDNNLayer            (64, 64, 31, 31)  3 //2
   9  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  10  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  11  MaxPool2DDNNLayer            (64, 128, 15, 15)  3 //2
  12  Conv2DDNNLayer               (64, 128, 15, 15)  3 //1
  13  Conv2DDNNLayer               (64, 128, 15, 15)  3 //1
  14  MaxPool2DDNNLayer            (64, 128, 7, 7)  3 //2
  15  Conv2DDNNLayer               (64, 256, 7, 7)  3 //1
  16  Conv2DDNNLayer               (64, 256, 7, 7)  3 //1
  17  Conv2DDNNLayer               (64, 256, 7, 7)  3 //1
  18  Conv2DDNNLayer               (64, 256, 7, 7)  3 //1
  19  MaxPool2DDNNLayer            (64, 256, 3, 3)  3 //2
  20  DropoutLayer                 (64, 256, 3, 3)   [0.50]
  21  DenseLayer                   (64, 1024)  
  22  FeaturePoolLayer             (64, 512)  2 //
  23  InputLayer                   (64, 2)  
  24  ConcatLayer                  (64, 514)  
  25  ReshapeLayer                 (32, 1028)  
  26  DropoutLayer                 (32, 1028)   [0.50]
  27  DenseLayer                   (32, 1024)  
  28  FeaturePoolLayer             (32, 512)  2 //
  29  DropoutLayer                 (32, 512)   [0.50]
  30  DenseLayer                   (32, 10)  
  31  ReshapeLayer                 (64, 5)  
  32  ApplyNonlinearity            (64, 5)  
    Inserted Max pool at layer 10-11
    '''
    
    #model_name = "L1"
    
    layers = []

    l_in_imgdim = nn.layers.InputLayer(
        shape=(batch_size, 2),
        name='imgdim'
    )

    #layer 0
    l_in1 = nn.layers.InputLayer(
        shape=(batch_size, num_channels, input_width, input_height),
        name='images'
    )
    layers.append(l_in1)

    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
    DenseLayer = nn.layers.DenseLayer

    #layer 1
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(5, 5), stride=(2,2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    #layer 2
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    #layer 3
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 4
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    #layer 5
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    
    #layer 6
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 7
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    #layer 8
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    #layer 9
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 10
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2),
                            name='coarse_last_pool')
    layers.append(l_pool)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(DenseLayer(layers[-1],
                             nonlinearity=None,
                             num_units=1024,
                             W=nn.init.Orthogonal(1.0),
                             b=nn.init.Constant(0.1),
                             name='first_fc_0'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_first_repr = layers[-1]

    l_coarse_repr = nn.layers.concat([l_first_repr,
                                      l_in_imgdim])
    layers.append(l_coarse_repr)

    # Combine representations of both eyes.
    layers.append(
        nn.layers.ReshapeLayer(layers[-1], shape=(batch_size // 2, -1)))

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(nn.layers.DenseLayer(layers[-1],
                                       nonlinearity=None,
                                       num_units=1024,
                                       W=nn.init.Orthogonal(1.0),
                                       b=nn.init.Constant(0.1),
                                       name='combine_repr_fc'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_hidden = nn.layers.DenseLayer(nn.layers.DropoutLayer(layers[-1], p=0.5),
                                    num_units=output_dim * 2,
                                    nonlinearity=None,  # No softmax yet!
                                    W=nn.init.Orthogonal(1.0),
                                    b=nn.init.Constant(0.1))
    layers.append(l_hidden)

    # Reshape back to 5.
    layers.append(nn.layers.ReshapeLayer(layers[-1],
                                         shape=(batch_size, 5)))

    # Apply softmax.
    l_out = ApplyNonlinearity(layers[-1],
                              nonlinearity=nn.nonlinearities.softmax)
    layers.append(l_out)

    l_ins = [l_in1, l_in_imgdim]

    return l_out, l_ins

def build_LUCERO_model_2(): #9mill params
    '''
   0  InputLayer                   (64, 3, 512, 512)
   1  Conv2DDNNLayer               (64, 32, 256, 256)  5 //2
   2  MaxPool2DDNNLayer            (64, 32, 127, 127)  3 //2
   3  Conv2DDNNLayer               (64, 32, 127, 127)  3 //1
   4  Conv2DDNNLayer               (64, 32, 127, 127)  3 //1
   5  MaxPool2DDNNLayer            (64, 32, 63, 63)  3 //2
   6  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
   7  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
   8  MaxPool2DDNNLayer            (64, 64, 31, 31)  3 //2
   9  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  10  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  11  MaxPool2DDNNLayer            (64, 128, 15, 15)  3 //2
  12  Conv2DDNNLayer               (64, 256, 15, 15)  3 //1
  13  Conv2DDNNLayer               (64, 256, 15, 15)  3 //1
  14  MaxPool2DDNNLayer            (64, 256, 7, 7)  3 //2
  15  Conv2DDNNLayer               (64, 256, 7, 7)  3 //1
  16  MaxPool2DDNNLayer            (64, 256, 3, 3)  3 //2
  17  DropoutLayer                 (64, 256, 3, 3)   [0.50]
  18  DenseLayer                   (64, 1024)
  19  FeaturePoolLayer             (64, 512)  2 //
  20  InputLayer                   (64, 2)
  21  ConcatLayer                  (64, 514)
  22  ReshapeLayer                 (32, 1028)
  23  DropoutLayer                 (32, 1028)   [0.50]
  24  DenseLayer                   (32, 1024)
  25  FeaturePoolLayer             (32, 512)  2 //
  26  DropoutLayer                 (32, 512)   [0.50]
  27  DenseLayer                   (32, 10)
  28  ReshapeLayer                 (64, 5)
  29  ApplyNonlinearity            (64, 5)


                Number of trainable parameters: 9202218

    '''
   #model_name = "L2"
    
    layers = []

    l_in_imgdim = nn.layers.InputLayer(
        shape=(batch_size, 2),
        name='imgdim'
    )

    #layer 0
    l_in1 = nn.layers.InputLayer(
        shape=(batch_size, num_channels, input_width, input_height),
        name='images'
    )
    layers.append(l_in1)

    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
    DenseLayer = nn.layers.DenseLayer

    #layer 1
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(5, 5), stride=(2,2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    #layer 2
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    #layer 3
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 4
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    #layer 5
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    
    #layer 6
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 7
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    #layer 8
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    #layer 9
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 10
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)
# 


    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    
    #added
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)



    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2),
                            name='coarse_last_pool')
    layers.append(l_pool)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(DenseLayer(layers[-1],
                             nonlinearity=None,
                             num_units=1024,
                             W=nn.init.Orthogonal(1.0),
                             b=nn.init.Constant(0.1),
                             name='first_fc_0'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_first_repr = layers[-1]

    l_coarse_repr = nn.layers.concat([l_first_repr,
                                      l_in_imgdim])
    layers.append(l_coarse_repr)

    # Combine representations of both eyes.
    layers.append(
        nn.layers.ReshapeLayer(layers[-1], shape=(batch_size // 2, -1)))

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(nn.layers.DenseLayer(layers[-1],
                                       nonlinearity=None,
                                       num_units=1024,
                                       W=nn.init.Orthogonal(1.0),
                                       b=nn.init.Constant(0.1),
                                       name='combine_repr_fc'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_hidden = nn.layers.DenseLayer(nn.layers.DropoutLayer(layers[-1], p=0.5),
                                    num_units=output_dim * 2,
                                    nonlinearity=None,  # No softmax yet!
                                    W=nn.init.Orthogonal(1.0),
                                    b=nn.init.Constant(0.1))
    layers.append(l_hidden)

    # Reshape back to 5.
    layers.append(nn.layers.ReshapeLayer(layers[-1],
                                         shape=(batch_size, 5)))

    # Apply softmax.
    l_out = ApplyNonlinearity(layers[-1],
                              nonlinearity=nn.nonlinearities.softmax)
    layers.append(l_out)

    l_ins = [l_in1, l_in_imgdim]

    return l_out, l_ins

def build_LUCERO_model_3():
    #leakiness changed from .5 to .3
    leakiness=0.3
    '''
     0  InputLayer                   (64, 3, 512, 512)  
   1  Conv2DDNNLayer               (64, 32, 256, 256)  5 //2
   2  Conv2DDNNLayer               (64, 32, 128, 128)  5 //2
   3  Conv2DDNNLayer               (64, 32, 128, 128)  3 //1
   4  Conv2DDNNLayer               (64, 32, 128, 128)  3 //1
   5  MaxPool2DDNNLayer            (64, 32, 63, 63)  3 //2
   6  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
   7  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
   8  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
   9  MaxPool2DDNNLayer            (64, 64, 31, 31)  3 //2
  10  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  11  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  12  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  13  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  14  MaxPool2DDNNLayer            (64, 128, 15, 15)  3 //2
  15  Conv2DDNNLayer               (64, 256, 15, 15)  3 //1
  16  Conv2DDNNLayer               (64, 256, 15, 15)  3 //1
  17  Conv2DDNNLayer               (64, 256, 15, 15)  3 //1
  18  Conv2DDNNLayer               (64, 256, 15, 15)  3 //1
  19  MaxPool2DDNNLayer            (64, 256, 7, 7)  3 //2
  20  DropoutLayer                 (64, 256, 7, 7)   [0.50]
  21  DenseLayer                   (64, 1024)  
  22  FeaturePoolLayer             (64, 512)  2 //
  23  InputLayer                   (64, 2)  
  24  ConcatLayer                  (64, 514)  
  25  ReshapeLayer                 (32, 1028)  
  26  DropoutLayer                 (32, 1028)   [0.50]
  27  DenseLayer                   (32, 1024)  
  28  FeaturePoolLayer             (32, 512)  2 //
  29  DropoutLayer                 (32, 512)   [0.50]
  30  DenseLayer                   (32, 10)  
  31  ReshapeLayer                 (64, 5)  
  32  ApplyNonlinearity            (64, 5)  


        Number of trainable parameters: 21778474

    '''
    
    #model_name = "L3"
    
    layers = []

    l_in_imgdim = nn.layers.InputLayer(
        shape=(batch_size, 2),
        name='imgdim'
    )

    #layer 0
    l_in1 = nn.layers.InputLayer(
        shape=(batch_size, num_channels, input_width, input_height),
        name='images'
    )
    layers.append(l_in1)

    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
    DenseLayer = nn.layers.DenseLayer

    #layer 1
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(5, 5), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(5, 5), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    
    

    #layer 3
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 4
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    


    #layer 5
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    
    #layer 6
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 7
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)



    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2),
                            name='coarse_last_pool')
    layers.append(l_pool)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(DenseLayer(layers[-1],
                             nonlinearity=None,
                             num_units=1024,
                             W=nn.init.Orthogonal(1.0),
                             b=nn.init.Constant(0.1),
                             name='first_fc_0'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_first_repr = layers[-1]

    l_coarse_repr = nn.layers.concat([l_first_repr,
                                      l_in_imgdim])
    layers.append(l_coarse_repr)

    # Combine representations of both eyes.
    layers.append(
        nn.layers.ReshapeLayer(layers[-1], shape=(batch_size // 2, -1)))

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(nn.layers.DenseLayer(layers[-1],
                                       nonlinearity=None,
                                       num_units=1024,
                                       W=nn.init.Orthogonal(1.0),
                                       b=nn.init.Constant(0.1),
                                       name='combine_repr_fc'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_hidden = nn.layers.DenseLayer(nn.layers.DropoutLayer(layers[-1], p=0.5),
                                    num_units=output_dim * 2,
                                    nonlinearity=None,  # No softmax yet!
                                    W=nn.init.Orthogonal(1.0),
                                    b=nn.init.Constant(0.1))
    layers.append(l_hidden)

    # Reshape back to 5.
    layers.append(nn.layers.ReshapeLayer(layers[-1],
                                         shape=(batch_size, 5)))

    # Apply softmax.
    l_out = ApplyNonlinearity(layers[-1],
                              nonlinearity=nn.nonlinearities.softmax)
    layers.append(l_out)

    l_ins = [l_in1, l_in_imgdim]

    return l_out, l_ins


def build_LUCERO_model_4(): #8.7mill 128 chunk, 64 batch 
    '''
    Layer 1 changed filter from 
    Inserted Max pool at layer 10-11
    '''
    
    #model_name = "L4" 
    layers = []

    l_in_imgdim = nn.layers.InputLayer(
        shape=(batch_size, 2),
        name='imgdim'
    )

    #layer 0
    l_in1 = nn.layers.InputLayer(
        shape=(batch_size, num_channels, input_width, input_height),
        name='images'
    )
    layers.append(l_in1)

    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
    DenseLayer = nn.layers.DenseLayer

    #layer 1
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(7, 7), stride=(2,2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    #layer 2
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    #layer 3
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 5
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    
    #layer 6
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=96, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    #layer 8
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    #layer 9
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2),
                            name='coarse_last_pool')
    layers.append(l_pool)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(DenseLayer(layers[-1],
                             nonlinearity=None,
                             num_units=1024,
                             W=nn.init.Orthogonal(1.0),
                             b=nn.init.Constant(0.1),
                             name='first_fc_0'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_first_repr = layers[-1]

    l_coarse_repr = nn.layers.concat([l_first_repr,
                                      l_in_imgdim])
    layers.append(l_coarse_repr)

    # Combine representations of both eyes.
    layers.append(
        nn.layers.ReshapeLayer(layers[-1], shape=((batch_size) // 2, -1)))

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(nn.layers.DenseLayer(layers[-1],
                                       nonlinearity=None,
                                       num_units=1024,
                                       W=nn.init.Orthogonal(1.0),
                                       b=nn.init.Constant(0.1),
                                       name='combine_repr_fc'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_hidden = nn.layers.DenseLayer(nn.layers.DropoutLayer(layers[-1], p=0.5),
                                    num_units=output_dim * 2,
                                    nonlinearity=None,  # No softmax yet!
                                    W=nn.init.Orthogonal(1.0),
                                    b=nn.init.Constant(0.1))
    layers.append(l_hidden)

    # Reshape back to 5.
    layers.append(nn.layers.ReshapeLayer(layers[-1],
                                         shape=(batch_size, 5)))

    # Apply softmax.
    l_out = ApplyNonlinearity(layers[-1],
                              nonlinearity=nn.nonlinearities.softmax)
    layers.append(l_out)

    l_ins = [l_in1, l_in_imgdim]

    return l_out, l_ins

def build_LUCERO_model_5(): #35.3mill
    '''
   0  InputLayer                   (64, 3, 512, 512)  
   1  Conv2DDNNLayer               (64, 32, 256, 256)  7 //2
   2  Conv2DDNNLayer               (64, 32, 128, 128)  3 //2
   3  Conv2DDNNLayer               (64, 64, 128, 128)  3 //1
   4  MaxPool2DDNNLayer            (64, 64, 63, 63)  3 //2
   5  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
   6  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
   7  MaxPool2DDNNLayer            (64, 64, 31, 31)  3 //2
   8  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
   9  Conv2DDNNLayer               (64, 256, 31, 31)  3 //1
  10  Conv2DDNNLayer               (64, 256, 31, 31)  3 //1
  11  MaxPool2DDNNLayer            (64, 256, 15, 15)  3 //2
  12  DropoutLayer                 (64, 256, 15, 15)   [0.50]
  13  DenseLayer                   (64, 512)  
  14  FeaturePoolLayer             (64, 256)  2 //
  15  InputLayer                   (64, 2)  
  16  ConcatLayer                  (64, 258)  
  17  ReshapeLayer                 (32, 516)  
  18  DropoutLayer                 (32, 516)   [0.50]
  19  DenseLayer                   (32, 10)  
  20  ReshapeLayer                 (64, 5)  
  21  ApplyNonlinearity            (64, 5)  
    '''
    
    #model_name = "L5"
    layers = []

    l_in_imgdim = nn.layers.InputLayer(
        shape=(batch_size, 2),
        name='imgdim'
    )

    #layer 0
    l_in1 = nn.layers.InputLayer(
        shape=(batch_size, num_channels, input_width, input_height),
        name='images'
    )
    layers.append(l_in1)

    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
    DenseLayer = nn.layers.DenseLayer

    #layer 1
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(7, 7), stride=(2,2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    #layer 3
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    
    #layer 6
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 5
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)


    #layer 9
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2),
                            name='coarse_last_pool')
    layers.append(l_pool)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(DenseLayer(layers[-1],
                             nonlinearity=None,
                             num_units=512,
                             W=nn.init.Orthogonal(1.0),
                             b=nn.init.Constant(0.1),
                             name='first_fc_0'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_first_repr = layers[-1]

    l_coarse_repr = nn.layers.concat([l_first_repr,
                                      l_in_imgdim])
    layers.append(l_coarse_repr)

    # Combine representations of both eyes.
    layers.append(
        nn.layers.ReshapeLayer(layers[-1], shape=((batch_size) // 2, -1)))


    l_hidden = nn.layers.DenseLayer(nn.layers.DropoutLayer(layers[-1], p=0.5),
                                    num_units=output_dim * 2,
                                    nonlinearity=None,  # No softmax yet!
                                    W=nn.init.Orthogonal(1.0),
                                    b=nn.init.Constant(0.1))
    layers.append(l_hidden)

    # Reshape back to 5.
    layers.append(nn.layers.ReshapeLayer(layers[-1],
                                         shape=(batch_size, 5)))

    # Apply softmax.
    l_out = ApplyNonlinearity(layers[-1],
                              nonlinearity=nn.nonlinearities.softmax)
    layers.append(l_out)

    l_ins = [l_in1, l_in_imgdim]

    return l_out, l_ins


def build_LUCERO_model_6():
    '''
   0  InputLayer                   (64, 3, 512, 512)  
   1  Conv2DDNNLayer               (64, 32, 256, 256)  7 //2
   2  MaxPool2DDNNLayer            (64, 32, 127, 127)  3 //2
   3  Conv2DDNNLayer               (64, 32, 127, 127)  3 //1
   4  DropoutLayer                 (64, 32, 127, 127)   [0.40]
   5  Conv2DDNNLayer               (64, 32, 127, 127)  3 //1
   6  MaxPool2DDNNLayer            (64, 32, 63, 63)  3 //2
   7  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
   8  DropoutLayer                 (64, 64, 63, 63)   [0.40]
   9  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
  10  DropoutLayer                 (64, 64, 63, 63)   [0.40]
  11  Conv2DDNNLayer               (64, 64, 63, 63)  3 //1
  12  MaxPool2DDNNLayer            (64, 64, 31, 31)  3 //2
  13  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  14  DropoutLayer                 (64, 128, 31, 31)   [0.40]
  15  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  16  DropoutLayer                 (64, 128, 31, 31)   [0.40]
  17  Conv2DDNNLayer               (64, 128, 31, 31)  3 //1
  18  MaxPool2DDNNLayer            (64, 128, 15, 15)  3 //2
  19  Conv2DDNNLayer               (64, 128, 15, 15)  3 //1
  20  DropoutLayer                 (64, 128, 15, 15)   [0.40]
  21  Conv2DDNNLayer               (64, 128, 15, 15)  3 //1
  22  DropoutLayer                 (64, 128, 15, 15)   [0.40]
  23  Conv2DDNNLayer               (64, 128, 15, 15)  3 //1
  24  DropoutLayer                 (64, 128, 15, 15)   [0.40]
  25  Conv2DDNNLayer               (64, 128, 15, 15)  3 //1
  26  MaxPool2DDNNLayer            (64, 128, 7, 7)  3 //2
  27  Conv2DDNNLayer               (64, 256, 7, 7)  3 //1
  28  DropoutLayer                 (64, 256, 7, 7)   [0.40]
  29  Conv2DDNNLayer               (64, 256, 7, 7)  3 //1
  30  DropoutLayer                 (64, 256, 7, 7)   [0.40]
  31  Conv2DDNNLayer               (64, 256, 7, 7)  3 //1
  32  DropoutLayer                 (64, 256, 7, 7)   [0.40]
  33  Conv2DDNNLayer               (64, 256, 7, 7)  3 //1
  34  DropoutLayer                 (64, 256, 7, 7)   [0.40]
  35  Conv2DDNNLayer               (64, 256, 7, 7)  3 //1
  36  MaxPool2DDNNLayer            (64, 256, 3, 3)  3 //2
  37  DropoutLayer                 (64, 256, 3, 3)   [0.50]
  38  DenseLayer                   (64, 1024)  
  39  FeaturePoolLayer             (64, 512)  2 //
  40  InputLayer                   (64, 2)  
  41  ConcatLayer                  (64, 514)  
  42  ReshapeLayer                 (32, 1028)  
  43  DropoutLayer                 (32, 1028)   [0.50]
  44  DenseLayer                   (32, 1024)  
  45  FeaturePoolLayer             (32, 512)  2 //
  46  DropoutLayer                 (32, 512)   [0.50]
  47  DenseLayer                   (32, 10)  
  48  ReshapeLayer                 (64, 5)  
  49  ApplyNonlinearity            (64, 5)  


        Number of trainable parameters: 11585514
    '''
    
    #model_name = "L6"
    layers = []

    l_in_imgdim = nn.layers.InputLayer(
        shape=(batch_size, 2),
        name='imgdim'
    )

    #layer 0
    l_in1 = nn.layers.InputLayer(
        shape=(batch_size, num_channels, input_width, input_height),
        name='images'
    )
    layers.append(l_in1)

    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
    DenseLayer = nn.layers.DenseLayer

    #C1
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(7, 7), stride=(2,2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
        
    #layer 2
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    #C2-3
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))
    
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    #layer 5
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)
    
    ##C4-6
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))

    #layer 7
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 8
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    ##C7-9
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))

    #layer 10
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
 
    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))
    
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)    

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    
    #C10-13
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)    

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)
    
    #C14-18
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.4))
    
    l_conv = Conv2DLayer(layers[-1],
                     num_filters=256, filter_size=(3, 3), stride=(1, 1),
                     pad='same',
                     nonlinearity=LeakyRectify(leakiness),
                     W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                     untie_biases=True)
    layers.append(l_conv)
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2),
                            name='coarse_last_pool')
    layers.append(l_pool)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(DenseLayer(layers[-1],
                             nonlinearity=None,
                             num_units=1024,
                             W=nn.init.Orthogonal(1.0),
                             b=nn.init.Constant(0.1),
                             name='first_fc_0'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_first_repr = layers[-1]

    l_coarse_repr = nn.layers.concat([l_first_repr,
                                      l_in_imgdim])
    layers.append(l_coarse_repr)

    # Combine representations of both eyes.
    layers.append(
        nn.layers.ReshapeLayer(layers[-1], shape=(batch_size // 2, -1)))

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(nn.layers.DenseLayer(layers[-1],
                                       nonlinearity=None,
                                       num_units=1024,
                                       W=nn.init.Orthogonal(1.0),
                                       b=nn.init.Constant(0.1),
                                       name='combine_repr_fc'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_hidden = nn.layers.DenseLayer(nn.layers.DropoutLayer(layers[-1], p=0.5),
                                    num_units=output_dim * 2,
                                    nonlinearity=None,  # No softmax yet!
                                    W=nn.init.Orthogonal(1.0),
                                    b=nn.init.Constant(0.1))
    layers.append(l_hidden)

    # Reshape back to 5.
    layers.append(nn.layers.ReshapeLayer(layers[-1],
                                         shape=(batch_size, 5)))

    # Apply softmax.
    l_out = ApplyNonlinearity(layers[-1],
                              nonlinearity=nn.nonlinearities.softmax)
    layers.append(l_out)

    l_ins = [l_in1, l_in_imgdim]

    return l_out, l_ins

def build_LUCERO_model_7():
    
    #model_name = "L7"
    layers = []

    l_in_imgdim = nn.layers.InputLayer(
        shape=(batch_size, 2),
        name='imgdim'
    )

    #layer 0
    l_in1 = nn.layers.InputLayer(
        shape=(batch_size, num_channels, input_width, input_height),
        name='images'
    )
    layers.append(l_in1)

    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
    DenseLayer = nn.layers.DenseLayer

    #layer 1
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(7, 7), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    
    #layer 2
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    #layer 3
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 4
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    #layer 5
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    
    #layer 6
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    #layer 7
    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)


    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2),
                            name='coarse_last_pool')
    layers.append(l_pool)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(DenseLayer(layers[-1],
                             nonlinearity=None,
                             num_units=1024,
                             W=nn.init.Orthogonal(1.0),
                             b=nn.init.Constant(0.1),
                             name='first_fc_0'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_first_repr = layers[-1]

    l_coarse_repr = nn.layers.concat([l_first_repr,
                                      l_in_imgdim])
    layers.append(l_coarse_repr)

    # Combine representations of both eyes.
    layers.append(
        nn.layers.ReshapeLayer(layers[-1], shape=(batch_size // 2, -1)))

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(nn.layers.DenseLayer(layers[-1],
                                       nonlinearity=None,
                                       num_units=1024,
                                       W=nn.init.Orthogonal(1.0),
                                       b=nn.init.Constant(0.1),
                                       name='combine_repr_fc'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_hidden = nn.layers.DenseLayer(nn.layers.DropoutLayer(layers[-1], p=0.5),
                                    num_units=output_dim * 2,
                                    nonlinearity=None,  # No softmax yet!
                                    W=nn.init.Orthogonal(1.0),
                                    b=nn.init.Constant(0.1))
    layers.append(l_hidden)

    # Reshape back to 5.
    layers.append(nn.layers.ReshapeLayer(layers[-1],
                                         shape=(batch_size, 5)))

    # Apply softmax.
    l_out = ApplyNonlinearity(layers[-1],
                              nonlinearity=nn.nonlinearities.softmax)
    layers.append(l_out)

    l_ins = [l_in1, l_in_imgdim]

    return l_out, l_ins




config_name += '_' + obj_loss

if obj_loss == 'kappalog':
    config_name += '_logscale_' + str(log_scale)
    config_name += '_logoffset_' + str(log_offset)
elif 'kappalogclipped' in obj_loss:
    config_name += '_logcutoff_' + str(log_cutoff)

config_name += '_reg_' + str(lambda_reg)

if obj_loss == 'log':
    loss_function = log_loss
elif obj_loss == 'acc':
    loss_function = accuracy_loss
elif obj_loss == 'kappa':
    def loss(y, t):
        return quad_kappa_loss(y, t,
                               y_pow=y_pow)
    loss_function = loss
elif obj_loss == 'kappalog':
    def loss(y, t):
        return quad_kappa_log_hybrid_loss(y, t,
                                          y_pow=y_pow,
                                          log_scale=log_scale,
                                          log_offset=log_offset)
    loss_function = loss
elif obj_loss == 'kappalogclipped':
    def loss(y, t):
        return quad_kappa_log_hybrid_loss_clipped(y, t,
                                                  y_pow=y_pow,
                                                  log_cutoff=log_cutoff)
    loss_function = loss
else:
    raise ValueError("Need obj_loss param.")

#removed 4-5-16 due to deprecation of Objective
# def build_objective(l_out, loss_function=loss_function,
#                     lambda_reg=lambda_reg):
#     params = nn.layers.get_all_params(l_out, regularizable=True)
# 
#     reg_term = sum(T.sum(p ** 2) for p in params)
# 
#     def loss(y, t):
#         return loss_function(y, t) + lambda_reg * reg_term
# 
#     return nn.objectives.Objective(l_out, loss_function=loss)


train_labels = p.read_csv(os.path.join(base_dir, 'trainLabels.csv'))
labels_split = p.DataFrame(list(train_labels.image.str.split('_')),
                           columns=['id', 'eye'])
labels_split['level'] = train_labels.level
labels_split['id'] = labels_split['id'].astype('int')

id_train, y_train, id_valid, y_valid = split_data(train_labels, labels_split,
                                                  valid_size=10,
                                                  SEED=SEED, pairs=True)

# Change train dataset to oversample other labels.
# Total sizes:
# (       image
#  level
#  0      25810
#  1       2443
#  2       5292
#  3        873
#  4        708,           image
#  level
#  0      0.734783
#  1      0.069550
#  2      0.150658
#  3      0.024853
#  4      0.020156)

pl_enabled = True #False
pl_softmax_temp = 2
pl_train_coef = 5

pl_train_fn = ''
pl_test_fn = ''

pl_log = False

if pl_enabled:
    pl_test_fn = '2015_07_14_072437_6_log_mean.npy'

    test_preds = np.load(os.path.join(base_dir, 'preds/' + pl_test_fn))

    if test_preds.shape[1] > 5:
        test_preds = test_preds[:, -5:].astype('float32')

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    print "Orig test preds:\n\n"

    print test_preds[:10], '\n'

    if np.mean(test_preds) > 0:
        # These are not log probs, so can do log.
        test_preds = np.log(1e-5 + test_preds)

    test_probs = softmax(test_preds, temp=pl_softmax_temp)

    # Double ids so only every other.
    images_test_pl = sorted(set(get_img_ids_from_dir(prefix_test)))
    labels_test_pl = test_probs.reshape((-1, 2, 5))

    print "\nImages for test:\n\n"
    print images_test_pl[:5], '\n'

    print "\nLabels for test:\n\n"
    print labels_test_pl[:5], '\n'

    # Add only test PL for now.
    id_train_oversample, labels_train_oversample = oversample_set(id_train,
                                                                  y_train,
                                                                  sample_coefs)

    # First train set.
    images_train_0 = list(id_train_oversample) + images_test_pl
    labels_train_pl = np.eye(5)[
        list(labels_train_oversample.flatten().astype('int32'))
    ].reshape((-1, 2, 5))

    labels_train_0 = np.vstack([labels_train_pl,
                                labels_test_pl]).astype('float32')

    # Second train set.
    images_train_1 = list(id_train) * pl_train_coef + images_test_pl
    labels_train_pl = np.eye(5)[
        list(y_train.flatten().astype('int32')) * pl_train_coef
    ].reshape((-1, 2, 5))

    labels_train_1 = np.vstack([labels_train_pl,
                                labels_test_pl]).astype('float32')

    images_train_eval = id_train[:]
    labels_train_eval = y_train[:].astype('int32')
    images_valid_eval = id_valid[:]
    labels_valid_eval = y_valid[:].astype('int32')
else:
    #oversample training/labels based on coefficients
    id_train_oversample, labels_train_oversample = oversample_set(id_train,
                                                                  y_train,
                                                                  sample_coefs)

    images_train_0 = id_train_oversample
    labels_train_0 = labels_train_oversample.astype('int32')
    images_train_1, labels_train_1 = id_train, y_train.astype('int32')

    images_train_eval = id_train[:]
    labels_train_eval = y_train[:].astype('int32')
    images_valid_eval = id_valid[:]
    labels_valid_eval = y_valid[:].astype('int32')
