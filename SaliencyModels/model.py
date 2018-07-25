from __future__ import print_function, division, unicode_literals
import numpy as np
import sys

sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/SaliencyModels/')
sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/IttiKoch/')
sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/')
sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/OliverPlayground/')

#import bethgeflow as bf
import pickle as pickle
import random
import argparse
import tensorflow as tf
slim = tf.contrib.slim
from utils_oliver import repeat, my_gather, avgpool2d, set_up_dir
from VGGFeatures import vgg_19_conv, extract_feature_outputs, initialize_deep_gaze, relu, gauss_blur, extract_feature_outputs_simple
from SteerableCNNTrainWeights import SteerableCNN
from vgg_structure import VGG
#from MLCFeatures import MLC_tf#, prepare_tf_image
from MLCFeatures1D  import MLC_tf
from GaborFilters import get_gabor_tf
from IttiKoch import IttiKoch

def normalize_0_1(src):
    M = tf.reduce_max(src,reduction_indices=[1,2], keep_dims=True)
       
    # Normalize to [0,1] range
    Min = tf.reduce_min(src,reduction_indices=[1,2], keep_dims=True)
    #src = (src-Min)/(M-Min)
    
    denom = M-Min
    #(x, mean, variance, offset, scale, variance_epsilon, name=None)
    src = tf.nn.batch_normalization(src, Min, denom**2, offset = None, scale= None ,variance_epsilon=tf.constant(1e-6,tf.float32))
    return src

def normalize_minus_plus_1(src):
    M = tf.reduce_max(src,reduction_indices=[1,2], keep_dims=True)
    Min = tf.reduce_min(src,reduction_indices=[1,2], keep_dims=True)
    denom = M-Min

    #(x, mean, variance, offset, scale, variance_epsilon, name=None)
    src = tf.nn.batch_normalization(src, Min, denom**2, offset = -1., scale= 2. ,variance_epsilon=tf.constant(1e-6,tf.float32))
    return src


def resize_upconv2d(X, n_ch_in, n_ch_out, kernel_size, strides,name = 'W'):
    """Resizes then applies a convolution.
    :param X
        Input tensor
    :param n_ch_in
        Number of input channels
    :param n_ch_out
        Number of output channels
    :param kernel_size
        Size of square shaped convolutional kernel
    :param strides
        Stride information
    Source: https://github.com/ghwatson/faststyle/blob/master/im_transf_net.py
    To use:
    
    with tf.variable_scope('upsample_0'):
        h = relu(inst_norm(resize_upconv2d(h, 64, 32, 3, [1, 2, 2, 1])))
    """
    shape = [kernel_size, kernel_size, n_ch_in, n_ch_out]

    # We first upsample two strides-worths. The convolution will then bring it
    # down one stride.
    new_h = X.get_shape().as_list()[1]*strides[1]**2
    new_w = X.get_shape().as_list()[2]*strides[2]**2
    upsized = tf.image.resize_images(X, [new_h, new_w], method=1)

    # Now convolve to get the channels to what we want.
    shape = [kernel_size, kernel_size, n_ch_in, n_ch_out]
    try:
        with tf.variable_scope('upsample', reuse = True):
            W = tf.get_variable(name=name,
                                shape=shape,
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer() )
    except ValueError:
        with tf.variable_scope('upsample'):
            W = tf.get_variable(name=name,
                                shape=shape,
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer() )
    h = tf.nn.conv2d(upsized,
                     filter=W,
                     strides=strides,
                     padding="SAME")

    return h


def log_sum_exp(A, axis=None, sum_op=tf.reduce_sum, eps=0):
    with tf.name_scope('log_sum_exp'):
        A_max = tf.reduce_max(A, reduction_indices=axis, keep_dims=True)
        B = tf.add(tf.log(sum_op(tf.exp(A - A_max), reduction_indices=axis, keep_dims=True) + eps), A_max)

        if axis is None:
            return tf.squeeze(B)  # collapse to scalar
        else:
            if not hasattr(axis, '__iter__'): axis = [axis]
            return tf.squeeze(B, squeeze_dims=axis)  # drop summed axes

def readout_network(inputs, hidden_units, scope='readout_network'):
    assert isinstance(hidden_units, list)
    assert hidden_units[-1] == 1
    
    with tf.variable_scope(scope, 'readout_network', inputs) as sc:
        end_points_collection = sc.name + '_end_points'
        debug = {}
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, relu, slim.bias_add, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            _nets = []
            for k, input in enumerate(inputs): # Loop over selected layers/heights/sigmas
                #up_scaled_input = repeat(input, repeats=8)
                up_scaled_input = input
                _net = slim.conv2d(up_scaled_input, hidden_units[0], [1, 1], scope='conv1_part{}'.format(k), activation_fn=None,
                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(False), biases_initializer=None)
                debug['conv1_part{}'.format(k)] = _net

                                   # weights_initializer=tf.truncated_normal_initializer(stddev=0.01), biases_initializer=None)

                _nets.append(_net)
            net = sum(_nets)
            
            for k, n_features in enumerate(hidden_units[1:]):
                net = slim.bias_add(net, scope='conv{}'.format(k+1),
                                    )
                net = relu(net, scope='relu{}'.format(k+1))
                
                net = slim.conv2d(net, n_features, [1, 1], scope='conv{}'.format(k+2),
                                  weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(False),activation_fn=None, biases_initializer=None)

                                  #  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),activation_fn=None, biases_initializer=None)
   
        # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points, debug
#vgg_19.default_image_size = 448#224

def image_log_sum_exp(inputs, scope='image_log_sum_exp', squeeze_dims=True):
    with tf.name_scope(scope, 'image_log_sum_exp', [inputs]) as sc:
        outputs = inputs
        outputs = log_sum_exp(outputs, axis=1)
        outputs = log_sum_exp(outputs, axis=1)

        if not squeeze_dims:
            outputs = tf.expand_dims(outputs, 1)
            outputs = tf.expand_dims(outputs, 1)
        
        return outputs

def saliency_map_construction(readout, centerbias, scope='saliency_map'):
    with tf.variable_scope(scope, 'saliency_map', [readout, centerbias]) as sc:
        end_points_collection = sc.name + '_end_points'
        
        alpha = tf.get_variable('alpha', initializer=tf.ones(()))
        sigma = tf.get_variable('sigma', initializer=5.0*tf.ones(()))
        
        blur = gauss_blur(readout, sigma, windowradius=20, scope='blur')
        
        expanded_centerbias = tf.expand_dims(centerbias, 3)
        with_centerbias = tf.add(blur, alpha*expanded_centerbias, name='with_centerbias')
        #net = slim.utils.collect_named_outputs(end_points_collection, sc, net)
        
        log_density = with_centerbias - image_log_sum_exp(with_centerbias, squeeze_dims=False)
        log_density_without_centerbias = blur - image_log_sum_exp(blur, squeeze_dims=False)
        #net = slim.utils.collect_named_outputs(end_points_collection, sc, net)
        
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        end_points[sc.name+'/blur'] = blur
        end_points[sc.name+'/blur_with_centerbias'] = with_centerbias
        end_points[sc.name+'/log_density'] = log_density
        end_points[sc.name+'/log_density_without_centerbias'] = log_density_without_centerbias
        return log_density, end_points

    
def log_likelihoods(log_density, x_inds, y_inds, b_inds=None):
    if b_inds is None:
        b_inds = tf.zeros_like(x_inds)
        c_inds = tf.zeros_like(x_inds)
        return my_gather(log_density, [b_inds, y_inds, x_inds, c_inds])
    else:
        c_inds = tf.zeros_like(x_inds)
        return my_gather(log_density, [b_inds, y_inds, x_inds, c_inds])



def construct_deep_gaze_loss(
        dg_endpoints, hidden_layers, x_inds=None, y_inds=None, b_inds=None,
        l2_penalty=0.0,
        l1_penalty=0.0,
        gabor_penalty = 0.1,
        feature_regularizations = None,
        space_regularizations = None, args = None,
        deep_gaze_scope='readout_network',
        scope='deep_gaze_loss',
        ):
    
    params = {v.name: v for v in tf.all_variables()}
    readout_layers = []
    
    k = 1
    
    end_points = {

    }
    
    while '{}/conv{}'.format(deep_gaze_scope, k) in dg_endpoints:
        readout_layers.append(dg_endpoints['{}/conv{}'.format(deep_gaze_scope, k)])
        k += 1
    
    readout_layer1_weights = [params[v] for v in params if v.startswith('{}/conv1_part'.format(deep_gaze_scope))]
    ####

    print(readout_layer1_weights)
    print([v.name for v in readout_layer1_weights])
   # readout_layer1_weights = tf.concat(1, readout_layer1_weights)
   #
    #other_weights = [params[v] for v in params
    #                 if v.startswith('{}/conv'.format(deep_gaze_scope))
    #                 and not 'part' in v and not 'biases' in v]
   # 
    #readout_weights = sorted([readout_layer1_weights] + other_weights, key=lambda v: v.name)
    #print('Readout weights', [v.name for v in readout_weights])
    ####
        
    if feature_regularizations is None:
        feature_regularizations = 0.0

    if not isinstance(feature_regularizations, list):
        feature_regularizations = [feature_regularizations for l in hidden_layers]
    
    if space_regularizations is None:
        space_regularizations = 0.0

    if not isinstance(space_regularizations, list):
        space_regularizations = [space_regularizations for l in hidden_layers]
    
    def add_cost(name, value):
        end_points['costs/{}'.format(name)] = value
        end_points['costs/loss'] += value
    
    with tf.name_scope(scope, 'deep_gaze_loss') as sc:
        
        if x_inds is None:
            x_inds = tf.placeholder(tf.int32, name='x_inds')

        if y_inds is None:
            y_inds = tf.placeholder(tf.int32, name='y_inds')

        binds = None
        if b_inds is None:
            b_inds = tf.placeholder(tf.int32, name='b_inds')
            
        
        end_points['{}/x_inds'.format(scope)] = x_inds
        end_points['{}/y_inds'.format(scope)] = y_inds
        end_points['{}/b_inds'.format(scope)] = b_inds
        
    
        end_points['costs/loss'] = 0.0
        end_points['costs/log_likelihoods'] = log_likelihoods(dg_endpoints['saliency_map/log_density'.format(deep_gaze_scope)], x_inds, y_inds, b_inds)
        end_points['costs/loss'] += end_points['costs/log_likelihoods']
        
        
        if args.gabor_loss == True:
            print('USING GABOR LOSS')
            current_weights = slim.get_variables('SteerablePyramid/W_orients')
            print('CURR_WEIGHT', [v.name for v in current_weights])
            # get_gabor_tf(lmbd,theta,phi,sig,gam, n_bands
            lmbd =  slim.get_variables('SteerablePyramid/gabor/lambda')
            theta =  slim.get_variables('SteerablePyramid/gabor/theta')
            phi =  slim.get_variables('SteerablePyramid/gabor/phi')
            sig =  slim.get_variables('SteerablePyramid/gabor/sigma')
            gam =  slim.get_variables('SteerablePyramid/gabor/gamma')

            #with tf.Session() as sess:
            #    sess.run(tf.initialize_all_variables())
            #    print(sess.run(tf.squeeze(gam)[0]))
            #raise
            gabor_parms = get_gabor_tf(lmbd,theta,phi,sig,gam, args.n_bands)
            end_points.update(gabor_parms)
            gabor_parms_list = [v for k,v in gabor_parms.items() if k]
            print(gabor_parms.keys())
            cost = tf.reduce_mean(tf.squared_difference(tf.squeeze(current_weights), gabor_parms_list))
            
            add_cost('gabor_loss', -gabor_penalty*cost) # Cost adding or substracting?
    
        if l1_penalty:
            weights = readout_weights[-1]
            l1_norm = tf.reduce_sum(tf.abs(weights)) / tf.sqrt(tf.reduce_sum(tf.square(weights)))
            
            add_cost('l1_penalty', l1_penalty*l1_norm)
        
        assert not l2_penalty
        
        assert not any(feature_regularizations)
        
        assert not any(space_regularizations)
        
    return end_points

def flat2gen(alist):
    for item in alist:
        if isinstance(item, list):
            for subitem in item: yield subitem
        else:
            yield item



def collect_features(select_features, data, upsample, downsample_factor =2):
    'Select data and put in a feature Tensor'
    
                
   # def collect_layer(layer,layer_data, select_features):
   #     layer_list = list(flat2gen([v for k,v in layer_data.iteritems() if k in select_features]))
   #     if downsample_factor >= 2:
   #         if upsample == 'repeat':
   #             layer_upsampled = [repeat(d, downsample_factor**layer) for d in layer_list]
   #         elif upsample == 'resize':
   #             with tf.variable_scope('upsample'):
   #                 layer_upsampled = [repeat(d, downsample_factor**layer) for d in layer_list]#
   #
   #                 h = resize_upconv2d(d, 1, 1, 3, [1,  downsample_factor**layer,  downsample_factor**layer, 1])
   #             
   #     else:
   #        layer_upsampled = layer_list
   #             
   #     return layer_upsampled
    
    if upsample == 'repeat':
        try:
            layer_list = [tf.concat(3,[repeat(data[k], downsample_factor**l) for k in keys])  for l, keys in select_features.items()]
        except TypeError:
            layer_list = [tf.concat([repeat(data[k], downsample_factor**l) for k in keys],3)  for l, keys in select_features.items()]

        
    elif upsample == 'resize':
            try:
                layer_list = [tf.concat(3,[ resize_upconv2d(data[k], 3, 3, 3, [1,  downsample_factor**l,  downsample_factor**l, 1]) for k in keys])  for l, keys in select_features.items()]
            except TypeError:
                layer_list = [tf.concat([ resize_upconv2d(data[k], 3, 3, 3, [1,  downsample_factor**l,  downsample_factor**l, 1]) for k in keys],3)  for l, keys in select_features.items()]
        
        
        
    # return tf.pack(list(flat2gen(layer_list)))
    # pprint(list(flat2gen(layer_list)))
    
    return list(flat2gen(layer_list))



def collect_features_mlc(data, n_sigmas = 1, n_maps = 6):
    print(tf.shape(data))
    
    features = []
    for i in range(n_sigmas):
            features.append(data[:,:,:,i,:])

    return features
    
    
    
def merge_up_scaled_features(mlc_features, steer_features,n_sigmas = 1,n_maps = 6):
    mlc = []
    for i in range(n_sigmas):
        for j in range(n_maps):
            print('k',mlc_features[i])
            mlc.append(mlc_features[i][:,:,:,j])
    print('a',np.shape(mlc))
    try:
        mlc =tf.pack(mlc,3)
    except AttributeError:
        mlc =tf.stack(mlc,3)
    return list(flat2gen([mlc, steer_features]))



#########################################################################################################################

class Model(object):
    def __init__(self, args,height,width, training = True):
        
        self.flag = args.flag
        

        self.height = height
        self.width = width
        
        self.height_ds = int(self.height/args.im_ds)
        self.width_ds = int(self.width/args.im_ds)

        
        
        # Check if downsampling requested
        if args.ds == 1:
            self.downsampling = False
        else:
            self.downsampling = True
        
        # Prepare placeholders
        #self.input_tensor = tf.placeholder('float', shape=(args.bs, 768, 1024, args.channels))
        #if args.flag == 'vgg':
        
        
        #self.input_tensor = tf.placeholder('float', shape=(args.bs,  self.height, self.width, args.channels))
        
        self.input_tensor = tf.placeholder('float', shape=(None,  None, None, args.channels))

        #else: 
        #    self.input_tensor = tf.placeholder('float', shape=(args.bs,  self.height_ds, self.width_ds, args.channels))

        
        #self.input_tensor = repeat(self.input_tensor,2)

            
        
        self.x_inds = tf.placeholder(tf.int32)
        self.y_inds = tf.placeholder(tf.int32)
        
        #Possibly get it from the BaselineModel? 
        #self.centerbias = tf.placeholder('float', shape=(args.bs, self.height_ds, self.width_ds))
        self.centerbias = tf.placeholder('float', shape=(None,None, None))

        
        # Determine model type
        
        if self.flag == 'steer':
            select_features = pickle.load(open(args.steer_dict, 'rb'), encoding ='latin1')
            levels = select_features.keys()
            
            filter_size, n_filters, n_bands, height =  15, 1, args.n_bands, 5
            A = SteerableCNN(filter_size, n_filters, n_bands, height, ds = self.downsampling, im_ds = args.im_ds, filter_type = args.filter_type, gabor_dir = args.gabor_dir)
            #Make SP an attribute:
            self.A = A
            data =  A.steerable_network_for_multiple_features(self.input_tensor) 
            self.data = data
            self.filters = A.filters
            
            self.up_scaled_inputs = collect_features(select_features, data, upsample = args.upsample, downsample_factor=args.ds)
            # print('After', [a.get_shape().as_list() for a in self.up_scaled_inputs])
            # raise
        elif self.flag == 'ittikoch':
            A = IttiKoch(feature_type=args.ik_features)
            data = A.build(self.input_tensor)

            if args.ik_features == 'ik_features':
                tmp = data['readout_features']
                layers = ['c_2_s_5' ,'c_2_s_6','c_3_s_6', 'c_3_s_7', 'c_4_s_7' ,'c_4_s_8']
                features = {l:[] for l in layers}
                _net = {}
                _net.update(tmp['color'])
                _net.update(tmp['orientation'])
                _net.update(tmp['intensity'])

                for key,val in _net.items():
                    features['_'.join(key.split('_')[1:5])].append(val)

                try:
                    self.up_scaled_inputs = [tf.concat(3,v) for k,v in features.items()] 
                except TypeError:
                    self.up_scaled_inputs = [tf.concat(v,3) for k,v in features.items()] 
             
            elif args.ik_features == 'raw_features':
                tmp = data['raw_readout_features']
                self.up_scaled_inputs = [tf.concat(3,tmp['intensity']),
                                         tf.concat(3,list(flat2gen(tmp['color']))),
                                         tf.concat(3,list(flat2gen(tmp['orientation'])))
                                         ]
                
                print('N_FEATURES',len(self.up_scaled_inputs))
                
            elif args.ik_features == 'conspicuity':
                tmp = data['conspicuity']
                self.up_scaled_inputs = [tf.concat(3,list(tmp['intensity'].values())),
                                         tf.concat(3,list(flat2gen(list(tmp['color'].values())))),
                                         tf.concat(3,list(flat2gen(list(tmp['orientation'].values()))))
                                        ]

            else:
                raise Error
                
        elif self.flag == 'ittikoch_mlc':
            
            A = IttiKoch(feature_type=args.ik_features)
            data = A.build(self.input_tensor)

            if args.ik_features == 'ik_features':
                tmp = data['readout_features']
                layers = ['c_2_s_5' ,'c_2_s_6','c_3_s_6', 'c_3_s_7', 'c_4_s_7' ,'c_4_s_8']
                features = {l:[] for l in layers}
                _net = {}
                _net.update(tmp['color'])
                _net.update(tmp['orientation'])
                _net.update(tmp['intensity'])

                for key,val in _net.items():
                    features['_'.join(key.split('_')[1:5])].append(val)

                try:
                    self.up_scaled_inputs_ik = [tf.concat(3,v) for k,v in features.items()] 
                except TypeError:
                    self.up_scaled_inputs_ik = [tf.concat(v,3) for k,v in features.items()] 

             
            elif args.ik_features == 'raw_features':
                tmp = data['raw_readout_features']
                self.up_scaled_inputs_ik = [tf.concat(3,tmp['intensity']),
                                         tf.concat(3,list(flat2gen(tmp['color']))),
                                         tf.concat(3,list(flat2gen(tmp['orientation'])))
                                         ]
                
                
            elif args.ik_features == 'conspicuity':
                tmp = data['conspicuity']
                self.up_scaled_inputs_ik = [tf.concat(3,list(tmp['intensity'].values())),
                                         tf.concat(3,list(flat2gen(list(tmp['color'].values())))),
                                         tf.concat(3,list(flat2gen(list(tmp['orientation'].values()))))
                                        ]
            else:
                raise Error

                
            mlc = MLC_tf((args.bs,self.height_ds,self.width_ds,3), color = True, downsample=1, sigmas=args.sigmas, im_ds=args.im_ds)

            net_mlc = mlc.build_input_features_color(self.input_tensor)

            self.up_scaled_inputs_mlc = collect_features_mlc(net_mlc['features'],n_sigmas=len(args.sigmas))

            self.up_scaled_inputs  = self.up_scaled_inputs_mlc + self.up_scaled_inputs_ik
                

        elif self.flag == 'vgg':

            #vgg = VGG((1,3,self.height,self.width))
            #print(vgg)
            net, end_points = vgg_19_conv(self.input_tensor,'deep_gaze/feature_network_0')

            #self.up_scaled_inputs_old, _ = extract_feature_outputs( end_points, ['conv5_1', 'relu5_1', 'relu5_2', 'conv5_3', 'relu5_4'],
            #                        'conv1_1', vgg , praefix='deep_gaze/feature_network_0/')
            #                     'conv2_1', vgg , praefix='deep_gaze/feature_network_0/')
            
            self.up_scaled_inputs, _ = extract_feature_outputs_simple(end_points, ['conv5_1', 'relu5_1', 'relu5_2', 'conv5_3', 'relu5_4'], 'conv2_1' , praefix='deep_gaze/feature_network_0/')
            

        elif self.flag == 'mlc':
            mlc = MLC_tf((args.bs,self.height_ds,self.width_ds,3), color = True, downsample=1, sigmas=args.sigmas, im_ds=args.im_ds)

            net_mlc = mlc.build_input_features_color(self.input_tensor)

            self.up_scaled_inputs = collect_features_mlc(net_mlc['features'],n_sigmas=len(args.sigmas))
            
            
            if False:
                # Only used to match number of parameters in gabor model
                print('Before',[a.get_shape().as_list() for a in self.up_scaled_inputs])
                #Repeated mlc to match number of parameters in gabor model
                # Repeat every [None,None,None,6] mlc feature tensor 5 times to yield [None,None,None,30]
                hilf = []
                for tmp_var in self.up_scaled_inputs:
                    tmp = [tmp_var for i in range(5)]
                    hilf.append(tf.concat(3,tmp))

                self.up_scaled_inputs  = hilf 


        elif self.flag == 'steer_mlc':
            print('Using: Steerable Pyramid and MLC')
            #Steerable features
            select_features = pickle.load(open(args.steer_dict, 'rb'), encoding ='latin1')
            levels = select_features.keys()
            filter_size, n_filters, n_bands, height =  15, 1, args.n_bands, 5
            #filter_size, n_filters, n_bands, height =  15, 1, 8, 5
            A = SteerableCNN(filter_size, n_filters, n_bands, height, ds = self.downsampling, im_ds = args.im_ds,filter_type = args.filter_type, gabor_dir = args.gabor_dir)
            self.A = A
            data =  A.steerable_network_for_multiple_features(self.input_tensor)    
            self.filters = A.filters

            self.up_scaled_inputs_steer = collect_features(select_features, data, upsample = args.upsample, downsample_factor=args.ds)

            #MLC features
            mlc = MLC_tf((args.bs,self.height_ds,self.width_ds,3), color = True, downsample=1, sigmas=args.sigmas, im_ds=args.im_ds)

            net_mlc = mlc.build_input_features_color(self.input_tensor)

            self.up_scaled_inputs_mlc = collect_features_mlc(net_mlc['features'], n_sigmas=len(args.sigmas))

            
            self.up_scaled_inputs = merge_up_scaled_features(self.up_scaled_inputs_mlc,self.up_scaled_inputs_steer, n_sigmas=len(args.sigmas)) 

            #self.up_scaled_inputs = list(flat2gen(list(self.up_scaled_inputs_mlc) + list(self.up_scaled_inputs_steer)))

           # self.up_scaled_inputs = list(self.up_scaled_inputs_mlc) + list(self.up_scaled_inputs_steer)
            
           # self.up_scaled_inputs = [tf.concat(3,self.up_scaled_inputs_mlc)] + [tf.concat(3,self.up_scaled_inputs_steer)]
           # self.up_scaled_inputs = [normalize_minus_plus_1(v) for v in self.up_scaled_inputs]
            
        elif self.flag == 'mlc_vgg':
            print('Using: Steerable Pyramid and MLC')
            #VGG
            
            vgg = VGG((1,3,self.height,self.width))
            print(vgg)
            net, end_points = vgg_19_conv(self.input_tensor,'deep_gaze/feature_network_0')

            self.up_scaled_inputs_vgg, _ = extract_feature_outputs( end_points, ['conv5_1', 'relu5_2', 'conv5_3', 'relu5_4'],
          #                        'conv1_1', vgg , praefix='deep_gaze/feature_network_0/')
                                 'conv2_1', vgg , praefix='deep_gaze/feature_network_0/')
            #MLC features
            mlc = MLC_tf((args.bs,self.height_ds,self.width_ds,3), color = True, downsample=1, sigmas=args.sigmas, im_ds=args.im_ds)

            net_mlc = mlc.build_input_features_color(self.input_tensor)

            self.up_scaled_inputs_mlc = collect_features_mlc(net_mlc['features'], n_sigmas=len(args.sigmas))

            
            #self.up_scaled_inputs2 = merge_up_scaled_features(self.up_scaled_inputs_mlc,self.self.up_scaled_inputs_vgg, n_sigmas=len(args.sigmas)) 

            #self.up_scaled_inputs = list(flat2gen(list(self.up_scaled_inputs_mlc) + list(self.up_scaled_inputs_steer)))

            self.up_scaled_inputs = list(self.up_scaled_inputs_mlc) + list(self.up_scaled_inputs_vgg)
            
            #self.up_scaled_inputs = [tf.concat(3,self.up_scaled_inputs_mlc)] + [tf.concat(3,self.up_scaled_inputs_steer)]
                 
            
            
        elif self.flag == 'centerbias_only':
            self.up_scaled_inputs = [self.centerbias[:,:,:,tf.newaxis]]


            

        # Set up readout network

        net2, end_points2, self.debug = readout_network(self.up_scaled_inputs, hidden_units=[16, 32, 2, 1])
        net3, self.end_points3 = saliency_map_construction(net2, self.centerbias)
        self.end_points3['costs/log_likelihoods'] = log_likelihoods(self.end_points3['saliency_map/log_density'], self.x_inds, self.y_inds)


        self.dg_loss = construct_deep_gaze_loss( self.end_points3, hidden_layers= [16, 32, 2, 1],
                                            l2_penalty=0.,
                                            l1_penalty=0., args = args
                                         )

        self.end_points3.update(self.dg_loss)  
        self.end_points3.update(end_points2)
        

        self.dg_mean = -tf.reduce_mean(self.end_points3['costs/loss'])

        if training == True:
        # define the optimizer
            self.lr = tf.placeholder(tf.float32, shape=[])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            # create the train op

            train_var = slim.get_variables(scope='readout_network') + slim.get_variables(scope='upsample')
            

            #if training additional weights set here:
            if args.train_sp == True:
                #train_SP_weights = slim.get_variables(scope='SteerablePyramid/W_orients') 
                train_SP_weights = slim.get_variables(scope='SteerablePyramid/gabor') 
                #train_SP_weights = slim.get_variables(scope='SteerablePyramid/') 
                #train_SP_weights = slim.get_variables(scope='SteerablePyramid/W_orients') + slim.get_variables(scope='SteerablePyramid/gabor') 
                
                
                train_var = train_var + train_SP_weights
                self.trained_SP_params = [v.name for v in train_SP_weights]
                print('Training SP weights', self.trained_SP_params)
                
            self.trained_params = [v.name for v in train_var]
            print('variables to train',  self.trained_params)
            self.train_op = slim.learning.create_train_op(self.dg_mean, optimizer, variables_to_train=train_var) # specify variables to train her
        else:

            print('No Training!')
            

    
