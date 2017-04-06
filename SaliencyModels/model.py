from __future__ import print_function, division, unicode_literals
import numpy as np
import sys

sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/SaliencyModels/')
sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/')
sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/OliverPlayground/')

#import bethgeflow as bf
import pickle as pickle
import random
import argparse
import tensorflow as tf
slim = tf.contrib.slim
from utils_oliver import repeat, my_gather, avgpool2d, set_up_dir
from VGGFeatures import vgg_19_conv, extract_feature_outputs, initialize_deep_gaze, relu, gauss_blur
from SteerableCNNTrainWeights import SteerableCNN
from vgg_structure import VGG
from MLCFeatures import MLC_tf#, prepare_tf_image




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
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, relu, slim.bias_add, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            _nets = []
            for k, input in enumerate(inputs): # Loop over selected layers/heights/sigmas
                #up_scaled_input = repeat(input, repeats=8)
                up_scaled_input = input
                _net = slim.conv2d(up_scaled_input, hidden_units[0], [1, 1], scope='conv1_part{}'.format(k), activation_fn=None,
                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(False), biases_initializer=None)

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
            return net, end_points
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
        feature_regularizations = None,
        space_regularizations = None,
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
    

    print(readout_layer1_weights)
    readout_layer1_weights = tf.concat(1, readout_layer1_weights)
    
    other_weights = [params[v] for v in params
                     if v.startswith('{}/conv'.format(deep_gaze_scope))
                     and not 'part' in v and not 'biases' in v]
    
    readout_weights = sorted([readout_layer1_weights] + other_weights, key=lambda v: v.name)
    print([v.name for v in readout_weights])
    
    
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



def collect_features(select_features, data, downsample_factor =2):
    'Select data and put in a feature Tensor'
    
                
    def collect_layer(layer,layer_data, select_features):
        layer_list = list(flat2gen([v for k,v in layer_data.iteritems() if k in select_features]))
        if downsample_factor >= 2:
            layer_upsampled = [repeat(d, downsample_factor**layer) for d in layer_list]
        else:
            layer_upsampled = layer_list
                
        return layer_upsampled
    
   
    layer_list = [tf.concat(3,[repeat(data[k], downsample_factor**l) for k in keys])  for l, keys in select_features.items()]
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
    mlc =tf.pack(mlc,3)
    
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
        self.input_tensor = tf.placeholder('float', shape=(args.bs,  self.height, self.width, args.channels))
        #else: 
        #    self.input_tensor = tf.placeholder('float', shape=(args.bs,  self.height_ds, self.width_ds, args.channels))

        
        #self.input_tensor = repeat(self.input_tensor,2)

            
        
        self.x_inds = tf.placeholder(tf.int32)
        self.y_inds = tf.placeholder(tf.int32)
        
        #Possibly get it from the BaselineModel? 
        self.centerbias = tf.placeholder('float', shape=(args.bs, self.height_ds, self.width_ds))
          
        
        # Determine model type
        
        if self.flag == 'steer':
            select_features = pickle.load(open(args.steer_dict, 'rb'), encoding ='latin1')
            levels = select_features.keys()
            
            filter_size, n_filters, n_bands, height =  15, 1, 8, 5
            A = SteerableCNN(filter_size, n_filters, n_bands, height, ds = self.downsampling, im_ds = args.im_ds)
            #Make SP an attribute:
            self.A = A
            data =  A.steerable_network_for_multiple_features(self.input_tensor) 

            self.filters = A.filters

            self.up_scaled_inputs = collect_features(select_features, data, downsample_factor=args.ds)


        elif self.flag == 'vgg':
            vgg = VGG((1,3,self.height,self.width))
            print(vgg)
            net, end_points = vgg_19_conv(self.input_tensor,'deep_gaze/feature_network_0')

            self.up_scaled_inputs, _ = extract_feature_outputs( end_points, ['conv5_1', 'relu5_1', 'relu5_2', 'conv5_3', 'relu5_4'],
          #                        'conv1_1', vgg , praefix='deep_gaze/feature_network_0/')
                                 'conv2_1', vgg , praefix='deep_gaze/feature_network_0/')

        elif self.flag == 'mlc':
            mlc = MLC_tf((args.bs,self.height_ds,self.width_ds,3), color = True, downsample=1, sigmas=args.sigmas, im_ds=args.im_ds)

            net_mlc = mlc.build_input_features_color(self.input_tensor)

            self.up_scaled_inputs = collect_features_mlc(net_mlc['features'],n_sigmas=len(args.sigmas))

        elif self.flag == 'steer_mlc':
            print('Using: Steerable Pyramid and MLC')
            #Steerable features
            select_features = pickle.load(open(args.steer_dict, 'rb'), encoding ='latin1')
            levels = select_features.keys()
            filter_size, n_filters, n_bands, height =  15, 1, 4, 5
            #filter_size, n_filters, n_bands, height =  15, 1, 8, 5
            A = SteerableCNN(filter_size, n_filters, n_bands, height, ds = self.downsampling, im_ds = args.im_ds)
            #Make SP an attribute:
            self.A = A
            data =  A.steerable_network_for_multiple_features(self.input_tensor)    
            self.filters = A.filters

            self.up_scaled_inputs_steer = collect_features(select_features, data, downsample_factor=args.ds)

            #MLC features
            mlc = MLC_tf((args.bs,self.height_ds,self.width_ds,3), color = True, downsample=1, sigmas=args.sigmas, im_ds=args.im_ds)

            net_mlc = mlc.build_input_features_color(self.input_tensor)

            self.up_scaled_inputs_mlc = collect_features_mlc(net_mlc['features'], n_sigmas=len(args.sigmas))

            self.up_scaled_inputs = merge_up_scaled_features(self.up_scaled_inputs_mlc,self.up_scaled_inputs_steer, n_sigmas=len(args.sigmas)) 

            
            
        elif self.flag == 'centerbias_only':
            self.up_scaled_inputs = [self.centerbias[:,:,:,tf.newaxis]]


            

        # Set up readout network

        net2, end_points2 = readout_network(self.up_scaled_inputs, hidden_units=[16, 32, 2, 1])
        net3, self.end_points3 = saliency_map_construction(net2, self.centerbias)
        self.end_points3['costs/log_likelihoods'] = log_likelihoods(self.end_points3['saliency_map/log_density'], self.x_inds, self.y_inds)


        self.dg_loss = construct_deep_gaze_loss( self.end_points3, hidden_layers= [16, 32, 2, 1],
                                            l2_penalty=0.,
                                            l1_penalty=0.
                                         )

        self.end_points3.update(self.dg_loss)  
        self.end_points3.update(end_points2)
        

        self.dg_mean = -tf.reduce_mean(self.end_points3['costs/loss'])

        if training == True:
        # define the optimizer
            self.lr = tf.placeholder(tf.float32, shape=[])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            # create the train op

            train_var = slim.get_variables(scope='readout_network')


            #if training additional weights set here:
            if args.train_sp == True:
                #train_SP_weights = slim.get_variables(scope='SteerablePyramid/W_orients') 
                train_SP_weights = slim.get_variables(scope='SteerablePyramid/') 
                train_var = train_var + train_SP_weights
                self.trained_SP_params = [v.name for v in train_SP_weights]
                print('Training SP weights', self.trained_SP_params)
            self.trained_params = [v.name for v in train_var]
            print('variables to train',  self.trained_params)
            self.train_op = slim.learning.create_train_op(self.dg_mean, optimizer, variables_to_train=train_var) # specify variables to train her
        else:
            print('No Training!')
            

    
