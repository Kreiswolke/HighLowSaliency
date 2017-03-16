from __future__ import print_function, division, unicode_literals
import numpy as np
import sys
sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/tf_steerable_pyramid')
sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/')
#import bethgeflow as bf
import pickle as pickle
import random
import argparse
import tensorflow as tf
slim = tf.contrib.slim
from utils_oliver import repeat, my_gather, avgpool2d, set_up_dir
from VGGFeatures import vgg_19_conv, extract_feature_outputs, initialize_deep_gaze, relu, gauss_blur
from SteerableCNN import SteerableCNN
from vgg_structure import VGG

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
            for k, input in enumerate(inputs):
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

def vgg_preprocess(images):
    """ images should be BHWC tensor of RGB images
    """
    VGG_MEAN = np.array([103.94, 116.78, 123.68])
    
    bgr_images = images[:, :, :,::-1]
    return bgr_images - VGG_MEAN

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


def prepare_batch(input_images,fixations_x, fixations_y, valid_ids, downscale = 2, factor = 1):
    import itertools
    im = np.array(input_images)[valid_ids]
   # x = list(itertools.chain(*np.reshape([[int((768./(downscale*1000.))*fix) for fix in fixations_x[n]] for n in valid_ids],-1)))        
    x = list(itertools.chain(*np.reshape([[int(factor*(1./downscale)*fix) for fix in fixations_x[n]] for n in valid_ids],-1)))        

    y = list(itertools.chain(*np.reshape([[int(factor*(1./downscale)*fix) for fix in fixations_y[n]] for n in valid_ids],-1)))        
    b = list(itertools.chain(*[int(i)*np.ones_like(fixations_x[n]) for i,n in enumerate(valid_ids)]))   
    return im, x, y, b

def collect_features(select_features, data, downsample_factor =2):
    'Select data and put in a feature Tensor'
    
    def flat2gen(alist):
        for item in alist:
            if isinstance(item, list):
                for subitem in item: yield subitem
            else:
                yield item
                
    def collect_layer(layer,layer_data, select_features):
        layer_list = list(flat2gen([v for k,v in layer_data.iteritems() if k in select_features]))
        layer_upsampled = [repeat(d, downsample_factor**layer) for d in layer_list]
        return layer_upsampled
    
    #layer_list = [collect_layer(k, data[v], v) for k,v in select_features.iteritems()]
    
    
   
    layer_list = [[repeat(data[k], downsample_factor**l) for k in keys]  for l, keys in select_features.items()]
    # return tf.pack(list(flat2gen(layer_list)))
    # pprint(list(flat2gen(layer_list)))
    return list(flat2gen(layer_list))


#########################################################################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--steer_dict', type=str, default='',
                       help='location of dict')
    parser.add_argument('--flag', type=str, default='vgg',
                       help='steer')
    parser.add_argument('--bs', type=int, default=4,
                       help='minibatch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='number of epochs')
    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels to be used, default 3')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='learning rate')
    parser.add_argument('--cb_dir', type=str, default='',
                        help='Centerbias dir, if provided it is used')
                  
    #parser.add_argument('--init_from', type=str, default=None,
    #                   help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
    #                        'config.pkl'        : configuration;
    #                        'chars_vocab.pkl'   : vocabulary definitions;
    #                        'checkpoint'        : paths to model file(s) (created by tf).
    #                                              Note: this file contains absolute paths, be careful when moving files around;
    #                        'model.ckpt-*'      : file(s) with model definition (created by tf)
    #                    """)
    args = parser.parse_args()
    
    # Init
    VGG_flag = False
    Steer_flag = False
    save_dir = args.save_dir
    steer_dict_dir = args.steer_dict
    bs = args.bs
    flag = args.flag
    epochs = args.epochs
    channels = args.channels
    lr = args.lr
    cb_dir = args.cb_dir
    
    
    set_up_dir(save_dir)
    # Prepare

   
    # Get Image data

    dat = open('/gpfs01/bethge/home/oeberle/Results/mit1003_463_onesize.p', 'rb')
    figrim_dat = pickle.load(dat, encoding='latin1')
    valid_inds = pickle.load(open('/gpfs01/bethge/home/oeberle/Results/dg_on_MIT/inds.p','rb'))
    
    input_images =  [figrim_dat['input_images'][i]/255. for i in valid_inds]
    fixations_x = [figrim_dat['fixations_x'][i] for i in valid_inds]
    fixations_y = [figrim_dat['fixations_y'][i] for i in valid_inds]
    
    n_ims = len(valid_inds)
    inds = list(range(n_ims))
    batch_ids = list([inds[(k*bs):((k+1)*bs)] for k in range(int(np.floor(n_ims/bs)))])




    if flag == 'steer':
        Steer_flag = True
        select_features = pickle.load(open(steer_dict_dir, 'rb'), encoding ='latin1')
        levels = select_features.keys()
        save_str = save_dir + 'deepgaze_train_lr_{}_batch_{}_steer'.format(lr,bs) + ''.join(['_{}'.format(s) for s in levels]) 
        
    elif flag == 'vgg':
        VGG_flag = True
        save_str = save_dir + 'deepgaze_train_lr_{}_batch_{}_vgg'.format(lr,bs)
    elif flag == 'centerbias_only':
        save_str = save_dir + 'deepgaze_train_lr_{}_batch_{}_centerbias_only'.format(lr,bs)
        
    if len(cb_dir) > 0:
        _centerbias = pickle.load(open(cb_dir, 'rb'), encoding ='latin1')
        _centerbias = np.repeat(_centerbias[np.newaxis,:,:], bs, axis = 0)
        save_str = save_str + 'cb.p'
    else:
        _centerbias = np.ones([bs, 384, 512])
        save_str = save_str + 'no_cb.p'
        

            
            
    vgg = VGG((1,3,768,1024))
    tf.reset_default_graph()
    g = tf.Graph()
    downscale = 2.
    
    print(save_str)
    #TRAINING

    with g.as_default():
        with tf.Session() as test_sess:
            
            x_inds = tf.placeholder(tf.int32)
            y_inds = tf.placeholder(tf.int32)
            centerbias = tf.placeholder('float', shape=(bs, 384, 512))

            # Extract image features

          

            
            
        if VGG_flag == True:

            input_tensor = tf.placeholder('float', shape=(bs, 768, 1024, channels))

            net, end_points = vgg_19_conv(input_tensor,'deep_gaze/feature_network_0')

            #initialize_deep_gaze(g, test_sess)

            up_scaled_inputs = extract_feature_outputs(
                end_points, ['conv5_1', 'relu5_1', 'relu5_2', 'conv5_3', 'relu5_4'],
                'conv2_1', vgg , praefix='deep_gaze/feature_network_0/')
            input_images2 = input_images

        elif Steer_flag == True:
            input_tensor = tf.placeholder('float', shape=(bs, 384, 512, channels))
            filter_size, n_filters, n_bands, height =  15, 1, 4, 5
            A = SteerableCNN(filter_size, n_filters, n_bands, height)

            data =  A.steerable_network_for_multiple_features(input_tensor)    
            up_scaled_inputs = collect_features(select_features, data, 2)
           # input_images2 = np.array([test_sess.run(maxpool2d(tf.cast(np.array(input_images[i])[np.newaxis,:,:,:], tf.float32))) for i in range(len(input_images))]).squeeze()
            input_images2 = pickle.load(open('/gpfs01/bethge/home/oeberle/Results/MIT_463_downsample_512_368_avgpool.p', 'rb'))

                
        elif flag == 'centerbias_only':
            # Centerbias only
            input_tensor = tf.placeholder('float', shape=(bs, 384, 512, channels))
            up_scaled_inputs = [tf.cast(_centerbias[:,:,:,np.newaxis], tf.float32)]
          #  input_images2 = np.array([test_sess.run(avgpool2d(tf.cast(np.array(input_images[i])[np.newaxis,:,:,:], tf.float32))) for i in range(len(input_images))]).squeeze()
            input_images2 = pickle.load(open('/gpfs01/bethge/home/oeberle/Results/MIT_463_downsample_512_368_avgpool.p', 'rb'))





        # Readout Network

        net2, end_points2 = readout_network(up_scaled_inputs, hidden_units=[16, 32, 2, 1])
        net3, end_points3 = saliency_map_construction(net2, centerbias)
        end_points3['costs/log_likelihoods'] = log_likelihoods(end_points3['saliency_map/log_density'], x_inds, y_inds)
        #init_op = tf.initialize_all_variables()


        dg_loss = construct_deep_gaze_loss( end_points3, hidden_layers= [16, 32, 2, 1],
                                        l2_penalty=0.,
                                        l1_penalty=0.
                                     )

        end_points3.update(dg_loss)  


        dg_mean = -tf.reduce_mean(end_points3['costs/loss'])

        # define the optimizer

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # create the train op

        train_var = slim.get_variables(scope='readout_network')
        train_op = slim.learning.create_train_op(dg_mean, optimizer, variables_to_train=train_var) # specify variables to train her

        # Initialize deepgaze
        test_sess.run(tf.initialize_all_variables())

        if VGG_flag == True:
            initialize_deep_gaze(g, test_sess)

        loss_list = []

        res_dict = {k: {'flag':flag,'lr': lr, 'inds':valid_inds,'sal_map': None, 'im':None, 'xinds':None,'yinds':None, 'binds': None, 'epoch_loss':None, 'run_loss': []} for k in range(epochs)}

        for k in range(epochs):
            print('epoch:', k)
            loss_run = []

            for i,j in enumerate(batch_ids):

                im, x, y, b = prepare_batch(input_images2,fixations_x, fixations_y, list(j))
                print(im.shape)

                init_feed_dict = {

                                input_tensor: im,
                                centerbias: _centerbias, # np.ones([bs, 384, 512]),
                                dg_loss['deep_gaze_loss/x_inds']: x,#np.array(xinds),
                                dg_loss['deep_gaze_loss/y_inds']: y,#np.array(yinds),
                                dg_loss['deep_gaze_loss/b_inds']: b,

                                  }     


                loss = test_sess.run(train_op, init_feed_dict)


                res_dict[k]['run_loss'].append(loss)
                print(k,i,loss)
                if i == 10:
                    a = np.exp(test_sess.run(end_points3['saliency_map/log_density'], init_feed_dict))
                    res_dict[k]['sal_map'] = a.squeeze()
                    res_dict[k]['im'] = im
                    res_dict[k]['xinds'] = x
                    res_dict[k]['yinds'] = y
                    res_dict[k]['binds'] = b



            res_dict[k]['epoch_loss'] = np.mean(res_dict[k]['run_loss'])
            pickle.dump(res_dict, open(save_str, 'wb'))

    pickle.dump(res_dict, open(save_str, 'wb'))

    

if __name__ == '__main__':
    main()
    print('DGII')
