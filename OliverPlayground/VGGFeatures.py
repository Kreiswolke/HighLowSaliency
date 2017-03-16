from __future__ import print_function, division, unicode_literals
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

#### UTIL FUNCTIONS #####


def transform_layer(input_tensor, position_information, from_name, to_name):
    """ Transform data from one layer to match shape of another layer in a network by resizing and cropping """
    n1 = position_information.layer_indices[to_name]
    n2 = position_information.layer_indices[from_name]
    upscale_params = position_information.upscale_parameters(n1, n2)

    return advanced_upscale_layer(input_tensor, upscale_params)

def advanced_upscale_layer(inputs, parameters, scope='upscale'):
    with tf.name_scope(scope, 'upscale', [inputs]) as sc:
        top_slice = inputs[:,
                           parameters['y_min_top']:parameters['y_max_top'],
                           parameters['x_min_top']:parameters['x_max_top'],
                           :
                           ]
        repeated = repeat(top_slice, repeats=int(parameters['factor']))
        bottom_slice = repeated[:,
                                parameters['y_min_bottom']:parameters['y_max_bottom'],
                                parameters['x_min_bottom']:parameters['x_max_bottom'],
                                :
                                ]
        return bottom_slice
    
    
def normalize_axis(input_tensor, axis):
    if axis < 0:
        ndims = len(input_tensor.get_shape())
        axis = ndims + axis
    return axis
    

def replication_padding(input_tensor, axis=0, size=1):
    with tf.name_scope('replication_padding'):
        if not isinstance(size, (tuple, list)):
            size = (size, size)
        ndims = len(input_tensor.get_shape())
        axis = normalize_axis(input_tensor, axis)
        start_slice_obj = [slice(None)] * axis + [slice(0, 1)]
        start_slice = input_tensor[start_slice_obj]
        repeats = [1] * axis + [size[0]] + [1] * (ndims-axis-1)
        start_part = tf.tile(start_slice, repeats)
        end_slice_obj = [slice(None)] * axis + [slice(-1, None)]
        end_slice = input_tensor[end_slice_obj]
        repeats = [1] * axis + [size[1]] + [1] * (ndims-axis-1)
        end_part = tf.tile(end_slice, repeats)
        return tf.concat(axis, (start_part, input_tensor, end_part))

def repeat(input_tensor, repeats=2):
    """ Analogue to np.repeat: resize a tensor by repeating pixels
        - input_tensor: tensor in BHWC format
        - repeats: integer or pair of integers, how often to repeat pixels
    """
    shape = tf.shape(input_tensor)
    known_shape = input_tensor.get_shape()
    
    if not isinstance(repeats, (tuple, list)):
        repeats = (repeats, repeats)
    n_channels = int(input_tensor.get_shape()[3])
    kernel_size = (repeats[0], repeats[1], n_channels, n_channels)
    output_shape = (shape[0], shape[1]*repeats[0], shape[2]*repeats[1], shape[3])
    infered_output_shape = (known_shape[0], known_shape[1]*repeats[0], known_shape[2]*repeats[1], n_channels)
    strides = (1, repeats[0], repeats[1], 1)
    
    kernel = np.zeros(kernel_size, dtype='float32')
    for i in range(n_channels):
        kernel[:, :, i, i] = np.ones((repeats[0], repeats[1]))
    
    out = tf.nn.conv2d_transpose(input_tensor,
                                 kernel,
                                 output_shape=output_shape,
                                 strides=strides)
    # conv2d_transpose does not infer the output shape correctly, therefore we set it
    out.set_shape(infered_output_shape)
    return out
    
def get_gaussian_kernel(sigma, windowradius=5):
    with tf.name_scope('gaussian_kernel'):
        kernel = tf.cast(tf.range(0, 2*windowradius+1), 'float') - windowradius
        kernel = tf.exp(-(kernel**2)/(2*sigma**2))
        kernel /= tf.reduce_sum(kernel)
        return kernel

def blowup_1d_kernel(kernel, axis=-1):
    #with tf.name_scope("blowup_1d_kernel")
    assert isinstance(axis, int)
    
    shape = [1 for i in range(4)]
    shape[axis] = -1
    return tf.reshape(kernel, shape)

def extract_feature_outputs(feature_network, layer_names, rel_layer, position_information, praefix=''):
    
    if isinstance(position_information, int):
        position_information = [position_information]*len(feature_network)
    
    features = []
    features_untransformed = []
    if isinstance(position_information, list):
        for k, layer_name in enumerate(layer_names):
            input_layer = feature_network[praefix+layer_name]
            rescaled_layer = repeat(input_layer, repeats=position_information[k])
            features.append(rescaled_layer)
    
    else:
    
        for layer_name in layer_names:
            layer = feature_network[praefix+layer_name]
            features_untransformed.append(layer)
            features.append(transform_layer(layer, position_information, layer_name, rel_layer))
    
    return features, features_untransformed


#### SLIM FUNCTION #####

@slim.add_arg_scope
def gaussian_convolution_along_axis(inputs, axis, sigma, windowradius=5, mode='NEAREST', scope=None,
                                    outputs_collections=None):
    with tf.name_scope(scope, 'gauss_1d', [inputs, sigma, windowradius]) as sc:
        if mode == 'NEAREST':
            inputs = replication_padding(inputs, axis=axis+1, size=windowradius)
        elif mode == 'VALID':
            pass
        else:
            raise ValueError(mode)

        kernel_1d = get_gaussian_kernel(sigma, windowradius=windowradius)
        kernel = blowup_1d_kernel(kernel_1d, axis)
        print(windowradius)
        
    
        output = tf.nn.conv2d(inputs, kernel, strides=[1,1,1,1], padding="VALID", name='gaussian_convolution')
        return output
    
        return slim.utils.collect_named_outputs(outputs_collections, sc, output)


@slim.add_arg_scope
def gauss_blur(inputs, sigma, windowradius=5, mode='NEAREST', scope=None,
                                    outputs_collections=None):
    with tf.name_scope(scope, 'gauss_blur', [inputs, sigma, windowradius]) as sc:
        
        outputs = inputs
        
        for axis in [0, 1]:
        
            outputs = gaussian_convolution_along_axis(outputs,
                                                      axis=axis,
                                                      sigma=sigma,
                                                      windowradius=windowradius,
                                                      mode=mode)
        return outputs
        
        return slim.utils.collect_named_outputs(outputs_collections, sc, outputs)

@slim.add_arg_scope
def relu(inputs, outputs_collections=None, scope=None):
    with tf.name_scope(scope, 'ReLu', [inputs]) as sc:
        inputs = tf.convert_to_tensor(inputs)
    
        outputs = tf.nn.relu(inputs)
        return slim.utils.collect_named_outputs(outputs_collections, sc, outputs)

@slim.add_arg_scope
def separate_conv_and_relu(*args, scope='conv', outputs_collections=None, **kwargs):
    net = slim.conv2d(*args, scope=scope, outputs_collections=outputs_collections, activation_fn=None, **kwargs)
    new_scope = 'relu'.join(scope.rsplit('conv', 1))
    net = relu(net, scope=new_scope, outputs_collections=outputs_collections)
    return net


def vgg_19_conv(inputs,
             scope='vgg_19'):
    """Oxford Net VGG 19-Layers version E Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([separate_conv_and_relu, slim.conv2d, relu, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, separate_conv_and_relu, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, separate_conv_and_relu, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, separate_conv_and_relu, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, separate_conv_and_relu, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, separate_conv_and_relu, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            
            # add caffe-style layer names for compatibility
            for block in range(5):
                for part in range(4):
                    for layer in ['conv', 'relu']:
                        source = '{sc}/conv{block}/{layer}{block}_{part}'.format(
                            sc=sc.name, block=block+1, layer=layer,part=part+1)
                        if source in end_points:
                            end_points['{sc}/{layer}{block}_{part}'.format(
                            sc=sc.name, block=block+1, layer=layer,part=part+1)] = end_points[source]

                    
            
            return net, end_points
        
        
def initialize_deep_gaze(deep_gaze, sess): # initialization='rand_normalized', initial_sigma=5.0):
    ops = []
    #with tf.Session(graph=self.graph) as sess: ???
    #with deep_gaze.as_default():
        
       # if deep_gaze.network_name == 'VGG':
            
    variables = []
    print("Set VGG..")
    for block in range(1, 6):
        block_size = 4 if block > 2 else 2
        for sub_module in range(1, block_size+1):
            target_name = 'feature_network_0/conv{0}/conv{0}_{1}'.format(block, sub_module)
            print(slim.get_variables_by_name(target_name+'/weights'))
            variables.append(slim.get_variables_by_name(target_name+'/weights')[0])
            variables.append(slim.get_variables_by_name(target_name+'/biases')[0])
    print("Done")

    saver = tf.train.Saver(variables)
    saver.restore(sess, '/gpfs01/bethge/home/oeberle/Scripts/deepgaze/vgg_data/vgg_19_conv.ckpt')
    print("Done")    

if __name__ == '__main__':
    print('VGG Features')
