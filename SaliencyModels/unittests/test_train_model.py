#!/usr/bin/env python

import unittest
import sys

sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/SaliencyModels/')
sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/')

from VGGFeatures import vgg_19_conv, extract_feature_outputs, initialize_deep_gaze, relu, gauss_blur
from vgg_structure import VGG

from MLCFeatures import MLC_tf,get_rgb2pca_batch_v1, get_gaussian
from model import collect_features, collect_features_mlc, merge_up_scaled_features, readout_network
from SteerableCNNTrainWeights import SteerableCNN
import numpy as np
import tensorflow as tf
import pickle


window_size = 141
sigmas = [10, 20, 40, 80, 160]

bs = 4
res_dict = pickle.load(open('/gpfs01/bethge/home/oeberle/Results/MLC/test_rgb_pca.p', 'rb'), encoding='latin-1')
expected = res_dict['RGB_PCA']
expected_batch = np.repeat(expected[np.newaxis,:,:,:],bs,axis = 0)
_input = np.swapaxes(res_dict['input'],0,1).astype(np.float32)
input_batch = np.repeat(_input[np.newaxis,:,:,:],bs,axis = 0)

input_batch_prep = input_batch/255. - 0.5


select_features = pickle.load(open('/gpfs01/bethge/home/oeberle/Results/dg_on_MIT/l_0_1_2_3_4_HP_LP_8_ORIENT.p', 'rb'), encoding ='latin1')



def fast_PCA(_input,bs):
    V_pca = np.array([[+0.40052664, -0.7275945,  -0.55694224],
                      [-0.81574151, -0.00632331, -0.57838206],
                      [+0.41730589,  0.68597833, -0.59606168]])
    V_pca = V_pca.T

    #[filter_height, filter_width, in_channels, channel_multiplier]
    depthwise_filter = tf.ones([1, 1, 3, 1], tf.float32)
    
    #[1, 1, channel_multiplier * in_channels, out_channels]
    pointwise_filter = tf.cast(V_pca[np.newaxis,np.newaxis,:,:],tf.float32)
    
    out = tf.nn.separable_conv2d(_input, depthwise_filter,pointwise_filter,strides=[1,1,1,1], padding="VALID")
    
    return out
    


class MLC_Model(unittest.TestCase):
    def test_PCA(self):
        bs = 4
        res_dict = pickle.load(open('/gpfs01/bethge/home/oeberle/Results/MLC/test_rgb_pca.p', 'rb'), encoding='latin-1')
        expected = res_dict['RGB_PCA']
        expected_batch = np.repeat(expected[np.newaxis,:,:,:],bs,axis = 0)
        _input = np.swapaxes(res_dict['input'],0,1)
        input_batch = np.repeat(_input[np.newaxis,:,:,:],bs,axis = 0)
        
        input_batch_prep = input_batch/255. - 0.5
        
         
        out =  get_rgb2pca_batch_v1(input_batch_prep, bs)
        
        with tf.Session() as sess:
            output = sess.run(tf.nn.relu(out))
                    
        self.assertTrue(np.allclose(output,expected_batch, atol = 5e-8*np.max(np.abs(expected))))    
        
        
    def test_fast_PCA(self):
        bs = 4
        res_dict = pickle.load(open('/gpfs01/bethge/home/oeberle/Results/MLC/test_rgb_pca.p', 'rb'), encoding='latin-1')
        expected = res_dict['RGB_PCA']
        expected_batch = np.repeat(expected[np.newaxis,:,:,:],bs,axis = 0)
        _input = np.swapaxes(res_dict['input'],0,1).astype(np.float32)
        input_batch = np.repeat(_input[np.newaxis,:,:,:],bs,axis = 0)
        
        input_batch_prep = input_batch/255. - 0.5
        
         
        out =  fast_PCA(input_batch_prep,bs)
        
        with tf.Session() as sess:
            output = sess.run(tf.nn.relu(out))
                    
        self.assertTrue(np.allclose(output,expected_batch, atol = 5e-8*np.max(np.abs(expected))))    
        
        
  #  def test_stacked_filters(self):
  #      filters = [get_gaussian(window_size, sigma) for sigma in sigmas]
  #      W = np.dstack(filters)[:,:,np.newaxis,:]
  #      _input = tf.placeholder(tf.float32, shape= (None,None,None,None), name = 'input_image')
  #      
  #      hilf = fast_PCA(_input, bs)
  #      i = 1#
#
 #       l_input = hilf[:,:,:,tf.constant(i)][:,:,:,tf.newaxis]

  #      l_means = tf.nn.conv2d(l_input, tf.cast(W, tf.float32), strides=[1,1,1,1], padding = 'SAME')
        
  #      l_orig_blowup = tf.pack([l_input]*(len(sigmas)), -1)

   #     with tf.Session() as sess:
   #         output = sess.run(l_orig_blowup, {_input:input_batch_prep} )
   #     print(W.shape,input_batch_prep.shape, output.shape)
        
        
    def test_MLC_net(self):
        input_tensor = tf.placeholder('float', shape=(bs, None,None, 3))

        mlc = MLC_tf(input_batch.shape, color = True, downsample=1, sigmas=sigmas, im_ds=2)

        net = mlc.build_input_features_color(input_tensor)
        
        up_scaled_inputs = collect_features_mlc(net['features'],n_sigmas=len(sigmas))

        with tf.Session() as sess:
            output = sess.run(net['l_variance_1'], {input_tensor:input_batch_prep} )
            all_features = sess.run(net['features'], {input_tensor:input_batch_prep} )
        print(all_features.shape, output.shape)
        
      
    def test_steer_MLC_net(self):
        input_tensor = tf.placeholder('float', shape=(bs, None,None, 3))


        levels = select_features.keys()
        filter_size, n_filters, n_bands, height =  15, 1, 8, 5
        #filter_size, n_filters, n_bands, height =  15, 1, 8, 5
        A = SteerableCNN(filter_size, n_filters, n_bands, height, ds = 1, im_ds = 2)
        #Make SP an attribute:

        data =  A.steerable_network_for_multiple_features(input_tensor)    

        up_scaled_inputs_steer = collect_features(select_features, data, downsample_factor=2)

        #MLC features
        mlc = MLC_tf(input_batch.shape, color = True, downsample=1, sigmas=sigmas, im_ds=2)

        net_mlc = mlc.build_input_features_color(input_tensor)
        up_scaled_inputs_mlc = collect_features_mlc(net_mlc['features'], n_sigmas=len(sigmas))

        #Combine
        up_scaled_inputs = merge_up_scaled_features(up_scaled_inputs_mlc,up_scaled_inputs_steer, n_sigmas=len(sigmas)) 

        net2, end_points2 = readout_network(up_scaled_inputs, hidden_units=[16, 32, 2, 1])


        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            #output = sess.run(net['l_variance_1'], {input_tensor:input_batch_prep} )
            #all_features = sess.run(net_mlc['features'], {input_tensor:input_batch_prep} )
            #output =  sess.run(up_scaled_inputs, {input_tensor:input_batch_prep} )
            steer_features = sess.run(up_scaled_inputs_steer, {input_tensor:input_batch_prep} )
            print('STEER_MLC',np.shape(steer_features),np.shape(all_features),np.shape(output))
    
    if False:
        class VGG_Model(unittest.TestCase):
            def test_vgg(self):

                input_tensor = tf.placeholder('float', shape=(bs, None,None, 3))

                vgg = VGG((1,3,768,1024))
                print(vgg)
                net, end_points = vgg_19_conv(input_tensor,'deep_gaze/feature_network_0')

                up_scaled_inputs, _ = extract_feature_outputs( end_points, ['conv5_1', 'relu5_1', 'relu5_2', 'conv5_3', 'relu5_4'],
                                                                       'conv2_1', vgg , praefix='deep_gaze/feature_network_0/')

                net2, end_points2 = readout_network(up_scaled_inputs, hidden_units=[16, 32, 2, 1])


                with tf.Session() as sess:
                    sess.run(tf.initialize_all_variables())
                    output =  sess.run(up_scaled_inputs, {input_tensor:input_batch_prep} )

                print('VGG', np.shape(output))

if __name__ == '__main__':
    unittest.main()