from __future__ import print_function, division, unicode_literals
import numpy as np
import os
os.chdir('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/')
import scipy
from scipy import misc
import pickle
import tensorflow as tf

def get_gaussian(window_size, sigma):
    x = np.arange(window_size).astype(np.float32)
    x -= 0.5*window_size
    #x = x.astype(theano.config.floatX)
    X, Y = np.meshgrid(x, x)
    W  = np.exp(-0.5*(X**2+Y**2)/sigma**2)
    W /= W.sum()
    return W


class MLC(object):
    """ Steerable CNN implementation """
    def __init__(self,  image_shape, color=True, sigmas=None, window_size=71, downsample=1,
                 pooling_type='max'):
        self.image_shape = image_shape
        self.color = color
        if sigmas is None:
            sigmas = [5, 10, 20, 40, 60]
        self.sigmas = sigmas
        self.window_size = window_size
        self.downsample = downsample
        self.pooling_type = pooling_type
        
        
    def prepare_tf_image(self,image):
        """Transform a 0-to-1 RGB image to network input"""
        if image.ndim == 2:
            image = np.dstack([image, image, image])
        image -= 0.5
        net_input = image.copy()
        return net_input
        
    def build(self):
        _input = tf.placeholder(tf.float32, shape= (None,None,None), name = 'input_image')
        self._input = _input
        if self.color:
            self.net = self.build_input_features_color(_input=self._input)
        else:
            #self.net = self.build_input_features(input=_input)
            print('Only support color for now')
        
        
            
        #self.read_structure()
        #self.layer_outputs = [self.data]
        #for l in self.layers:
        #    print l
        #    self.layer_outputs.append(l.build(self.layer_outputs[-1]))
        #self.layer_outputs.append(ln.get_output(self.net['features_fullsize']))
        #self.layer_outputs.append(ln.get_output(self.net['features']))
        #self.output = theano.function([self.data], self.pool1)
        #self._full_output = theano.function([self.data], self.layer_outputs) 
    
    
    def build_input_features_color(self, _input):
        net = {}
    
        net['input'] = _input
        print('Image preprocessed?')
        
        net['PCA'] = get_rgb2pca(net['input'])
        
        #fixate_layer(net['PCA'])
        print('Are layers fixated?')
        

        feature_layers = []

        for i in range(3):
            
            l_input =  net['PCA'][:,:,tf.constant(i)][tf.newaxis,:,:,tf.newaxis]
           # l_input = ln.SliceLayer(net['PCA'], indices=slice(i, i+1), axis=1)
        
            net['l_input_{}'.format(i)] = l_input
            #print(ln.get_output_shape(l_input))

            window_size = self.window_size
            sigmas = self.sigmas
            filters = [get_gaussian(window_size, sigma) for sigma in sigmas]
            
            W = np.dstack(filters)[:,:,np.newaxis,:]
            net['W_{}'.format(i)] = W
            #print(W)
            
            
            l_means = tf.nn.conv2d(l_input, tf.cast(W, tf.float32), strides=[1,1,1,1], padding = 'SAME')
            
            #l_means = ln.Conv2DLayer(l_input, num_filters=len(filters), filter_size=(window_size, window_size),
            #                         W=W, pad=int(0.5*(window_size-1)), nonlinearity=None,)

            
            net['l_means_{}'.format(i)] = l_means
            
            print('Fixate l_means?')
            #fixate_layer(l_means)
            #l_orig_blowup = l        ##########n.ConcatLayer([l_input]*len(sigmas))
            l_orig_blowup = tf.pack([l_input]*(len(sigmas)), -1)
            net['l_orig_blowup_{}'.format(i)] = l_orig_blowup
            
            #l_variance_local = ln.ElemwiseMergeLayer([l_orig_blowup, l_means], merge_function=lambda X, Y: (X-Y)**2)
            l_variance_local = tf.square(tf.squeeze(l_orig_blowup,[3]) - l_means)
           # net['l_means_{}'.format(i)] = l_means
            net['l_variance_local_{}'.format(i)] = l_variance_local

            
            #variance_filters = np.zeros((len(filters), len(filters), window_size, window_size), theano.config.floatX)
            variance_filters = np.zeros((window_size, window_size, len(filters), len(filters)))
            
            
            for j in range(len(filters)):
                #variance_filters[i, i, :, :] = W[i, 0, :, :]
                variance_filters[:, :, j, j] = W[:, :, 0, j]
                print('var_filter',variance_filters.shape)
                
                
            #l_variance = ln.Conv2DLayer(l_variance_local, num_filters=len(filters), filter_size=(window_size, window_size),
            #                            W=variance_filters, pad=int(0.5*(window_size-1)), nonlinearity=None)
            l_variance = tf.nn.conv2d(l_variance_local, tf.cast(variance_filters, tf.float32), strides=[1,1,1,1], padding = 'SAME')
          
            net['l_variance_filter_{}'.format(i)] = variance_filters
            net['l_variance_{}'.format(i)] = l_variance
            
            print('Fixate l_variance?')
           # fixate_layer(l_variance)
            feature_layers.append(l_means)
            feature_layers.append(l_variance)
         
    
        #net['features_fullsize'] = ln.ConcatLayer(feature_layers)

        net['features_fullsize'] = tf.pack(feature_layers,-1)

        if self.downsample ==2:
            print('Downsample features by 2!')
            
            #net['features'] = tf.nn.conv2d(net['features_fullsize'], self.downsample,
            #                                 mode=self.pooling_type)
            net['features'] = tf.nn.avg_pool(tf.reshape(net['features_fullsize'],(1, 768, 1024, 5*6)), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            
            
        else:
            net['features'] = net['features_fullsize']
        return net
    
if __name__ == '__main__':
    print('MLC Network')