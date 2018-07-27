from __future__ import print_function, division, unicode_literals
import numpy as np
import os
os.chdir('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/')
import sys
sys.path.append('/gpfs01/bethge/home/oeberle/Scripts/deepgaze/SaliencyModels/')
import scipy
from scipy import misc
import pickle
import tensorflow as tf
from VGGFeatures import gauss_blur, replication_padding, repeat

def get_gaussian(window_size, sigma):
    x = np.arange(window_size).astype(np.float32)
    x -= 0.5*window_size
    #x = x.astype(theano.config.floatX)
    X, Y = np.meshgrid(x, x)
    W  = np.exp(-0.5*(X**2+Y**2)/sigma**2)
    W /= W.sum()
    return W

cv2pyrdown_kernel = np.array([[1,4,6,4,1],
                                [4,16,24,16,4],
                                [6,24,36,24,6],
                                [4,16,24,16,4],
                                [1,4,6,4,1],
                                ])/256.


def get_gabor(angle, phase = 0., filtersize = -1, filterperiod = np.pi, elongation = 2., major_stddev = 2.):
    # Compute gabor filters for orientation features
    # Reimplemented from Matlab GBVS package /gbvs/saltoolbox/makeGaborFilterGBVS.m
    minor_stddev = major_stddev * elongation
    max_stddev = np.max([major_stddev,minor_stddev])

    sz = filtersize
    if (sz == -1):
        sz = np.ceil(max_stddev*np.sqrt(10.))
    else:
        sz = np.floor(sz/2.)

    psi = np.pi / 180. * phase
    rtDeg = np.pi / 180. * angle

    omega = 2. * np.pi / filterperiod
    co = np.cos(rtDeg)
    si = -np.sin(rtDeg)
    major_sigq = 2. * major_stddev**2
    minor_sigq = 2. * minor_stddev**2

    vec = np.arange(-sz,sz+1)
    vlen = len(vec)
    vco = vec*co
    vsi = vec*si

    major = np.repeat(vco[np.newaxis,:].T, vlen, 1) + np.repeat(vsi[np.newaxis,:], vlen, 0)
    major2 = major**2

    minor = np.repeat(vsi[np.newaxis,:].T, vlen, 1) - np.repeat(vco[np.newaxis,:], vlen, 0)
    minor2 = minor**2

    result = np.cos(omega*major + psi)*np.exp(-major2/major_sigq - minor2/minor_sigq)

    gabor = result - np.mean(result)
    gabor = gabor / np.sqrt(np.sum(gabor**2))
    
    return gabor


class IttiKoch(object):
    def __init__(self, scale = 8, sigma = 1., windowsize = 8, orientations = [0., 45., 90., 135. ], ref_scale = 4, s_min_max= [2,5], deltas= [3,4], gabor_size = -1, extra_blur = False, norm_stepsize = 10,feature_type = 'ik_features'):
        self.scale = scale
        self.sigma = sigma
        self.windowsize = windowsize
        self.gauss_kernel = get_gaussian(self.windowsize,self.sigma)
        self.orientations = orientations
        self.ref_scale = ref_scale
        self.s_min = s_min_max[0]
        self.s_max = s_min_max[1]
        self.deltas = deltas
        self.gabor_size = gabor_size
        self.extra_blur = extra_blur
        self.norm_stepsize = norm_stepsize
        self.feature_type = feature_type
        self.gabor_dict = {angle: get_gabor(angle, phase = 0., filtersize = self.gabor_size, filterperiod = np.pi, elongation = 2., major_stddev = 2.) for angle in self.orientations}
        #self.width = tf.Placeholder(tf.float32, shape = [])
        #self.height = tf.Placeholder(tf.float32, shape = [])
        self.pyramid_dict = {k: (int(480/(2**k)),int((640/(2**k)))) for k in range(self.scale+1)}

        print(self.s_min, self.s_max, self.deltas)

 
    def map_normalization(self,src):
        # Itti & Koch map normalization
        # Reimplemented from Matlab GBVS package /gbvs/saltoolbox/maxNormalizeStdGBVS.m 
        # and mexLocalMaximaGBVS.cc
       
        
        # Normalize to [0,1] range
        src = normalize_min_max(src)
  
        # Finding other local maxima      
        stepsize = self.norm_stepsize
        shape = src.get_shape().as_list()
        width = shape[2]
        height = shape[1]
        # find local maxima
        numlocal = tf.cast(0.,tf.float32)
        lmaxmean = tf.cast(0.,tf.float32)
        
        for y in range(0, height-stepsize, stepsize):
            for x in range(0, width-stepsize, stepsize):
                localimg = src[:,y:y+stepsize, x:x+stepsize,:]
                lmin, lmax = tf.reduce_min(localimg), tf.reduce_max(localimg)

                #lmin, lmax, dummy1, dummy2 = cv2.minMaxLoc(localimg)
                lmaxmean = lmaxmean + lmax
                numlocal =  numlocal + tf.cast(1.,tf.float32)
                
        # averaging over all the local regions (substract global max
        m_mean = lmaxmean/numlocal
        
        factor = (1.-m_mean)**2
        result = factor*src
        return result
    
    def normalize_min_max(self,src):
        # Normalize to [0,1] range
        M = tf.reduce_max(src)
        Min = tf.reduce_min(src)
        variance = (M-Min)**2
        
        target = tf.nn.batch_normalization(src, Min, variance, offset = None, scale= None ,variance_epsilon=tf.constant(1e-6,tf.float32))
        return target
    
    def map_normalization_tf(self,src):
        # Itti & Koch map normalization in Tensorflow
        # Reimplemented from Matlab GBVS package /gbvs/saltoolbox/maxNormalizeStdGBVS.m 
        # and mexLocalMaximaGBVS.cc
       
        # Renormalize to [0,1] range
        src = normalize_min_max(src)

        # Finding other local maxima
        
        stepsize = self.norm_stepsize
        # find local maxima
  
        patches = tf.extract_image_patches(src, ksizes = [1,stepsize,stepsize,1], strides= [1,stepsize,stepsize,1],rates=[1,1,1,1] ,padding = 'SAME')
        
        #lmins = tf.reduce_min(patches,reduction_indices=3, keep_dims=True) 
        lmaxs  = tf.reduce_max(patches,reduction_indices=3, keep_dims=True)
        
        
        m_mean = tf.reduce_mean(lmaxs)
        numlocal = tf.reshape(lmaxs,[-1])
        
        factor = (1.-m_mean)**2
        result = factor*src
        return result
        
    def gaussian_pyramid(self,src):
        net = {}
        _input = src
        net[0] = _input
        for level in range(1,self.scale+1):
            net[level] =  gauss_blur(_input, self.sigma, windowradius=int(self.windowsize/2),strides=[1,2,2,1], mode='NEAREST')
            _input = net[level]
            
        return net
    
    
    def gaussian_pyramid_center_surround(self,src):
        net_tmp = self.gaussian_pyramid(src)
        self.net['pre_features']['intensity_pyr'] = net_tmp
        net = self.center_surround_diffs(net_tmp)
           
        return net
    
    
    def center_surround_diffs(self, gauss_features, name = ''):
        net = {}
        for c in range(self.s_min,self.s_max):
            for d in self.deltas:
                s = c + d
                tmp = gauss_features[c]
                ref_shape = tf.shape(tmp)[1:3]
                resized = tf.image.resize_images(gauss_features[s], ref_shape)
                net[name + '_c_{}_s_{}'.format(c,s)] = tf.nn.relu(tmp-resized)
 
        return net
                    
    def get_color_features(self,R,G,B,Y):
        net = {}
        RR = self.gaussian_pyramid(R)
        GG = self.gaussian_pyramid(G)
        BB = self.gaussian_pyramid(B)
        YY = self.gaussian_pyramid(Y)
        
        self.net['pre_features']['R_pyr'] = RR
        self.net['pre_features']['G_pyr'] = GG
        self.net['pre_features']['B_pyr'] = BB
        self.net['pre_features']['Y_pyr'] = YY

                    
        for c in range(self.s_min,self.s_max):
            for d in self.deltas:
                
                s = c + d
                tmp1 = RR[c] - GG[c]
                ref_shape = tf.shape(tmp1)[1:3]
                
                resized_1 = tf.image.resize_images(GG[s], ref_shape)
                resized_2 = tf.image.resize_images(RR[s], ref_shape)
                tmp2 =  resized_1 - resized_2
                
                net['RG_c_{}_s_{}'.format(c,s)] = tf.nn.relu(tmp1-tmp2)       
                    
                tmp1 = BB[c] - YY[c]
                ref_shape = tf.shape(tmp1)[1:3]
                
                resized_1 = tf.image.resize_images(YY[s], ref_shape)
                resized_2 = tf.image.resize_images(BB[s], ref_shape)
                tmp2 =  resized_1 - resized_2
                
                net['BY_c_{}_s_{}'.format(c,s)] = tf.nn.relu(tmp1-tmp2)   

        return net
                    
    def get_orientation_features(self, I):
        II = self.gaussian_pyramid(I)
        II_orients = {}
        for level,I_level in II.items():
             II_orients[level] = self.orientation_features(I_level)
        
        self.net['pre_features']['orientation_pyr'] = II_orients
          
        net =  {}
        for c in range(self.s_min,self.s_max):
            for d in self.deltas:
                s = c + d
                for angle in self.orientations:
                    net['O_c_{}_s_{}_ang_{}'.format(c,s,int(angle))] = self.get_OrientationFM(II_orients[c]['orient_{}'.format(int(angle))],II_orients[s]['orient_{}'.format(int(angle))])

        return net
    
    
    def orientation_features(self, _input):
        
        net_orients = {}
        for angle in self.orientations:

            shape_0, shape_1 = np.shape(self.gabor_dict[angle])
            print(shape_0, shape_1)
            tmp1 = replication_padding(_input, axis=1, size=int(np.floor(shape_0/2)))
            tmp2 = replication_padding(tmp1, axis=2, size=int(np.floor(shape_1/2)))
            
            print('BEFORE', _input)
            net_orients['orient_{}'.format(int(angle))] = tf.nn.conv2d(tmp2, tf.cast(self.gabor_dict[angle][:,:,np.newaxis,np.newaxis], tf.float32), strides=[1, 1, 1, 1], padding="VALID")
            print('AFTER', net_orients['orient_{}'.format(int(angle))])
                        
        return net_orients
    
    @staticmethod
    def get_OrientationFM(I_c, I_s):
        ref_shape = tf.shape(I_c)[1:3]
        I_s_resized = tf.image.resize_images(I_s, ref_shape)
        result =  tf.nn.relu(I_c-I_s_resized)
        return result
    
    
    def get_intensity_conspicuity(self, net, ref_shape):
        normalized_net =  {k:self.map_normalization_tf(repeat(v, 2**(int(k.split('_')[2])-1))) for k,v in net.items()}
        return normalized_net

    def get_color_conspicuity(self, net, ref_shape ):

        normalized_net_RG =  {k:self.map_normalization_tf(repeat(v, 2**(int(k.split('_')[2])-1))) for k,v in net.items() if k.startswith('RG')}
        normalized_net_BY =  {k:self.map_normalization_tf(repeat(v, 2**(int(k.split('_')[2])-1))) for k,v in net.items() if k.startswith('BY')}
       
        net = {'{}_+_{}'.format(k1,k2): v1 + v2 for (k1,v1),(k2,v2) in zip(normalized_net_RG.items(), normalized_net_BY.items())}
        return net
    
    def get_orientation_conspicuity(self, net, ref_shape):
        normalized_net =  {k:self.map_normalization_tf(repeat(v, 2**(int(k.split('_')[2])-1))) for k,v in net.items()}
        net = {}
        # Add across-scale maps for each orientation
        for angle in self.orientations:
            net[angle] = self.map_normalization_tf(tf.add_n([v for k,v in normalized_net.items() if k.endswith(str(int(angle)))]))
       
        return net

        
    
    def build(self, _input):
        
        self.net = {'raw_readout_features': {}, 'readout_features': {}, 'filters': {}, 'pre_features':{},'features':{}, 'conspicuity':{}}
        self._input = _input
        R,G,B,Y,I = self.color_itensity_features(self._input)

 
        # 1. Obtain Feature maps and compute center surround differences
        
        # Intensity Gaussian Pyramid and Intensity Contrast features (six maps)
        self.net['features']['intensity'] = self.gaussian_pyramid_center_surround(I)
        
        # Across-scale color features (12 maps)
        self.net['features']['color'] = self.get_color_features(R,G,B,Y)

        # Across-scale orientation features (24 maps)
        self.net['features']['orientation'] = self.get_orientation_features(I)

        if self.feature_type == 'conspicuity':
            # 2. Compute Conspicuity maps 
            ref_shape = self.pyramid_dict[self.ref_scale]

            # normalizing and combining intensity feature maps (6 maps)
            self.net['conspicuity']['intensity'] =  self.get_intensity_conspicuity(self.net['features']['intensity'], ref_shape)
            # normalizing and combining color feature maps (6 maps)
            
            self.net['conspicuity']['color'] = self.get_color_conspicuity(self.net['features']['color'],ref_shape)

            # normalizing and combining orientation feature maps (4 maps)
            self.net['conspicuity']['orientation'] = self.get_orientation_conspicuity(self.net['features']['orientation'],ref_shape)

        # In case intermediate processing layers are desired, features are resized and normalized and can 
        # be used directly for readout
        
        elif self.feature_type == 'ik_features':
            # 1) Center surround maps
            # Normalized intensity center surround features
            self.net['readout_features']['intensity'] = {k: self.map_normalization_tf(repeat(v, 2**(int(k.split('_')[2])-1))) for k,v in self.net['features']['intensity'].items()}

            # Normalized RG and BY center surround features
            self.net['readout_features']['color'] = {k: self.map_normalization_tf(repeat(v, 2**(int(k.split('_')[2])-1)))  for k,v in self.net['features']['color'].items()}


            self.net['readout_features']['orientation'] = {k: self.map_normalization_tf(repeat(v, 2**(int(k.split('_')[2])-1))) for k,v in self.net['features']['orientation'].items()}

        
        elif self.feature_type == 'raw_features':
            _ref_shape = tf.shape(self.net['pre_features']['intensity_pyr'][1])[1:3]
            # 2) Raw feature maps
            # Intensity
            self.net['raw_readout_features']['intensity'] =[self.normalize_min_max(tf.image.resize_images(repeat(v, 2**(k-1)),_ref_shape)) for k,v in self.net['pre_features']['intensity_pyr'].items() if k >=1]

            # R,G,B,Y
            tmp_list = []
            for pyr in ['R_pyr', 'G_pyr', 'B_pyr', 'Y_pyr']: 
                tmp_list.append([self.normalize_min_max(tf.image.resize_images(repeat(v, 2**(k-1)),_ref_shape)) for k,v in self.net['pre_features'][pyr].items() if k >=1])            
            self.net['raw_readout_features']['color'] = tmp_list

            tmp_list = []
            for angle in self.orientations: 
                pyr = 'orient_{}'.format(int(angle))
                # Orientation
                print(self.net['pre_features']['orientation_pyr'])
                tmp_list.append([self.normalize_min_max(tf.image.resize_images(repeat(v[pyr], 2**(k-1)),_ref_shape)) for k,v in self.net['pre_features']['orientation_pyr'].items() if k >=1])
            self.net['raw_readout_features']['orientation'] = tmp_list

      
        
        # Optional: Add extra blurring to all features (not done in original Itti & Koch model)
        if self.extra_blur == True:
            self.net['readout_features']['intensity'] = {k:gauss_blur(v, 10., windowradius=5,) for k,v in self.net['readout_features']['intensity'].items()}
            
            self.net['readout_features']['color'] = {k:gauss_blur(v, 10., windowradius=5,) for k,v in self.net['readout_features']['color'].items()}
            
            self.net['readout_features']['orientation'] = {k:gauss_blur(v, 10., windowradius=5,) for k,v in self.net['readout_features']['orientation'].items()}



        return self.net
    
    
        
    def color_itensity_features(self, _input):
        
        def set_smaller_to_zeros(inp, threshold = 0.1):
            filled = tf.fill(tf.shape(inp), threshold)
            replace = tf.fill(tf.shape(inp), 0.)
            try:
                output = tf.select(tf.greater_equal(inp, filled),inp,replace)
            except AttributeError:
                output = tf.where(tf.greater_equal(inp, filled),inp,replace)

            return output
        
        r = _input[:,:,:,0][:,:,:,tf.newaxis]
        g = _input[:,:,:,1][:,:,:,tf.newaxis]
        b = _input[:,:,:,2][:,:,:,tf.newaxis]
        
        
        I = (r+g+b)/3. # maybe tf.reduce_mean?
        
        
        # Normalize r,g,b
        
        r = tf.truediv(r,I) 
        g = tf.truediv(g,I) 
        b = tf.truediv(b,I) 
        
        r = set_smaller_to_zeros(r)
        g = set_smaller_to_zeros(g)
        b = set_smaller_to_zeros(b)


        net_color = {}
        net_intensity = {}
        R = r-(g+b)/2.
        G = g-(r+b)/2.
        B = b - (r+g)/2.
        #Y =(r+g)/2. - tf.nn.relu(r-g)/2. - b 
        Y = r + g -2*(tf.nn.relu(r-g)/2. + b)

        return R,G,B,Y,I
        

    @staticmethod
    def get_IntensityFM(I_c,I_s):
        ref_shape = tf.shape(I_c)[1:3]
        
        print('Interpolation: from _, to_', I_c, I_s)

        #print(ref_shape)
        I_s_resized = tf.image.resize_images(I_s, ref_shape)
        result =  tf.nn.relu(I_c-I_s_resized)
        return result
    
    @staticmethod
    def get_ColorFM(I1_c, I2_c, I1_s, I2_s):
        ref_shape = tf.shape(I1_c)[1:3]
        print('Interpolcation: from _, to_', I1_c, I1_s)
        
        I1_s_resized = tf.image.resize_images(I1_s, ref_shape)
        I2_s_resized = tf.image.resize_images(I2_s, ref_shape)
                
        result =  tf.nn.relu((I1_c - I2_c  )- (I1_s_resized - I2_s_resized))
        return result

        
        
if __name__ == '__main__':
    print('Itti & Koch model')        


        
        
        
        
        
            
            
            
            
            
        


