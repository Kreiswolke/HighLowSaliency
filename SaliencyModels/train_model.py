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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Model


def init_steerable(sess, filters, height =5, n_features = 3):
    
    for f in range(n_features):
        for l in range(height):
            for case in ['real', 'imag']:
                lp_name = 'steer_layer_{}/lp_weight_{}_f{}'.format(l, case,f)
                hp_name = 'steer_layer_{}/hp_weight_{}_f{}'.format(l, case,f)

                var_op = slim.get_variables_by_name(lp_name)[0]
                var_op.assign(tf.cast(filters[0]['highpass_{}'.format(case)], tf.float32))
                sess.run(var_op)

                var_op = slim.get_variables_by_name(lp_name)[0]
                var_op.assign(tf.cast(filters[0]['lowpass_{}'.format(case)], tf.float32))
                sess.run(var_op)

                #for o in range(4):
                for o in range(8):
                    orient_name = 'steer_layer_{}/orient_weight_o{}_{}_f{}'.format(l, o, case,f)
                    var_op = slim.get_variables_by_name(orient_name)[0]
                    var_op.assign(tf.cast(filters[0]['orientation_{}'.format(case)][o], tf.float32))
                    sess.run(var_op)
    print('Init SP: Done')

def prepare_batch(input_images,fixations_x, fixations_y, valid_ids, downscale = 2, factor = 1, vgg_prep=False):
    import itertools
    if vgg_prep==True:
        print('VGG preprocessing')
        im = vgg_preprocess(np.array(input_images)[valid_ids])
    else:
        im = (np.array(input_images)[valid_ids])/255.

   # x = list(itertools.chain(*np.reshape([[int((768./(downscale*1000.))*fix) for fix in fixations_x[n]] for n in valid_ids],-1)))        
    x = list(itertools.chain(*np.reshape([[int(factor*(1./downscale)*fix) for fix in fixations_x[n]] for n in valid_ids],-1)))        

    y = list(itertools.chain(*np.reshape([[int(factor*(1./downscale)*fix) for fix in fixations_y[n]] for n in valid_ids],-1)))        
    b = list(itertools.chain(*[int(i)*np.ones_like(fixations_x[n]) for i,n in enumerate(valid_ids)]))   
    return im, x, y, b

def prepare_batch_salicon(input_images,fixations_x, fixations_y, downscale = 2, factor = 1, vgg_prep=False):
    import itertools
    if vgg_prep==True:
        im = vgg_preprocess(np.array(input_images))
        #im = vgg_preprocess(np.asarray(input_images, np.float32))
    else:
        im = (np.array(input_images))/255.

    n_ims = len(input_images)    
    x = list(itertools.chain(*np.reshape([[int(factor*(1./downscale)*fix) for fix in fixations_x[n]] for n in range(n_ims)],-1)))        
    y = list(itertools.chain(*np.reshape([[int(factor*(1./downscale)*fix) for fix in fixations_y[n]] for n in range(n_ims)],-1)))        
    b = list(itertools.chain(*[int(n)*np.ones_like(fixations_x[n]) for n  in range(n_ims)]))   
    return im, x, y, b


def prepare_tf_image(image):
    """Transform a 0-to-1 RGB image to network input"""
    if image.ndim == 2:
        image = np.dstack([image, image, image])
    net_input = image.copy()
    return net_input

def vgg_preprocess(images):
    """ images should be BHWC tensor of RGB images
    """
    VGG_MEAN = np.array([103.94, 116.78, 123.68])
    
    bgr_images = images[:, :, :,::-1]
    return bgr_images - VGG_MEAN

def get_MIT_463_data(data_dir, inds_dir):
    dat = open(data_dir, 'rb')
    figrim_dat = pickle.load(dat, encoding='latin1')
    valid_inds = pickle.load(open(inds_dir,'rb'))
    
    input_images =  [figrim_dat['input_images'][i] for i in valid_inds]
    fixations_x = [figrim_dat['fixations_x'][i] for i in valid_inds]
    fixations_y = [figrim_dat['fixations_y'][i] for i in valid_inds]
    
    return input_images, fixations_x, fixations_y, valid_inds


def get_SALICON_data(inds):

    input_images = []
    fixations_x = []
    fixations_y = []
    for i in inds:
        _str = '/gpfs01/bethge/home/oeberle/Results/SALICON_train_full/SALICON_train_{}.p'.format(i)
        figrim_dat = pickle.load(open(_str,'rb'), encoding='latin1')

        # input_images.append(figrim_dat['input_images']) # Expecting (HxWx3) data
        input_images.append(prepare_tf_image(figrim_dat['input_images']))
        fixations_x.append(figrim_dat['fixations_x'])
        fixations_y.append(figrim_dat['fixations_y'])

    return input_images, fixations_x, fixations_y
    
    
def get_data(data_dir, inds):

    input_images = []
    fixations_x = []
    fixations_y = []
    for i in inds:
        _str = data_dir.format(i)
        figrim_dat = pickle.load(open(_str,'rb'), encoding='latin1')

        # input_images.append(figrim_dat['input_images']) # Expecting (HxWx3) data
        input_images.append(prepare_tf_image(figrim_dat['input_images']))
        fixations_x.append(figrim_dat['fixations_x'])
        fixations_y.append(figrim_dat['fixations_y'])

    return input_images, fixations_x, fixations_y

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
   # parser.add_argument('--gamma', type=float, default=0.92,
   #                    help='learning rate decay')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='learning rate decay')
    parser.add_argument('--cb_dir', type=str, default='',
                        help='Centerbias dir, if provided it is used')
    parser.add_argument('--sigmas', type=int, nargs='+', default = None,
                       help='Provide sigmas for MLC features')
    parser.add_argument('--ds', type=int, default=2,
                       help='Downsample factor, ds=1 means no downsampling.')
    parser.add_argument('--data_dir', type=str, default='/gpfs01/bethge/home/oeberle/Results/MIT463_onesize/MIT_onesize_{}.p',
                       help='file to get image and fixation data from')
    parser.add_argument('--inds_dir', type=str, default='/gpfs01/bethge/home/oeberle/Results/dg_on_MIT/inds.p',
                       help='file to get image and valid_inds data from')
    parser.add_argument('--im_ds', type=int, default=2,
                        help='Downsample factor for the input image.')
    parser.add_argument('--train_sp', type=bool, default=False,
                        help='Decide if SP filters should be trained.')
    parser.add_argument('--save_flag', type=str, default='',
                        help='Additional descriptor for results directory.')
    parser.add_argument('--ckpt_dir', type=str, default='',
                       help='directory to load and use checkpointed models')
    parser.add_argument('--fold', type=int, default = None,
                        help='Nth-fold of crossvalidation')
    args = parser.parse_args()
    
   
    set_up_dir(args.save_dir)
    
    train(args)
    
def train(args):
   
    bs = args.bs

    # Get Image data
    #input_images, fixations_x, fixations_y, valid_inds = get_saliency_data(args.data_dir, args.inds_dir)
    image_dummy, _, _ = get_data(args.data_dir,[0,1])
    
    valid_inds = pickle.load(open(args.inds_dir,'rb'))

    
    height, width = np.shape(image_dummy[0])[0], np.shape(image_dummy[0])[1]
    
    
    # Preparing centerbias

    if len(args.cb_dir) > 0:
        print('Using centerbias')
        _centerbias = pickle.load(open(args.cb_dir, 'rb'), encoding ='latin1')
        _centerbias = np.repeat(_centerbias[np.newaxis,:,:], args.bs, axis = 0)
        cb_str = '_cb'
    else:
        print('Using no/uniform bias')
        _centerbias = np.ones([bs, int(height/args.im_ds),  int(width/args.im_ds)])
        
        cb_str = '_no_cb'
        
        
    save_dir = args.save_dir  + '/' +  args.flag + args.save_flag + cb_str + '/train/fold_{}/'.format(args.fold)   
    set_up_dir(save_dir)
    
    checkdir =  save_dir + '/Checkpoints/' 
    set_up_dir(checkdir)
    
    filter_dir = save_dir + 'Filters/'
    set_up_dir(filter_dir)
    
    save_str = save_dir + '/' + args.flag + '_lr_{}_batch_{}{}_fold_{}.p'.format(args.lr,args.bs, cb_str, args.fold)

    
    n_ims = len(valid_inds)
    inds = list(range(n_ims))
    batch_ids = list([inds[(k*bs):((k+1)*bs)] for k in range(int(np.floor(n_ims/bs)))])
            
    
    print('Image shape: {}x{}'.format(height, width))
    print('save_dir', save_dir)
    tf.reset_default_graph()
    g = tf.Graph()
    

    with g.as_default():
        model = Model(args,height,width)
        with tf.Session() as test_sess:    
            saver = tf.train.Saver()
            
            if len(args.ckpt_dir) > 0:
                print('Restore model from checkpoint')
                saver.restore(test_sess, args.ckpt_dir)
            else:
            # Initialize deepgaze
                print('Initialize all variables and model')
                test_sess.run(tf.initialize_all_variables())

                if args.flag == 'vgg':
                    print('Initialize VGG')
                    initialize_deep_gaze(g, test_sess)
                elif args.flag == 'steer' or args.flag == 'steer_mlc' :
                    model.A.init_filters(test_sess)
                    #print('No initialization')
            

            res_dict = {k: {'gamma': args.gamma, 'ckpt_dir': args.ckpt_dir, 'im_ds': args.im_ds,'ds': args.ds, 'sigmas':args.sigmas, 'inds_dir': args.inds_dir, 'cb_dir': args.cb_dir, 'steerdict':args.steer_dict, 'flag':args.flag,'lr': args.lr, 'inds':valid_inds,'sal_map': None, 'im':None, 'xinds':None,'yinds':None, 'binds': None, 'epoch_loss':None, 'run_loss': []} for k in range(args.epochs)}
            pickle.dump(res_dict, open(save_str, 'wb'))

            
            curr_lr = args.lr
            for k in range(args.epochs):
                print('epoch:', k)
                loss_run = []

                for i,j in enumerate(batch_ids):
                    print('batch_i',j)
                    
                    if args.flag == 'vgg':
                        
                        input_images,fixations_x, fixations_y = get_data(args.data_dir, list(j))
                        im, x, y, b = prepare_batch_salicon(input_images,fixations_x, fixations_y, downscale = args.im_ds, vgg_prep = True)
                    else:
                        print('Normalization and -0.5 Preprocessing')
                        #im, x, y, b = prepare_batch(input_images,fixations_x, fixations_y, list(j), downscale = args.ds, vgg_prep = False)
                        input_images,fixations_x, fixations_y = get_data(args.data_dir, list(j))
                        im, x, y, b = prepare_batch_salicon(input_images,fixations_x, fixations_y, downscale = args.im_ds,vgg_prep = False)
                        im = im - 0.5
                        
                    

                    init_feed_dict = {

                                    model.input_tensor: im,
                                    model.centerbias: _centerbias, # np.ones([bs, 384, 512]),
                                    model.dg_loss['deep_gaze_loss/x_inds']: x,#np.array(xinds),
                                    model.dg_loss['deep_gaze_loss/y_inds']: y,#np.array(yinds),
                                    model.dg_loss['deep_gaze_loss/b_inds']: b,
                                    model.lr: curr_lr 

                                      }     
                
                    loss = test_sess.run(model.train_op, init_feed_dict)


                    res_dict[k]['run_loss'].append(loss)
                    print(k,i,loss,test_sess.run(model.lr, init_feed_dict))

                    #return test_sess.run(up_scaled_inputs, init_feed_dict)
          
                    if i %10 == 0 and args.train_sp == True:
                        filt_dict = {k:test_sess.run(k) for k in model.trained_SP_params}
                        pickle.dump(filt_dict, open(filter_dir +'filt_ep_{}_batch_{}.p'.format(k,i), 'wb'))

                    
                    if k == 0 and i ==0:
                        a = np.exp(test_sess.run(model.end_points3['saliency_map/log_density'], init_feed_dict))
                        xinds = list(np.array(x)[np.array(b) == 2])
                        yinds = list(np.array(y)[np.array(b) == 2])
                        check_dict = {'map':a.squeeze()[1,:,:], 'x': xinds, 'y': yinds}
                        pickle.dump(check_dict, open(save_dir + 'check_dict.p', 'wb'))

                            

                    if i == 10:
                        a = np.exp(test_sess.run(model.end_points3['saliency_map/log_density'], init_feed_dict))
                        res_dict[k]['sal_map'] = a.squeeze()
                        res_dict[k]['im'] = im
                        res_dict[k]['xinds'] = x
                        res_dict[k]['yinds'] = y
                        res_dict[k]['binds'] = b

                            
                # Decay learning rate by gamma        
                curr_lr = curr_lr * args.gamma 

                res_dict[k]['epoch_loss'] = np.mean(res_dict[k]['run_loss'])
                pickle.dump(res_dict, open(save_str, 'wb'))
                
                # Save the variables to disk.
                save_path = saver.save(test_sess, checkdir + '/model_{}.ckpt'.format(k))
                print("Model saved in file: %s" % save_path)
    pickle.dump(res_dict, open(save_str, 'wb'))
        
    
    
if __name__ == '__main__':
    main()
    print('DGII Training')