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
import sys
import subprocess
import os
from tensorflow.core.protobuf import saver_pb2


def get_loss(res_dict, i = 0):

    if len(res_dict[i]['sal_map'].shape) == 3:
        h,w = np.shape(res_dict[i]['sal_map'])[1],np.shape(res_dict[i]['sal_map'])[2]
    elif len(res_dict[i]['sal_map'].shape) == 2:
        h,w = np.shape(res_dict[i]['sal_map'])[0],np.shape(res_dict[i]['sal_map'])[1]
    else:
        raise Error
    #print('get_loss: HxW: {} X {}'.format(h,w)) 

    loss = -(np.mean(res_dict[i]['run_loss']) - np.log((h*w)))/np.log(2.) 
    
    return loss


def get_inds(args, inds_dir):
    valid_inds = pickle.load(open(inds_dir,'rb'))
    if args.exclude_inds is not None:
        valid_inds_copy = np.copy(valid_inds)
        valid_inds = [i for i in valid_inds if i not in args.exclude_inds]
        print('Excluded indices: {}'.format(list(set(valid_inds_copy) - set(valid_inds))))
    return valid_inds


def run_evaluation(args, eval_dir, ckpt_dir, ep):
    test_file = '/gpfs01/bethge/home/oeberle/Scripts/deepgaze/SaliencyModels/test_model.py'  
    
    if args.stage == 'finetune':
        print(args.inds_dir)
        inds_test_dir = str(args.inds_dir).replace('train', 'test')
        print(inds_test_dir)
    elif args.stage == 'pretrain':
        inds_test_dir = '/gpfs01/bethge/home/oeberle/Results/MIT1003/all_1003_inds.p'

    
    
    
    cmd = ['python3',test_file,
                     '--save_dir', eval_dir, 
                     '--flag',args.flag,
                     '--epochs', '1',
                     '--ep', '{}'.format(ep),
                     '--im_ds', '{}'.format(args.im_ds),
                     '--inds_dir', inds_test_dir,
                     '--data_dir', args.data_dir,
                     '--fold', '{}'.format(args.fold),
                     '--cb_dir', args.cb_dir,
                     '--ckpt_dir', ckpt_dir,
                     '--gabor_dir', args.gabor_dir,
                     '--steer_dict', args.steer_dict,
                     '--filter_type',  args.filter_type,
                     '--bs', '{}'.format(args.bs),
                     '--stage', args.stage
                     ]
    
    print('TEST_cmd', cmd)
    
    cb_str = '_cb'
    eval_dir = os.path.join(eval_dir, args.flag + '_batch_{}{}_fold_{}_ep_{}.p'.format(args.bs, cb_str, args.fold, ep))
   
        
    if 'mlc' in args.flag.split('_'):
        cmd = cmd + ['--sigmas',  ' '.join(['{}'.format(sig) for sig in args.sigmas])]
    
    subprocess.call(cmd)
    
    return eval_dir
    

def flat2gen(alist):
    for item in alist:
        if isinstance(item, list):
            for subitem in item: yield subitem
        else:
            yield item
            
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

def prepare_batch(input_images,fixations_x, fixations_y, valid_ids, downscale = 2, vgg_prep=False):
    import itertools
    if vgg_prep==True:
        print('VGG preprocessing')
        im = vgg_preprocess(np.array(input_images)[valid_ids])
    else:
        im = (np.array(input_images)[valid_ids])/255.

   # x = list(itertools.chain(*np.reshape([[int((768./(downscale*1000.))*fix) for fix in fixations_x[n]] for n in valid_ids],-1)))        
    x = list(itertools.chain(*np.reshape([[int((1./downscale)*fix) for fix in fixations_x[n]] for n in valid_ids],-1)))        

    y = list(itertools.chain(*np.reshape([[int((1./downscale)*fix) for fix in fixations_y[n]] for n in valid_ids],-1))) 
    
    b = list(itertools.chain(*[int(i)*np.ones_like(fixations_x[n]) for i,n in enumerate(valid_ids)]))   
    return im, x, y, b

def prepare_batch_salicon(input_images,fixations_x, fixations_y, downscale = 2, vgg_prep=False):
    import itertools
    if vgg_prep==True:
        im = vgg_preprocess(np.array(input_images))
        #im = vgg_preprocess(np.asarray(input_images, np.float32))
    else:
        im = (np.array(input_images))/255.

    n_ims = len(input_images)    
    
    if n_ims > 1:
        x = list(itertools.chain(*np.reshape([[int((1./downscale)*fix) for fix in fixations_x[n]] for n in range(n_ims)],-1)))        
        y = list(itertools.chain(*np.reshape([[int((1./downscale)*fix) for fix in fixations_y[n]] for n in range(n_ims)],-1)))        
        b = list(itertools.chain(*[int(n)*np.ones_like(fixations_x[n]) for n  in range(n_ims)]))   
    elif n_ims == 1:
        
        x = np.reshape([[int((1./downscale)*fix) for fix in fixations_x[n]] for n in range(n_ims)],-1)        
        y = np.reshape([[int((1./downscale)*fix) for fix in fixations_y[n]] for n in range(n_ims)],-1)     
        b = np.zeros_like(fixations_x[0]).tolist()
    else:
        raise Error
        
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



def run_evaluation_on_train_graph(sess, model,  args, _centerbias_portrait, _centerbias_landscape):
    
    inds_dir = np.copy(args.inds_dir)
    
    if args.stage == 'finetune':
        print(args.inds_dir)
        inds_test_dir = str(args.inds_dir).replace('train', 'test')
        print(inds_test_dir)
    elif args.stage == 'pretrain':
        inds_test_dir = '/gpfs01/bethge/home/oeberle/Results/MIT1003/all_1003_inds.p'

       
    test_inds = get_inds(args, inds_test_dir) #pickle.load(open(inds_test_dir,'rb'))    
    eval_data_dir = '/gpfs01/bethge/home/oeberle/Results/MIT1003/MIT_{}.p'
    
    
    _dict = {'gamma': None, 'ckpt_dir': args.ckpt_dir, 'im_ds': args.im_ds,'ds': args.ds, 'sigmas':args.sigmas, 'inds_dir':inds_test_dir, 'cb_dir': args.cb_dir, 'steerdict':args.steer_dict, 'flag':args.flag,'lr': None, 'inds':test_inds,'sal_map': None, 'im':None, 'xinds':None,'yinds':None, 'binds': None, 'epoch_loss':None, 'run_loss': []} 
                         
    count = 0 
    for j in test_inds:
        print('batch_i',j)

        if args.flag == 'vgg':

            input_images,fixations_x, fixations_y = get_data(eval_data_dir, [j])
            im, x, y, b = prepare_batch_salicon(input_images,fixations_x, fixations_y, downscale = args.im_ds, vgg_prep = True)
        else:
            print('Normalization and -0.5 Preprocessing')

            input_images,fixations_x, fixations_y = get_data(eval_data_dir, [j])
            im, x, y, b = prepare_batch_salicon(input_images,fixations_x, fixations_y, downscale = args.im_ds,vgg_prep = False)
            im = im - 0.5
                        
        if im.shape[1]>im.shape[2]:
            print('PORTRAIT')
            _centerbias = _centerbias_portrait[0,:,:][np.newaxis,:,:]

            #landscape
        elif im.shape[1]<im.shape[2]:
            print('LANDSCAPE')
            _centerbias = _centerbias_landscape[0,:,:][np.newaxis,:,:]
        else:
            raise Error

        init_feed_dict = {
                model.input_tensor: im,
                model.centerbias: _centerbias, # np.ones([bs, 384, 512]),
                model.dg_loss['deep_gaze_loss/x_inds']: x,#np.array(xinds),
                model.dg_loss['deep_gaze_loss/y_inds']: y,#np.array(yinds),
                model.dg_loss['deep_gaze_loss/b_inds']: b,
            }     

        eval_loss = sess.run(model.dg_mean, init_feed_dict)


        _dict['run_loss'].append(eval_loss)
        print('EVAL', j,eval_loss)

        if count == 10:
            a = np.exp(sess.run(model.end_points3['saliency_map/log_density'], init_feed_dict))
            _dict['sal_map'] = a.squeeze()
            _dict['im'] = im
            _dict['xinds'] = x
            _dict['yinds'] = y
            _dict['binds'] = b

    
        _dict['epoch_loss'] = np.mean(_dict['run_loss'])
        #pickle.dump(_dict, open(eval_dict_str, 'wb'))
        count = count + 1
        
    return _dict
    #cb_str = '_cb'
    #eval_dir = os.path.join(eval_dir, args.flag + '_batch_{}{}_fold_{}_ep_{}.p'.format(args.bs, cb_str, args.fold, ep))
   
        
    #  if 'mlc' in args.flag.split('_'):
    #      cmd = cmd + ['--sigmas',  ' '.join(['{}'.format(sig) for sig in args.sigmas])]








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
    parser.add_argument('--ep_max', type=int, default=400,
                        help='number of epochs')
    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels to be used, default 3')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='learning rate')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='learning rate decay')
    parser.add_argument('--cb_dir', type=str, default='',
                        help='Centerbias dir, if provided it is used')
    parser.add_argument('--sigmas', type=int, nargs='+', default = None,
                       help='Provide sigmas for MLC features')
    parser.add_argument('--ds', type=int, default=2,
                       help='Downsample factor, ds=1 means no downsampling.')
    parser.add_argument('--data_dir', type=str,
                       help='file to get image and fixation data from')
    parser.add_argument('--inds_dir', type=str,
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
    parser.add_argument('--filter_type', type=str, default='optimized',
                        help='Choose what SP filters to use: optimized, gabor or gabor_param')
    parser.add_argument('--n_bands', type=int, default=4,
                        help='Number of orientations')
    parser.add_argument('--decay', type=str, default='exp',
                        help='decay for learning rate, exp or step')
    parser.add_argument('--gabor_dir', type=str,
                        help='Get gabor dict to restore filters from.')
    parser.add_argument('--upsample', type=str, default='repeat',
                        help='Upsample method: repeat or resize')
    parser.add_argument('--gabor_loss', type=bool, default=False,
                        help='Apply gabor loss')
    parser.add_argument('--stage', type=str,
                        help='pretrain or finetune')
    parser.add_argument('--ik_features', type=str, default = 'ik_features',
                        help='ik_features, raw_features or conspicuity')
    parser.add_argument('--exclude_inds', type=int, nargs='+', default = None,
                        help='specify indices of samples you want to exclude')

    args = parser.parse_args()
    
   
    set_up_dir(args.save_dir)
    
    train(args)
    
def train(args):
   
    bs = args.bs

    # Get Image data
    #input_images, fixations_x, fixations_y, valid_inds = get_saliency_data(args.data_dir, args.inds_dir)
    image_dummy, _, _ = get_data(args.data_dir,[0,1])
    
    valid_inds = get_inds(args, args.inds_dir)
    
    height, width = np.shape(image_dummy[0])[0], np.shape(image_dummy[0])[1]
    
    
    # Preparing centerbias

    if len(args.cb_dir) > 0:
        print('Using centerbias')
        _centerbias_landscape = pickle.load(open(args.cb_dir.format(384,512), 'rb'), encoding ='latin1')
        _centerbias_landscape = np.repeat(_centerbias_landscape[np.newaxis,:,:], args.bs, axis = 0)
        
        _centerbias_portrait = pickle.load(open(args.cb_dir.format(512,384), 'rb'), encoding ='latin1')
        _centerbias_portrait = np.repeat(_centerbias_portrait[np.newaxis,:,:], args.bs, axis = 0)
        
        
        salicon_cb_dir = '/gpfs01/bethge/home/oeberle/Results/dg_on_salicon/centerbias_BaselineModel_240_320.p'
        _centerbias_salicon = pickle.load(open(salicon_cb_dir, 'rb'), encoding ='latin1')
        _centerbias_salicon = np.repeat(_centerbias_salicon[np.newaxis,:,:], args.bs, axis = 0)
        cb_str = '_cb'

        
        
    else:
        print('Using no/uniform bias')
        _centerbias = np.ones([bs, int(height/args.im_ds),  int(width/args.im_ds)])
        
        cb_str = '_no_cb'
        
    save_dir = os.path.join(args.save_dir, args.flag + args.save_flag + cb_str + '_' + args.stage, 'train/fold_{}/'.format(args.fold))   
    set_up_dir(save_dir)
    
    eval_dir = os.path.join(save_dir,'eval/')
    set_up_dir(eval_dir)
    
    checkdir =  os.path.join(save_dir, 'Checkpoints/')
    set_up_dir(checkdir)
    
    filter_dir = os.path.join(save_dir, 'Filters/')
    set_up_dir(filter_dir)
    
    save_str = os.path.join(save_dir, args.flag + '_lr_{}_batch_{}{}_fold_{}.p'.format(args.lr,args.bs, cb_str, args.fold))
    eval_str = os.path.join(eval_dir, args.flag + '_batch_{}{}_fold_{}.p'.format(args.bs, cb_str, args.fold))

    
    n_ims = len(valid_inds)
    inds = list(range(n_ims))
    batch_ids = list([inds[(k*bs):((k+1)*bs)] for k in range(int(np.floor(n_ims/bs)))])
            
    
    print('Image shape: {}x{}'.format(height, width))
    print('save_dir', save_dir)
    tf.reset_default_graph()
    g = tf.Graph()
    


    
    with g.as_default():
        model = Model(args,height,width)

        f = open(save_dir + 'trained_vars.txt','w')
        title =  [f.write(var + '\n') for var in model.trained_params]
        f.close()

        f = open(save_dir + 'all_vars.txt','w')
        title =  [f.write(var.name + '\n') for var in tf.all_variables()]
        f.close()
        
        
        eval_txt = open(save_dir + 'eval_list.txt', mode='a')
        
        
        with tf.Session() as test_sess:   
            saver = tf.train.Saver(max_to_keep=20, write_version = saver_pb2.SaverDef.V1)

            
            #print('A',[v.name for v in tf.GraphKeys.VARIABLES])
            #print('B',[v.name for v in tf.GraphKeys.GLOBAL_VARIABLES])
            
            
            
            #Use this for reusing gabor_params filters that now should be trained fully
            if False:
                print([v.name for v in  tf.get_collection(tf.GraphKeys.VARIABLES)])
                restore_vars=[]
                for sc in ['saliency_map/alpha', 'saliency_map/sigma', 'global_step', 'beta1_power', 'beta2_power','readout_network', 'SteerablePyramid/gabor']:
                    restore_vars.append(tf.get_collection(tf.GraphKeys.VARIABLES, scope=sc))

                restore_vars = list(flat2gen(restore_vars))
                print('RESTORE',restore_vars)
                saver = tf.train.Saver(restore_vars, write_version = saver_pb2.SaverDef.V1)
            
            if len(args.ckpt_dir) > 0:
                print('Restore model from checkpoint')
                print([v.name for v in  tf.get_collection(tf.GraphKeys.VARIABLES)])
                
                saver.restore(test_sess, args.ckpt_dir  )
                
                print('IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIINIIIIIIIIIIIIIIIIIIIIIIIIIT FILTERS!!!!!!!!!!!!!!!!!!!!')
                if args.flag == 'steer':
                    #model.A.init_filters(test_sess)
                
                    #This I used for retraining gabor_joint on MIT
                    
                    model.A.init_hp_lp_filters(test_sess)
                    #model.A.init_orients_from_pretrained(test_sess)
                
                #print([v.name for v in  tf.get_collection(tf.GraphKeys.VARIABLES)])
                #raise
            else:
            # Initialize deepgaze
                print('Initialize all variables and model')
                test_sess.run(tf.initialize_all_variables())

                if args.flag == 'vgg' or args.flag == 'mlc_vgg' :
                    print('Initialize VGG')
                    initialize_deep_gaze(g, test_sess)
                elif args.flag == 'mlc' or args.flag == 'ittikoch' or args.flag == 'ittikoch_mlc' or args.flag==  'centerbias_only' :
                    print('Not initialization for mlc')
                elif args.flag == 'steer' or args.flag == 'steer_mlc' :
                    if args.filter_type == 'optimized' or args.filter_type ==  'gabor' or args.filter_type == 'optimized_init':
                        print('INIT GABORS')
                        model.A.init_filters(test_sess)
                    elif args.filter_type == 'gabor_param' or args.filter_type == 'gabor_joint':
                        model.A.init_hp_lp_filters(test_sess)
                        print('Try to not init')
                    else:
                        raise Error
                    
                else:
                    raise Error

                    #print('No initialization')
            if False:        
                uninitialized_vars = []
                for var in tf.all_variables():
                    try:
                        test_sess.run(var)
                    except tf.errors.FailedPreconditionError:
                        uninitialized_vars.append(var)
    #
                init_new_vars_op = tf.initialize_variables(uninitialized_vars)

                print('UNINITIALIZED VARIABLES', init_new_vars_op)
                test_sess.run(init_new_vars_op)
            
            res_dict = {k: {'upsample': args.upsample,'gamma': args.gamma, 'ckpt_dir': args.ckpt_dir, 'im_ds': args.im_ds,'ds': args.ds, 'sigmas':args.sigmas, 'inds_dir': args.inds_dir,'cb_dir_salicon':salicon_cb_dir, 'cb_dir_mit': args.cb_dir, 'steerdict':args.steer_dict, 'flag':args.flag,'lr': args.lr, 'inds':valid_inds,'sal_map': None, 'im':None, 'xinds':None,'yinds':None, 'binds': None, 'epoch_loss':None, 'run_loss': [], 'joint_loss':[]} for k in range(args.ep_max)}
            
            
           # try:
           #     res_dict_before = pickle.load(open(save_str, 'rb'))
           #     res_dict.update(res_dict_before)
           #     #print(res_dict, len(res_dict_before))
           #     #pickle.dump(res_dict, open(save_str.replace('.p', 'TEST.p'), 'wb'))
           #     # raise
           # except IOError:
           #     print('No before dict found')
            #raise
            
            
            
            if args.filter_type == 'gabor_param' or args.filter_type == 'gabor_joint' :
                params_dict = {'theta':[], 'lambda':[], 'phi': [], 'sigma':[], 'gamma': []}
                [res_d.update(params_dict) for k,res_d in res_dict.items()]
                
            #pickle.dump(res_dict, open(save_str, 'wb'))

            
            curr_lr = args.lr
            eval_loss = []
            eval_dict = {}
            
            #eval_dict = pickle.load(open(eval_str,'rb'))
            #[eval_loss.append(get_loss(eval_dict, i = k)) for k in range(20)]
            
            #for k in range(args.epochs):
            stop_criterion = False
            k = 0
            
            #batch_ids = batch_ids[:10]
            while stop_criterion == False:    
                
                print('epoch:', k)

                #ckpt_dir = checkdir + '/model_{}.ckpt'.format(k)
                #run_evaluation(args, eval_dir, ckpt_dir, k)
                for i,j in enumerate(batch_ids):
                    print('batch_i',j)
                    
                    if args.flag == 'vgg':
                        
                        input_images,fixations_x, fixations_y = get_data(args.data_dir, list(j))
                        im, x, y, b = prepare_batch_salicon(input_images,fixations_x, fixations_y, downscale = args.im_ds, vgg_prep = True)
                    else:
                        print('Normalization and -0.5 Preprocessing')

                        input_images,fixations_x, fixations_y = get_data(args.data_dir, list(j))
                        im, x, y, b = prepare_batch_salicon(input_images,fixations_x, fixations_y, downscale = args.im_ds,vgg_prep = False)
                        im = im - 0.5
                        
                    print(np.shape(im))
                    # portrait
                    if np.shape(im)[1]>np.shape(im)[2] and args.stage == 'finetune':
                        print('PORTRAIT')
                        _centerbias = _centerbias_portrait
                    elif np.shape(im)[1]<np.shape(im)[2] and args.stage == 'finetune':
                        print('LANDSCAPE')
                        _centerbias = _centerbias_landscape
                    elif args.stage == 'pretrain':
                        print('SALICON')
                        _centerbias = _centerbias_salicon
                    else:
                        raise Error

                    
                    init_feed_dict = {

                                    model.input_tensor: im,
                                    model.centerbias: _centerbias, # np.ones([bs, 384, 512]),
                                    model.dg_loss['deep_gaze_loss/x_inds']: x,#np.array(xinds),
                                    model.dg_loss['deep_gaze_loss/y_inds']: y,#np.array(yinds),
                                    model.dg_loss['deep_gaze_loss/b_inds']: b,
                                    model.lr: curr_lr,
                        

                                      }   
                    #print(init_feed_dict)
                    #raise
                    #up_old = test_sess.run(model.up_scaled_inputs_old, init_feed_dict)
                    #up = test_sess.run(model.up_scaled_inputs, init_feed_dict)
                    #print(np.allclose(up_old, up))
                    #print(np.shape(up_old), np.shape(up), type(up_old), type(up)) 
                    #raise
                    
                    
                   # if args.flag == 'vgg':
                   #     init_feed_dict[model.height_vgg] = im.shape[1]
                   #     init_feed_dict[model.width_vgg] = im.shape[2]
                        
                    print('KKKKKKKKKK',im.shape,_centerbias.shape)
            
                    loss = test_sess.run(model.train_op, init_feed_dict)
                    
                    if args.filter_type == 'gabor_joint':
                        pure_loss = loss - test_sess.run(model.end_points3['costs/gabor_loss'])
                        res_dict[k]['run_loss'].append(pure_loss)
                        res_dict[k]['joint_loss'].append(loss)
                        print(args.flag, k,i,loss,pure_loss,test_sess.run(model.lr, init_feed_dict))

                    else:                                   
                        res_dict[k]['run_loss'].append(loss)
                        print(args.flag,k,i,loss,test_sess.run(model.lr, init_feed_dict))

                    res_dict[k]['lr'] = test_sess.run(model.lr, init_feed_dict)

#                    if args.filter_type == 'gabor_param':
#                        {res_dict[k][key.split('/')[2][:-2]].append(test_sess.run(slim.get_variables(key[:-2]))) for key in ##['SteerablePyramid/gabor/theta:0', 'SteerablePyramid/gabor/lambda:0', 'SteerablePyramid/gabor/phi:0', 'SteerablePyramid/gabor/sigma:0', 'SteerablePyramid/gabor/gamma:0'] }
                                     
                                     
                    #return test_sess.run(up_scaled_inputs, init_feed_dict)
          
        
                    if i %10 == 0 and args.train_sp == True:
                        
                
                
                        print('MODEL_KEYS',model.A.filters.keys())
                        print(save_dir)
                        #keys = ['SteerablePyramid/W_hp','SteerablePyramid/W_lp',
                        keys = ['SteerablePyramid/W_hp','SteerablePyramid/W_lp'] + ['SteerablePyramid/W_orients_{}_imag'.format(i) for i in range(args.n_bands) ] + ['SteerablePyramid/W_orients_{}_real'.format(i) for i in range(args.n_bands) ] 

                        filt_dict = {k:test_sess.run(k) for k in model.trained_SP_params}
                                                         
                        if args.filter_type == 'gabor_joint':
                                                         
                            gabor_params_filt = {k: test_sess.run(v) for k,v in model.end_points3.items() if k.startswith('GaborParam')}
                            filt_dict.update(gabor_params_filt)
                        
                        #For training on gabor params
                        filt_dict.update({k + ':0':test_sess.run(model.A.filters[k]) for k in keys})
                        
                        #filt_dict.update({k + ':0':model.A.filters[k ] for k in keys})
                        
                        pickle.dump(filt_dict, open(filter_dir +'filt_ep_{}_batch_{}.p'.format(k,i), 'wb'))
                        
                        
                  
                        

                    if False:
                        if k == 0 and i ==0:
                            a = np.exp(test_sess.run(model.end_points3['saliency_map/log_density'], init_feed_dict))
                            xinds = list(np.array(x)[np.array(b) == 0])
                            yinds = list(np.array(y)[np.array(b) == 0])
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
                if args.decay == 'exp':
                    curr_lr = curr_lr * args.gamma 
                    
                #For training gabor params on SALICON  
                if curr_lr <= 0.001:
                    curr_lr = 0.001
               
                
                
                ckpt_dir = os.path.join( checkdir , 'model_{}.ckpt'.format(k))
  
                    
                res_dict[k]['epoch_loss'] = np.mean(res_dict[k]['run_loss'])
                pickle.dump(res_dict, open(save_str, 'wb'))
                
                # Save the variables to disk.
                save_path = saver.save(test_sess,ckpt_dir)
                print("Model saved in file: %s" % save_path)
                
                # Evaluate
                
                
               
                eval_dict[k] = run_evaluation_on_train_graph(test_sess, model, args, _centerbias_portrait, _centerbias_landscape)

                pickle.dump(eval_dict, open(eval_str, 'wb'))

                # Compute evaluation loss and compare
                eval_loss.append(get_loss(eval_dict, k))
                    
                eval_txt.write(str(k) + ',    ' + str(eval_loss) + '\n')
                eval_txt.flush()
                print('EVAL_LOSS', eval_loss)
                if len(eval_loss) > 5:
                    last_3 = np.array(eval_loss[-3:])

                    if (last_3 < eval_loss[-5]).all() or k == args.ep_max:
                        if k > args.epochs:
                            print('STOP CRITERION FULFILLED')
                            stop_criterion = True
                            sys.exit()

                k = k + 1

    pickle.dump(res_dict, open(save_str, 'wb'))
        
        
    
if __name__ == '__main__':
    main()
    print('DGII Training')
