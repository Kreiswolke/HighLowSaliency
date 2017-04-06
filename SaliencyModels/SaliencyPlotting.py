import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_comparison(res_dict, ep = 14):
    
    
    def get_loss(res_dict, i = -1):
        
        if res_dict[0]['flag'] =='mlc':
            print('Using missing mlc losses', len(res_dict))
            i = 19
            h,w = np.shape(res_dict[i]['sal_map'])[1],np.shape(res_dict[i]['sal_map'])[2]
            print('get_loss: HxW: {} X {}'.format(h,w)) 

            loss = -(np.mean(res_dict[i]['run_loss']) - np.log((h*w)))/np.log(2.) 

        else: 
            h,w = np.shape(res_dict[i]['sal_map'])[1],np.shape(res_dict[i]['sal_map'])[2]
            print('get_loss: HxW: {} X {}'.format(h,w)) 

            loss = -(np.mean(res_dict[i]['run_loss']) - np.log((h*w)))/np.log(2.) 

                                                                                   
        return loss
    
    
    exps =  [v['title'] for k,v in res_dict.items()]
    colors = [v['color'] for k,v in res_dict.items()]


    N = len(exps)
    means = [get_loss(pickle.load(open(v['dir'], 'rb')),ep) for k,v in res_dict.items()]
    #std = [np.std((acc_last[e])) for e in exps]
    ind = np.arange(1,N*2,2)  # the x locations for the groups
    width = 1.0      # the width of the bars
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_ylabel('bit/fix', fontsize = 17)
    ax.set_title('Saliency prediction performance', fontsize = 17)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    #ax.set_ylim([0.6,1.1])
    ax.set_xlim([1.0,N*2+1])
    X = ind + width + 0.05
    ax.set_xticks(X)
    #ax.set_yticks(np.arange(0.6,1.05,0.05))
    ax.set_xticklabels(exps, fontsize = 15)


    rects1 = ax.bar(ind + width, means, width, color=colors)#, yerr=std)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '%0.3f' % height,
                    ha='center', va='bottom', fontsize = 13)

    autolabel(rects1)
    def label_diff(i,j,text,X,Y, scaleytext,arm):
        x = (X[i]+X[j])/2
        y = 1.04*max(Y[i], Y[j])
        dx = abs(X[i]-X[j])
        print(dx)

        props = {'connectionstyle':'bar','arrowstyle':'-',
                     'shrinkA':arm,'shrinkB':arm,'lw':1}
        ax.annotate(text, xy=(x,scaleytext*y), zorder=10, ha = 'center', size=12)
        ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props, ha = 'center')

    #label_diff(0,4,'p={}'.format(float(p_4)),X, means, 1.08,60)
    #label_diff(4,5,'p={:6f}'.format(float(p_5)),X, means,1.04,8)
    #ax.spines['bottom'].set_position(('axes'))

    ax.spines['left'].set_position(('axes', -0.05))
    plt.show()
    
    
    
def normalize_log_density(log_density):
    """ convertes a log density into a map of the cummulative distribution function.
    """
    density = np.exp(log_density)
    flat_density = density.flatten()
    inds = flat_density.argsort()[::-1]
    sorted_density = flat_density[inds]
    cummulative = np.cumsum(sorted_density)
    unsorted_cummulative = cummulative[np.argsort(inds)]
    return unsorted_cummulative.reshape(log_density.shape)

def visualize_distribution(log_densities, ax = None):
    if ax is None:
        ax = plt.gca()
    t = normalize_log_density(log_densities)
    ax.imshow(t, cmap=plt.cm.viridis)
    levels = levels=[0, 0.25, 0.5, 0.75, 1.0]
    cs = ax.contour(t, levels=levels, colors='black', linewidths=0.5)
    
    
def get_plot_dict(res_dict, str_bias):
    flag = res_dict[0]['flag']
    plot_dict = {'flag': flag,
                 'linestyle':None,
                 'color': None,
                 'label': flag + ', lr: {}'.format(res_dict[0]['lr']) + ', ' + str_bias,
                 'marker' : 'x',
                 'linewidth': 3,
                  }
    
    if flag == 'vgg':
        plot_dict['linestyle'] = '--'
        plot_dict['color']= next(c_vgg)
        
    
    elif flag == 'steer':
        plot_dict['linestyle'] = '-'
        plot_dict['color']=next(c_steer)

        
    elif flag == 'centerbias_only':
        plot_dict['linestyle'] = ':'
        plot_dict['color']= '#000000'
    
    return plot_dict



def plot_training(c_steer, c_vgg, dict_of_res_dirs, l = 1,  n_epochs = 12, tix_min=0.1,tix_max=1.8, _max=None):
    
    if _max == None:
        _max = n_epochs
    def get_plot_dict(res_dict, str_bias):
        flag = res_dict[0]['flag']
        plot_dict = {'flag': flag,
                     'linestyle':None,
                     'color': None,
                     'label': str_bias,
                     'marker' : 'x',
                     'linewidth': 3,
                      }

        if flag == 'vgg':
            plot_dict['linestyle'] = '--'
            plot_dict['color']= next(c_vgg)


        elif flag == 'steer':
            plot_dict['linestyle'] = '-'
            plot_dict['color']=next(c_steer)


        elif flag == 'centerbias_only':
            plot_dict['linestyle'] = ':'
            plot_dict['color']= '#000000'

        return plot_dict
    


    fig,ax = plt.subplots(1,1,figsize=(14,6))
    
    for _,v in dict_of_res_dirs.items():
        print(v.keys())
        _dir = v['dir']
        title_ext = v['ext']
        res_dict= pickle.load(open(_dir, 'rb'))
        height, width  = np.shape(res_dict[0]['sal_map'][0,:,:])[0], np.shape(res_dict[0]['sal_map'][0,:,:])[1]
        print('HxW:',height, width )
            
        str_bias = v['title']
        
        
        propd = get_plot_dict(res_dict, str_bias)
        
        n_epochs = len([k for i,k in enumerate(res_dict.keys()) if res_dict[i]['run_loss']])
        
        #loss_list = np.array([np.mean(res_dict[i]['run_loss']) for i in range(n_epochs)])

        loss_list = np.array([-np.mean(res_dict[i]['run_loss'] - np.log((height*width)))/np.log(2.) for i in range(n_epochs)])
        
       
        plt.plot(np.array(loss_list), marker = propd['marker'], label = propd['label'], linestyle = propd['linestyle'], linewidth = propd['linewidth'], color= propd['color'])
  

       # plt.plot(list(itertools.chain(*[(res_dict[i]['run_loss']- np.log((512*384)))/np.log(2.) for i in range(10)])))
        #ax.set_yticks([np.min(loss_list), np.max(loss_list)])
        #ax.set_yticklabels(['{:0.2f}'.format(i) for i in [np.min(loss_list),  np.max(loss_list)]])
        tix = list(np.linspace(tix_min,tix_max,6))
        ax.set_yticks(tix)
        ax.set_yticklabels(['{:0.2f}'.format(i) for i in tix])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xlabel('epoch', fontsize = 20)
        plt.ylabel(r'loss bit/fix', fontsize = 20)#: $\frac{1}{N}\sum_{i} ^{N_{fix} }log\left(p\left(x_i,y_i/I\right)\right)$', fontsize = 16)
        
        plt.ylim([tix_min-0.2,tix_max])
        #plt.xlim([0,n_epochs])
        #plt.legend(fontsize=13)
        ax.legend(ncol=len(dict_of_res_dirs), loc='lower center',fontsize=13)
        ax.spines['left'].set_position(('axes', -0.05))

    plt.show()
    
    
   # res_img_dict= pickle.load(open('/gpfs01/bethge/home/oeberle/Results/dg_on_MIT/test/BLA/deepgaze_train_lr_0.01_batch_4_vggcb.p', 'rb'))
    b = res_dict[0]['binds']
    xinds = list(np.array(res_dict[0]['xinds'])[np.array(b) == l])
    yinds = list(np.array(res_dict[0]['yinds'])[np.array(b) == l])
    bs = 4
    im_ds = 1 #res_dict[2]['im_ds']
    
    
    
    fig,ax = plt.subplots(n_epochs,len(dict_of_res_dirs)+1,figsize=(14,22))
    
    for j in range(n_epochs):
        ax[j,0].imshow(res_dict[0]['im'][l,:,:]+200, alpha = 0.5)
        if j != 0:
            ax[j,0].scatter([im_ds*x for x in xinds], [im_ds*y for y in yinds], marker= 'x', color = '#F21A00')# '#00366d')
        ax[j,0].axis('off')
        
        print('Image shape', np.shape(res_dict[0]['im']))

    
    for j,v in dict_of_res_dirs.items():
        _dir = v['dir']
        res_dict= pickle.load(open(_dir, 'rb'))
        str_bias = v['title']
        
        for k,v in res_dict.items():
            
            bottom, height = .25, .5
            top = bottom + height
            
            im = ax[k,j+1].matshow(normalize_log_density(np.log(v['sal_map'][l,:,:])))
            visualize_distribution(np.log(v['sal_map'][l,:,:]), ax = ax[k,j+1])
            ax[k,j+1].axis('off')
            ax[k,j+1].scatter(xinds, yinds, marker= 'x', color = '#F21A00')# '#00366d')

            if k ==0:
                ax[k,j+1].set_title('epoch {},'.format(k) + '\n' + str_bias + '\n'+ 'lr: ' + '{}'.format(res_dict[0]['lr']), y = 1.1)
                if j == 0:
                    ax[k,j].text(-0.05, 0.5*(bottom+top), 'epoch {}'.format(k),
                                             horizontalalignment='right',
                                             verticalalignment='center',
                                             rotation='vertical',
                                             transform=ax[k,j].transAxes
                                             )
                
            elif j == 0:
                ax[k,j].text(-0.03, 0.5*(bottom+top), 'epoch {}'.format(k ),
                                 horizontalalignment='right',
                                 verticalalignment='center',
                                 rotation='vertical',
                                 transform=ax[k,j].transAxes
                                 )
                
          
            #if k==0 and (j+1) ==5:
                #fig.colorbar(im, ax=ax[k,j+1], fraction=0.035, pad=0.04)
 
    plt.subplots_adjust(wspace=0.04,hspace=0.1)
    plt.show()
    
    
