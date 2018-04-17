import MINE_net as MINE

import tensorflow as tf
import numpy as np

def get_batch(d,l,n):
    ii = np.random.choice(np.arange(len(d)),n)
    
    return d[ii],l[ii]

def get_MINE_variable(data,label,vdata,vlabel,ndim,nbatch = 1000,PRINT = True):

    tf.reset_default_graph()

    vinit = np.float32(np.random.randn(ndim,int(data.shape[-1])))
    
    dirs = tf.Variable(vinit)
    dnorm = dirs / tf.sqrt(tf.reshape(tf.reduce_sum(dirs*dirs,axis = 1),[-1,1]))

    IN = tf.placeholder(tf.float32,shape = [nbatch,int(data.shape[-1])])
    
    INvar = tf.tensordot(IN,dnorm,axes = [[1],[1]])
    OUT = tf.placeholder(tf.float32,shape = [nbatch,int(label.shape[-1])])

    MI = MINE.get_MINE(INvar,OUT)

    MI_up = tf.train.AdamOptimizer().minimize(-MI)

    init_op = tf.global_variables_initializer()
    
    sess = tf.Session()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess = sess)
    
    sess.run(init_op)

    mr = 0.
    vmr = 0.
    lr = .05
    
    for time in range(3000):

        batch,lab = get_batch(data,label,nbatch)
        vbatch,vlab = get_batch(vdata,vlabel,nbatch)
            
        mine,_ = sess.run([MI,MI_up],{IN:batch,OUT:lab})
        vmine = sess.run(MI,{IN:vbatch,OUT:vlab})

        mr = (1. - lr)*mr + lr*mine
        vmr = (1. - lr)*vmr + lr*vmine
        
        print("{}\t{}\t{}".format(time,round(mr,3),round(vmr,3)))
        
    VAR = sess.run(dnorm)
    sess.close()

    return VAR,mr
