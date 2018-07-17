import tensorflow as tf

import numpy as np

old_layers_val = [128,64,64,64]

def get_MINE(in1,in2,nlayer = [10,10],tag = "MINE",noise_1 = 0.,noise_2 = 0.,flatten_time_axis = False,getrawout = False,reuse = False,grad_timescale = 10,unbias_grad = False):

    P = in1    
    O = in2
    M = tf.concat([in2[1:],in2[:1]],axis = 0)#this takes the second var and rotates it in the batch dimention

    if flatten_time_axis:
        P = tf.reshape(P,[P.shape[0]*P.shape[1],-1])
        O = tf.reshape(O,[O.shape[0]*O.shape[1],-1])
        M = tf.reshape(M,[M.shape[0]*M.shape[1],-1])
        
    P = P + noise_1 * tf.random_normal(P.shape)
    O = O + noise_2 * tf.random_normal(O.shape)
    M = M + noise_2 * tf.random_normal(M.shape)

    JOINT = tf.reshape(tf.concat([P,O],1),[int(P.shape[0]),1,-1])
    MARGI = tf.reshape(tf.concat([P,M],1),[int(P.shape[0]),1,-1])

    IN = tf.concat([JOINT,MARGI],1)
    IN = tf.reshape(IN,[-1,int(IN.shape[-1])])

    net = IN
    
    for k in range(len(nlayer)):
        net = tf.layers.dense(net,nlayer[k],name = "{}_layer_{}".format(tag,k),activation = tf.nn.sigmoid,reuse = reuse)
        net = tf.layers.dropout(net,name = "{}_drop_{}".format(tag,k))

    net = tf.layers.dense(net,1,name = "{}_layer_final".format(tag),reuse = reuse)

    net = tf.reshape(net,[-1,2])

    MINE1 = tf.reduce_mean(net[:,0])

    if unbias_grad:
        try:
            @tf.RegisterGradient(tag + "CustomMINE_Log")
            def _custom_log_grad(op,x):
                '''normal gradient is x/op[0] (or the gradient of the input, divided by the input).'''
                inp = op.inputs[0]

                with tf.variable_scope("",reuse = tf.AUTO_REUSE):
                    unbiased_grad = tf.get_variable(tag + "unbias_MINE_grad",initializer = np.float32(1.))
                    ugrad_tracker = tf.get_variable(tag + "unbias_MINE_grad_track",initializer = False)

                if ugrad_tracker:
                    unbiased_grad = tf.assign(unbiased_grad,unbiased_grad * tf.exp(-1./grad_timescale) + (1. - tf.exp(-1./grad_timescale))*inp)
                else:
                    unbiased_grad = tf.assign(unbiased_grad,inp)
                    ugrad_tracker = tf.assign(ugrad_tracker,True)

                return inp/unbiased_grad
        except:
            print("already registered gradient")
        
        G = tf.get_default_graph()
        with G.gradient_override_map({"Log":tag + "CustomMINE_Log"}):
            MINE2 =  - tf.log(tf.reduce_mean(tf.exp(net[:,1])),name = "Log")

    else:
        MINE2 =  - tf.log(tf.reduce_mean(tf.exp(net[:,1])))

    MINE = MINE1 + MINE2
    
    if getrawout:
        return MINE,net
    else:
        return MINE

def get_MINE_grad_vars(in1,in2,nlayer = [10,10],tag = "MINE",noise_1 = 0.,noise_2 = 0.,flatten_time_axis = False,getrawout = False,reuse = False):

    P = in1    
    O = in2
    M = tf.concat([in2[1:],in2[:1]],axis = 0)#this takes the second var and rotates it in the batch dimention

    if flatten_time_axis:
        P = tf.reshape(P,[P.shape[0]*P.shape[1],-1])
        O = tf.reshape(O,[O.shape[0]*O.shape[1],-1])
        M = tf.reshape(M,[M.shape[0]*M.shape[1],-1])
        
    P = P + noise_1 * tf.random_normal(P.shape)
    O = O + noise_2 * tf.random_normal(O.shape)
    M = M + noise_2 * tf.random_normal(M.shape)

    JOINT = tf.reshape(tf.concat([P,O],1),[int(P.shape[0]),1,-1])
    MARGI = tf.reshape(tf.concat([P,M],1),[int(P.shape[0]),1,-1])

    temp = tf.random_uniform(JOINT.shape)

    IN = JOINT*temp + (1.-temp)*MARGI

    net = IN
    
    for k in range(len(nlayer)):
        net = tf.layers.dense(net,nlayer[k],name = "{}_layer_{}".format(tag,k),activation = tf.nn.sigmoid,reuse = reuse)
        net = tf.layers.dropout(net,name = "{}_drop_{}".format(tag,k))

    net = tf.layers.dense(net,1,name = "{}_layer_final".format(tag),reuse = reuse)

    net = tf.reshape(net,[-1,2])
    
    return net,IN
