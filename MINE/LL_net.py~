import tensorflow as tf

def get_MINE(in1,in2,nlayer = 4,tag = "MINE",noise_1 = 0.,noise_2 = 0.):

    P = in1    
    O = in2
    M = tf.concat([in2[1:],in2[:1]],axis = 0)

    P = P + noise_1 * tf.random_normal(P.shape)

    O = O + noise_2 * tf.random_normal(O.shape)
    M = M + noise_2 * tf.random_normal(M.shape)

    JOINT = tf.reshape(tf.concat([P,O],1),[int(P.shape[0]),1,-1])
    MARGI = tf.reshape(tf.concat([P,M],1),[int(P.shape[0]),1,-1])

    IN = tf.concat([JOINT,MARGI],1)
    IN = tf.reshape(IN,[-1,int(IN.shape[-1])])

    net = IN
    
    for k in range(nlayer):
        net = tf.layers.dense(net,64,name = "{}_layer_{}".format(tag,k),activation = tf.nn.sigmoid)
        net = tf.layers.dropout(net,name = "{}_drop_{}".format(tag,k))

    net = tf.layers.dense(net,1,name = "{}_layer_final".format(tag))

    net = tf.reshape(net,[-1,2])
    
    MINE = tf.reduce_mean(net[:,0]) - tf.log(tf.reduce_mean(tf.exp(net[:,1])))
#    MINE = tf.reduce_mean(net[:,0]) - tf.reduce_mean(tf.exp(net[:,1] - 1))

    return MINE
