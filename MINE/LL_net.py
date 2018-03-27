import tensorflow as tf

def get_MINE(inP,inD,nlayer = 4,tag = "LL",noise_1 = 0,noise_2 = 0):

    P = inP
    O = inD

    P = P + noise_1 * tf.random_normal(P.shape)
    O = O + noise_2 * tf.random_normal(O.shape)

    P = tf.reshape(P,[int(P.shape[0]),1,-1])
    O = tf.reshape(O,[int(O.shape[0]),1,-1])

    JOINT = tf.concat([P,O],1)

    IN = tf.reshape(JOINT,[-1,int(JOINT.shape[2])])    

    KLnet = IN
    
    for k in range(nlayer):
        KLnet = tf.layers.dense(KLnet,64,name = "{}_kl_layer_{}".format(tag,k),activation = tf.nn.sigmoid)
        KLnet = tf.layers.dropout(KLnet,name = "{}_kl_drop_{}".format(tag,k))

    KLnet = tf.layers.dense(KLnet,1,name = "{}_kl_layer_final".format(tag))
    KLnet = tf.reshape(KLnet,[-1,2])

    Enet = IN
    
    for k in range(nlayer):
        Enet = tf.layers.dense(Enet,64,name = "{}_E_layer_{}".format(tag,k),activation = tf.nn.sigmoid)
        Enet = tf.layers.dropout(Enet,name = "{}_E_drop_{}".format(tag,k))

    Enet = tf.layers.dense(Enet,1,name = "{}_E_layer_final".format(tag))
    Enet = tf.reshape(Enet,[-1,2])
    
    KL = tf.reduce_mean(KLnet[:,0]) - tf.log(tf.reduce_mean(tf.exp(KLnet[:,1])))
    E = tf.reduce_mean(Enet[:,0]) - tf.log(tf.reduce_mean(tf.exp(Enet[:,1])))
#    MINE = tf.reduce_mean(net[:,0]) - tf.reduce_mean(tf.exp(net[:,1] - 1))

    return MINE
