import tensorflow as tf

def get_pred_GRU_stuff(nunit,ninput,tag = "pGRU",reuse = False):

    '''

    defines predictive GRU variables for a given unit and input shape

    needs to make:
    - input weights (for gate and update)
    - recurrect weights (for gate and update)
    - predictive weights (for pred_update)
    - gate bias
    - recurrent bias

    '''

    with tf.variable_scope(tag,reuse = reuse):
        
        g_input = tf.get_variable("g_input",shape = [nunit,ninput])
        g_rec = tf.get_variable("g_rec",shape = [nunit,nunit])
        g_bias = tf.get_variable("g_bias",shape = [nunit])
        
        z_input = tf.get_variable("z_input",shape = [nunit,ninput])
        z_rec = tf.get_variable("z_rec",shape = [nunit,nunit])
        z_bias = tf.get_variable("z_bias",shape = [nunit])
    
        r_input = tf.get_variable("r_input",shape = [nunit,ninput])
        r_rec = tf.get_variable("r_rec",shape = [nunit,nunit])
        r_bias = tf.get_variable("r_bias",shape = [nunit])
        
        pred = tf.get_variable("pred",shape = [ninput,nunit])
        
    out = {
        "g_input":g_input,
        "g_rec":g_rec,
        "g_bias":g_bias,
        "z_input":z_input,
        "z_rec":z_rec,
        "z_bias":z_bias,
        "r_input":r_input,
        "r_rec":r_rec,
        "r_bias":r_bias,
        "pred":pred
    }

    return out

def apply_GRU_to_data(data,activity_init,gruD):

    def update(a,d):
        print(a.shape)
        print(d.shape)
        go = tf.sigmoid(tf.tensordot(d,gruD["g_input"],axes=[[1],[1]])#inputs
                       +
                       tf.tensordot(a,gruD["g_rec"],axes=[[1],[1]])#recurrent
                       +
                       gruD["g_bias"])#bias
        
        zo = tf.sigmoid(tf.tensordot(d,gruD["z_input"],axes=[[1],[1]])#inputs
                       +
                       tf.tensordot(a,gruD["z_rec"],axes=[[1],[1]])#recurrent
                       +
                       gruD["z_bias"])#bias

        p = tf.tensordot(a,gruD["pred"],axes = [[1],[1]])
        
        ao = (1. - zo)*a +  zo * tf.tanh(tf.tensordot(d - p,gruD["r_input"],axes=[[1],[1]])#inputs
                                         +
                                         tf.tensordot(go*a,gruD["r_rec"],axes=[[1],[1]])#recurrent
                                         +
                                         gruD["r_bias"])#bias

        return ao

    activations = tf.transpose(tf.scan(lambda a,x:update(a,x),tf.transpose(data,[1,0,2]),activity_init),[1,0,2])
    predictions = tf.tensordot(activations,gruD["pred"],axes = [[2],[1]])

    return tuple([activations,predictions])

def make_multilayer_pred_GRU(layers,input_tensor,initial_activity = None,tag = "ML_pGRU",copy = False):
    layer_size = [input_tensor.shape[-1].value]+layers
    
    GRUs = [get_pred_GRU_stuff(layer_size[k],layer_size[k-1],tag = tag + "_" +  str(k) + "_",reuse = copy) for k in range(1,len(layer_size))]

    GRU_outs = [(input_tensor,None)]

    if initial_activity == None:
        initial_activity = [tf.squeeze(tf.zeros([input_tensor.shape[0].value,k])) for k in layers]

    for k in range(len(GRUs)):
        GRU_outs.append(apply_GRU_to_data(GRU_outs[-1][0],initial_activity[k],GRUs[k]))

    return GRU_outs[1:],GRUs

if __name__ == "__main__":

    import numpy as np

    with tf.device("/cpu:0"):
        
        dplace = tf.placeholder(tf.float32,shape = [2,100,3])
        
        GRUac = make_multilayer_pred_GRU([10,11,12],dplace)
        
        init = tf.global_variables_initializer()

        sess = tf.Session()

        sess.run(init)

        out = sess.run(GRUac[-1][0],{dplace: np.float32(np.random.rand(2,100,3))})

        print(out.shape)
        
