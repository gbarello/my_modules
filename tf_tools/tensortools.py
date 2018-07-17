import tensorflow as tf
import numpy as np

def make_orthogonal_vectors(num,dim,dtype = tf.float32,tag = "othovec"):

    vecs = [tf.get_variable(tag + "_{}".format(k),[dim],dtype = dtype) for k in range(num)]

    ovec = []

    for k in range(len(vecs)):
        temp = vecs[k]

        for j in range(len(ovec)):
            temp = temp - tf.tensordot(temp,ovec[j],axes = [[0],[0]])*ovec[j]/tf.tensordot(ovec[j],ovec[j],axes = [[0],[0]])

        ovec.append(temp)        

    return ovec

def test_make_orthogonal_vectors(nvec = 10,ndim = 10,dtype = tf.float64,tag = "poodle"):
    
    VV = make_orthogonal_vectors(nvec,ndim,dtype = dtype,tag = tag)

    assert len(VV) == nvec

    for v in VV:
        print(v.shape)
        assert v.shape == [ndim]

    test = []
    tlen = []
    for k in range(len(VV)):
        tlen.append(tf.sqrt(tf.reduce_sum(tf.tensordot(VV[k],VV[k],axes=[[0],[0]]))))
        for j in range(k+1,len(VV)):
            test.append(tf.tensordot(VV[k],VV[j],axes =[[0],[0]]))

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    dots = sess.run(test)
    lens = sess.run(tlen)

    for k in dots:
        assert np.isclose(k,0)
    for k in lens:
        print(k)
        
        
    sess.close()
if __name__ == "__main__":
    with tf.device("/device:GPU:0"):
        test_make_orthogonal_vectors()
