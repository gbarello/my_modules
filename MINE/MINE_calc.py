import MINE_net
import tensorflow

def calc_MINE(data1,data2,vdata1,vdata2,nbatch = 1000,nstep = 2000,PRINT = True):

    tf.reset_default_graph()

    IN1 = tf.placeholder(tf.float32,shape = [nbatch,int(data1.shape[-1])])
    IN2 = tf.placeholder(tf.float32,shape = [nbatch,int(data2.shape[-1])])

    MI = MINE.get_MINE(IN1,IN2)

    MI_up = tf.train.AdamOptimizer().minimize(-MI)

    init_op = tf.global_variables_initializer()
    
    sess = tf.Session()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess = sess)
    
    sess.run(init_op)

    mr = 0.
    vmr = 0.
    lr = .05
    
    for time in range(nstep):

        batch,lab = get_batch(data1,data2,nbatch)
        vbatch,vlab = get_batch(vdata1,vdata2,nbatch)
            
        mine,_ = sess.run([MI,MI_up],{IN:batch,OUT:lab})
        vmine = sess.run(MI,{IN:vbatch,OUT:vlab})

        mr = (1. - lr)*mr + lr*mine
        vmr = (1. - lr)*vmr + lr*vmine

        if PRINT:
            print("{}\t{}\t{}".format(time,round(mr,3),round(vmr,3)))
        
    sess.close()

    return mr,vmr

def get_batch(d,l,n):
    ii = np.random.choice(np.arange(len(d)),n)
    
    return d[ii],l[ii]
