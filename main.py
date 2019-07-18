from ista import *
from keras import backend as K
from keras.backend import clear_session

def train(name, Phi_input, loss_alpha = 1,loss_beta = 1,loss_gamma = 10, Training_inputs=None, classifier_trainable=False):
    """
    Parameters:
        name: Name of the experiment. This is used to name the log files and saved weights
        Phi_input: matrix of size (784, d). This is the forward model; it takes the 784 dimensional mnist image as input and produces a d-dimensional (real) output
        loss_alpha: weight on the reconstruction loss 
        loss_beta: weight on the symmetric loss (in ista-net, this loss enforces invertibility of the neural network layers)
        loss_gamma: weight on classification loss
        Training_inputs: (optional) Distorted measurements. If this argument is not given, then the inputs are computed automatically using Phi_input.
        classifier_trainable: (optional) Toggles whether the classifier being used should be trained at the same time as ista-net.

    """
   # tf.reset_default_graph()
    clean_images = x_train.reshape([-1,784])
    print(clean_images.shape)
    if Training_inputs is None:
        X0, X_input, X_output, PhiTPhi, PhiTb = make_matrices(clean_images, Phi_input, clean_images.dot(Phi_input))
    else: # Training inputs in kspace
        #clean_images = clean_images[0:num_train,:]
        clean_images = clean_images[0:nrtrain,:]
        print(clean_images.shape)
        #Training_inputs = Training_inputs.reshape([-1,784])
        Training_inputs = Training_inputs.reshape([-1,784*2])
        print(Training_inputs.shape)
        X0, X_input, X_output, PhiTPhi, PhiTb = make_matrices(clean_images, Phi_input, Training_inputs)       
   
    with tf.device('/device:GPU:0'):
        [Prediction, Pre_symetric] = inference_ista(X0, PhaseNumber, X_output=X_output, PhiTPhi=PhiTPhi, PhiTb=PhiTb, reuse=False)

        model = load_model(trainable=classifier_trainable)

        [cost, cost_sym] = compute_cost(Prediction, X_output, PhaseNumber, Pre_symetric)
        reshaped_predictions = [tf.reshape(p, [-1, 28,28,1]) for p in Prediction]
        digit_pred = reshaped_predictions[-1]
        X_output_im = tf.reshape(X_output, [-1, 28,28,1])
        for i in range(len(reshaped_predictions)):
            p = reshaped_predictions[i]
            tf.summary.image('Intermediate_'+str(i), p)
        tf.summary.image('Predicted_image', digit_pred)     
        tf.summary.image('Ideal_image', X_output_im)
     
        classified = model(digit_pred)
        #tf.summary.tensor_summary("Classifier_Prediction", classified)
        Y_label = tf.placeholder(tf.float32, shape=(None, 10))
        classification_loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(Y_label, classified))
        class_accuracy =tf.reduce_mean(1-(tf.reduce_sum(tf.abs(Y_label-classified), axis=-1)/2)) # tf.metrics.accuracy(Y_label, classified)
        # confusion_mat = tf.confusion_matrix(Y_label, classified)
        cost_all = loss_alpha*cost + loss_beta*cost_sym + loss_gamma*classification_loss
        tf.summary.scalar('Reconstruction_Loss', cost)
        tf.summary.scalar('Symmetric_Loss', cost_sym)
        tf.summary.scalar('Classification_Cross_Entropy_loss', classification_loss)
        tf.summary.scalar('Prediction Accuracy', class_accuracy)

        print("...............................")
        model_dir = name+'HALF_Phase_%d_ratio_0_ISTA_Net_plus_Model' % (PhaseNumber)
        output_file_name = "Log_output_%s.txt" % (model_dir)
        print("Model Directory:", model_dir)
        print("...............................")
        print("Log file:",output_file_name)
        print("...............................")
        print("Tensorboard Directory:",model_dir+'/train')
        print("...............................")
        print("Phase Number is %d" % (PhaseNumber))
        print("...............................\n")
        
    with tf.variable_scope("opt", reuse=tf.AUTO_REUSE):
        optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(model_dir + '/train',
                                          sess.graph)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    init = tf.global_variables_initializer()
    sess.run(init)
    model.load_weights('mnist_classifier_weights.hdf5')

    print("...............................")
    print("Phase Number is %d" % (PhaseNumber))
    print("...............................\n")



    model_dir = name+'HALF_Phase_%d_ratio_0_ISTA_Net_plus_Model' % (PhaseNumber)
    # model_dir = 'Phase_%d_ratio_0_ISTA_Net_plus_Model' % (PhaseNumber)

    output_file_name = "Log_output_%s.txt" % (model_dir)

    if TRAIN:
        print("Start Training..")
        for epoch_i in range(0, EpochNum+1):
            randidx_all = np.random.permutation(nrtrain)
            for batch_i in range(nrtrain // batch_size):
                randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

                batch_ys = clean_images[randidx, :]
                batch_xs = np.dot(batch_ys, Phi_input)

                feed_dict = {X_input: batch_xs, X_output: batch_ys, Y_label:y_train[randidx]}
                sess.run(optm_all, feed_dict=feed_dict)
                if (batch_i%log_every_k_batches) == 0:
                    randidx = randidx_all[batch_i*batch_size:(batch_i+10)*batch_size]
                    batch_ys = clean_images[randidx, :]
                    batch_xs = np.dot(batch_ys, Phi_input)
                    feed_dict = {X_input: batch_xs, X_output: batch_ys, Y_label:y_train[randidx]}
                    #print("Shape of batch ys ={}".format(batch_ys.shape))
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary,epoch_i*(nrtrain//batch_size)+batch_i)
            print("Epoch finished! Computing metrics")
            randidx = randidx_all[batch_i*batch_size:(batch_i+50)*batch_size]
            batch_ys = clean_images[randidx, :]
            batch_xs = np.dot(batch_ys, Phi_input)
            feed_dict = {X_input: batch_xs, X_output: batch_ys, Y_label:y_train[randidx]}
            c_n,c_s,c_l,c_a = sess.run([cost, cost_sym, classification_loss, class_accuracy], feed_dict=feed_dict)
            output_data = "[%02d/%02d] cost: %.4f, cost_sym: %.4f, class_loss: %.4f, class_accuracy: %.4f \n"% (epoch_i, EpochNum, c_n, c_s, c_l, c_a)
            print(output_data)
            #print("p = {}".format(p))
            #print("length of p = {}".format(len(p)))
            #print("Input = {}".format(batch_xs))
            #print("mean = {}".format(p.sum()))

            output_file = open(output_file_name, 'a')
            output_file.write(output_data)
            output_file.close()

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if epoch_i <= 30:
                saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
            else:
                if epoch_i % 20 == 0:
                    saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
        
        #tf.reset_default_graph()
        clear_session()
        print("Training Finished")

    else:
        saver.restore(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, EpochNum))
        print("Finished Restoring model")
    return
    #return sess, Prediction,Pre_symetric, classified, clean_images, Phi_input, X_input


def inference(sess, Prediction, classified, X_input, input_array):
    return sess.run([Prediction, classified], feed_dict = {X_input: input_array})
