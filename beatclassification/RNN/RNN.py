import tensorflow as tf
import numpy as np

class RNN:
    def predict(self, X_train, X_test, Y_train, Y_test, labels_counter):
        element_size = 2;time_steps = 340;num_classes = 4
        batch_size = 128;hidden_layer_size = 128

        _inputs = tf.placeholder(tf.float32,shape=[None, time_steps, element_size],
            name='inputs')
        y = tf.placeholder(tf.float32, shape=[None, num_classes],name='inputs')

        # TensorFlow built-in functions
        rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
        outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=.01))
        bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))


        def get_linear_layer(vector):
            return tf.matmul(vector, Wl) + bl

        last_rnn_output = outputs[:, -1, :]
        final_output = get_linear_layer(last_rnn_output)

        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,
                                                          labels=y)
        cross_entropy = tf.reduce_mean(softmax)
        train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        i = 0
        while i< X_train.shape[0] - batch_size:
            iter_number = int(i/ batch_size)
            total_number = int(X_train.shape[0] / batch_size)
            print(str(iter_number)+" of "+ str(total_number))
            batch_x= X_train[i:i+batch_size]
            batch_y= Y_train[i:i+batch_size]
            batch_x = batch_x.reshape((batch_size, time_steps, element_size))
            sess.run(train_step, feed_dict={_inputs: batch_x,
                                            y: batch_y})
            if i % 100 == 0:
                acc = sess.run(accuracy, feed_dict={_inputs: batch_x,
                                                    y: batch_y})
                loss = sess.run(cross_entropy, feed_dict={_inputs: batch_x,
                                                      y: batch_y})
                print ("Iter " + str(iter_number) + ", Minibatch Loss=  {:.6f}".format(loss) + ", Training Accuracy= " 
                   "{:.5f}".format(acc))
            i += batch_size
        print("Test step:")
        i = 0
        predictions = []
        while i < X_test.shape[0] - batch_size:
            iter_number = int(i / batch_size)
            total_number = int(X_test.shape[0] / batch_size)
            print(str(iter_number) + " of " + str(total_number))
            batch_x = X_test[i:i + batch_size]
            batch_y = Y_test[i:i + batch_size]
            batch_x = batch_x.reshape((batch_size, time_steps, element_size))
            acc = sess.run(accuracy, feed_dict={_inputs: batch_x,
                                            y: batch_y})
            loss, pred = sess.run([cross_entropy,final_output], feed_dict={_inputs: batch_x,
                                                      y: batch_y})
            print("Iter " + str(i) + ", Minibatch Loss=  {:.6f}".format(loss) + ", Training Accuracy= "
                                                                              "{:.5f}".format(acc))
            predictions.append(pred)
            i += batch_size
        return predictions
        #print ("Testing Accuracy:" + str(np.mean(accuracies)))