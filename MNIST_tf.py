import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn as sl
#network model
def net(data):
    #1
    x = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    x = tf.nn.relu(x + b1)
    #2
    x = tf.nn.conv2d(x, w2, [1, 1, 1, 1], padding='SAME')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    x = tf.nn.relu(x + b2)

    x = tf.reshape(x, (-1, 7 * 7 * 64))
    x = tf.nn.relu(tf.matmul(x, w3) + b3)
    return tf.matmul(x, w4) + b4

#data preprocessing
traindata = pd.read_csv('train.csv') 
trainlabel = np.array(traindata.pop('label')) 
trainlabel = sl.preprocessing.LabelEncoder().fit_transform(trainlabel)[:, None]
trainlabel = sl.preprocessing.OneHotEncoder().fit_transform(trainlabel).todense()
traindata = sl.preprocessing.StandardScaler().fit_transform(np.float32(traindata.values)) 
traindata = traindata.reshape(-1, 28, 28, 1) 
train_data, valid_data = traindata[:-3000], traindata[-3000:]
train_label, valid_label = trainlabel[:-3000], trainlabel[-3000:]
#set model
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))
#First Convolutional Layer
w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b1 = tf.Variable(tf.zeros([32]))
#Second Convolutional Layer
w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b2 = tf.Variable(tf.constant(1.0, shape=[64]))
#Densely Connected Layer
w3 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b3 = tf.Variable(tf.constant(1.0, shape=[1024]))
#Readout Layer
w4 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b4 = tf.Variable(tf.constant(1.0, shape=[10]))

pred = tf.nn.softmax(net(x))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net(x), y))
acc = 100*tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))))
tf_step = tf.train.AdamOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

batch = sl.model_selection.ShuffleSplit(n_splits=10000, train_size=100) #batch = 100
batch.get_n_splits(train_data, train_label)
for step, (idx, _) in enumerate(batch.split(train_data,train_label), start=1):
    fd = {x:train_data[idx], y:train_label[idx]}
    session.run(tf_step, feed_dict=fd)
    if step%1000 == 0:
        fd = {x:valid_data, y:valid_label}
        valid_loss, valid_accuracy = session.run([loss, acc], feed_dict=fd)
        print('Step %i \t Acc. = %f'%(step, valid_accuracy))

testdata = pd.read_csv('test.csv') 
test_image = sl.preprocessing.StandardScaler().fit_transform(np.float32(testdata.values)).reshape(-1, 28, 28, 1)
#prediction and output
test_pred = session.run(pred, feed_dict={x:test_image})
test_labels = np.argmax(test_pred, axis=1)

submission = pd.DataFrame(data={'imageid':(np.arange(test_labels.shape[0])+1), 'label':test_labels})
submission.to_csv('submission.csv', index=False).tail()