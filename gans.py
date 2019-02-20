import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

# Create Object class to contain both hour image and hour label
class fig_with_label:
    def __init__(self,image,label):
        self.image = image
        self.label = label

# Loading the dataset for further use
def create_dataset():
    r = 150.0 / 432
    dim = (150, int(288 * r))
    dataset = []
    dir = './Picture/'
    i = 1
    for folders in os.listdir(dir):
        for file in os.listdir(dir+folders):
            img_raw = cv2.imread(dir+folders+'/'+file)
            resized = cv2.resize(img_raw, dim, interpolation = cv2.INTER_AREA)
            img = resized[:,:,0]
            dataset.append(fig_with_label(img,folders))
            i = i+1
    return dataset

# Select particular hour image
def select_label(label):
    l_dataset = []
    for each in dataset:
        if each.label == label:
            l_dataset.append(each.image)
    return l_dataset

# Format dataset
class dataset_format:
    def __init__(self,dataset):
        images = []
        labels = []
        for each in dataset:
            images.append(each.image.reshape(15000))
            labels.append(each.label)
        self.images = np.array(images)
        self.labels = np.array(labels)

#Grab Next batch of dataset
r = 0
def next_batch(data,size):
    global r
    if r*size*size > len(data):
        r = 0
    x_train_batch = data[size*r:r*size*size,:]
    r = r+1
    return x_train_batch


# Initial Weight
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.1))

# Initial bias
def init_bias(shape):
    return tf.Variable(tf.constant(0.2,shape=shape))

# Define Generator
class Generator:
    def __init__(self):
        with tf.variable_scope('gen'):
            self.gW1 = init_weight([1000,500])
            self.gb1 = init_bias([500])
            self.gW2 = init_weight([500,15000])
            self.gb2 = init_bias([15000])

    def forward(self,z,training = True):
        fc1 = tf.matmul(z,self.gW1) + self.gb1
        fc1 = tf.layers.batch_normalization(fc1,training=training)
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.nn.sigmoid(tf.matmul(fc1,self.gW2)+self.gb2)
        return fc2

class Discriminator:
    def __init__(self):
        with tf.variable_scope('dis'):
            self.dW1 = init_weight([5,5,1,8])
            self.db1 = init_bias([8])
            self.dW2 = init_weight([5,5,8,16])
            self.db2 = init_bias([16])

            self.W3 = init_weight([100*150*16,240])
            self.b3 = init_bias([240])
            self.W4 = init_weight([240,1])
            self.b4 = init_bias([1])

    def forward(self,X):
        self.X = tf.reshape(X,shape=[-1,100,150,1])
        conv1 = tf.nn.relu(tf.nn.conv2d(self.X,self.dW1,strides=[1,1,1,1],padding='SAME')+self.db1)
        conv1 = tf.layers.batch_normalization(conv1,True)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1,self.dW2,strides=[1,1,1,1],padding='SAME')+self.db2)
        conv2 = tf.layers.batch_normalization(conv2,True)
        conv2 = tf.reshape(conv2,shape=[-1,100*150*16])

        fc1 = tf.nn.relu(tf.matmul(conv2,self.W3)+self.b3)
        logits = tf.matmul(fc1,self.W4) + self.b4
        fc2 = tf.nn.sigmoid(logits)

        return fc2,logits

def cost(logits,labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))

# Loading dataset
dataset= create_dataset()
dataset_format = dataset_format(dataset)
hour_images = dataset_format.images
hour_labels = dataset_format.labels.reshape(len(dataset_format.labels))
print('Finish loading dataset!')

d = Discriminator()
g = Generator()

phX = tf.placeholder(tf.float32,[None,15000])
phZ = tf.placeholder(tf.float32,[None,1000])

G_out = g.forward(phZ)
G_out_sample = g.forward(phZ,False)

D_out_real, D_logits_real = d.forward(phX)
D_fake_loss, D_logits_fake = d.forward(G_out)

D_real_loss = cost(D_logits_real,tf.ones_like(D_logits_real))
D_fake_loss = cost(D_logits_fake,tf.zeros_like(D_logits_fake))

D_loss = D_real_loss + D_fake_loss
G_loss = cost(D_logits_fake,tf.ones_like(D_logits_fake))

learning_rate = 0.001

epochs = 7000

pretrain_epochs = 1000
batch_size = 10

train_vars = tf.trainable_variables()

dvars = [var for var in train_vars if 'dis' in var.name]
gvars = [var for var in train_vars if 'gen' in var.name]

D_train = tf.train.AdamOptimizer(learning_rate).minimize(D_loss,var_list=dvars)
G_train = tf.train.AdamOptimizer(learning_rate).minimize(G_loss,var_list=gvars)

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    for i in range(24):

        sess.run(init)
        print('Start training!')

        k = 0
        l = 10
        data = hour_images[hour_labels==str(i)]

        print("Starting training for label {}".format(i))
        g_cost = []
        d_cost = []

        for j in range(epochs):

            print('{}'.format(j))

            batch_X = next_batch(data,batch_size)

            batch_z = np.random.randn(batch_size,1000)

            #Training Discriminator
            _,d_loss = sess.run([D_train,D_loss],feed_dict={phX:batch_X,phZ:batch_z})
            #Training Generator
            _, g_loss = sess.run([G_train,G_loss],feed_dict={phZ:batch_z})

            #Append loss for later plotting
            d_cost.append(d_loss)
            g_cost.append(g_loss)

            #Images generation countdown
            if j % pretrain_epochs//10 == 0 and j < pretrain_epochs:
                print('Pretraining. Generating images for label {} in {}'.format(i,l))
                l = l-1

            #Generating Images
            if j % 10 ==0 and j>= pretrain_epochs:
                print("Generate Picture")

                sample_z = np.random.randn(1,1000)

                gen_sample = sess.run(G_out_sample,feed_dict={phZ:sample_z})

                # Print iteration and d_cost
                print('Iteration {}. G_loss {}. D_loss {}'.format(j,G_loss,D_loss))

                image = plt.imshow(gen_sample.reshape(100,150))

                plt.savefig('./{}.{}'.format(i,j))
