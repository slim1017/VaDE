# -*- coding: utf-8 -*-
"""
VaDE (Variational Deep Embedding:A Generative Approach to Clustering)
Reuters(685K) clustering accuracy: 79.38%
@code author: Zhuxi Jiang
"""
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
import scipy.io as scio
import theano
import theano.tensor as T
import math
from sklearn import preprocessing
from keras.layers.core import Dropout,Activation
import copy


import warnings
warnings.filterwarnings("ignore")

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
#=====================================
def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size
             
#==================================================
def load_data():
  print ('loading data....')
  from sklearn.feature_extraction.text import CountVectorizer
  did_to_cat = {}
  cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
  with open('dataset/reuters/rcv1-v2.topics.qrels') as fin:
    for line in fin.readlines():
      line = line.strip().split(' ')
      cat = line[0]
      did = int(line[1])
      if cat in cat_list:
        did_to_cat[did] = did_to_cat.get(did, []) + [cat]
    copy_dc=copy.copy(did_to_cat)
    for did in copy_dc.keys():
      if len(did_to_cat[did]) > 1:
        del did_to_cat[did]
  
  dat_list = ['lyrl2004_tokens_test_pt0.dat', 
              'lyrl2004_tokens_test_pt1.dat',
              'lyrl2004_tokens_test_pt2.dat',
              'lyrl2004_tokens_test_pt3.dat',
              'lyrl2004_tokens_train.dat']
  data = []
  target = []
  cat_to_cid = {'CCAT':0, 'GCAT':1, 'MCAT':2, 'ECAT':3}
  del did
  for dat in dat_list:
    print (dat+'....')
    with open('dataset/reuters/'+dat) as fin:
      for line in fin.readlines():
        if line.startswith('.I'):
          if 'did' in locals():
            assert doc != ''
            if did in did_to_cat.keys():
              data.append(doc)
              target.append(cat_to_cid[did_to_cat[did][0]])
          did = int(line.strip().split(' ')[1])
          doc = ''
        elif line.startswith('.W'):
          assert doc == ''
        else:
          doc += line

  assert len(data) == len(did_to_cat)

  X = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
  Y = np.asarray(target)

  from sklearn.feature_extraction.text import TfidfTransformer
  X = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(X)
  X = np.asarray(X.todense())*np.sqrt(X.shape[1])
  X = preprocessing.normalize(X, norm='l2')*200
  X = X.astype('float32')
  X = X[:685000]  # for 100 minibatch training
  Y = Y[:685000]
  return X,Y

        
def gmm_para_init():
    
    gmm_weights=scio.loadmat('trained_model_weights/reuters_all_weights_gmm.mat')
    u_init=gmm_weights['u']
    lambda_init=gmm_weights['lambda']
    theta_init=np.squeeze(gmm_weights['theta'])
    
    theta_p=theano.shared(np.asarray(theta_init,dtype=theano.config.floatX),name="pi")
    u_p=theano.shared(np.asarray(u_init,dtype=theano.config.floatX),name="u")
    lambda_p=theano.shared(np.asarray(lambda_init,dtype=theano.config.floatX),name="lambda")
    return theta_p,u_p,lambda_p

#================================
def get_gamma(tempz):
    temp_Z=T.transpose(K.repeat(tempz,n_centroid),[0,2,1])
    temp_u_tensor3=T.repeat(u_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_lambda_tensor3=T.repeat(lambda_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_theta_tensor3=theta_p.dimshuffle('x','x',0)*T.ones((batch_size,latent_dim,n_centroid))
    
    temp_p_c_z=K.exp(K.sum((K.log(temp_theta_tensor3)-0.5*K.log(2*math.pi*temp_lambda_tensor3)-\
                       K.square(temp_Z-temp_u_tensor3)/(2*temp_lambda_tensor3)),axis=1))
    return temp_p_c_z/K.sum(temp_p_c_z,axis=-1,keepdims=True)
#=====================================================

ispretrain = True
batch_size = 100
latent_dim = 10
intermediate_dim = [500,500,2000]
theano.config.floatX='float32'
X,Y = load_data()
original_dim = 2000
n_centroid = 4 
theta_p, u_p, lambda_p = gmm_para_init()
#===================

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim[0])(x)
h=Activation('relu')(h)
h=Dropout(0.2)(h)
h = Dense(intermediate_dim[1])(h)
h=Activation('relu')(h)
h=Dropout(0.2)(h)
h = Dense(intermediate_dim[2])(h)
h=Activation('relu')(h)
h=Dropout(0.2)(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
h_decoded = Dense(intermediate_dim[-1])(z)
h_decoded = Dense(intermediate_dim[-2])(h_decoded)
h_decoded = Dense(intermediate_dim[-3])(h_decoded)
x_decoded_mean = Dense(original_dim)(h_decoded)

#========================
p_c_z = Lambda(get_gamma, output_shape=(n_centroid,))(z_mean)
sample_output = Model(x, z_mean)
p_c_z_output = Model(x, p_c_z)
#===========================================      
vade = Model(x, x_decoded_mean)
vade.load_weights('trained_model_weights/reuters_all_weights_nn.h5')

accuracy = cluster_acc(np.argmax(p_c_z_output.predict(X,batch_size=batch_size),axis=1),Y)
print ('Reuters_all dataset VaDE - clustering accuracy: %.2f%%'%(accuracy*100))