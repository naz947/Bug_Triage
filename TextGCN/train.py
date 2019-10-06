from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.sparse as sp
import os
import pandas as pd
from utils import *
from models import GCN, MLP, GCN_APPRO

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
dataset = sys.argv[1]#+'/clean'
name=dataset[-3:]
# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', dataset, 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_appr', 'Model string.')  # 'gcn', 'gcn_appr'
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
# Load data


def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]

def main(rank1, rank0):
    #adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
    adj, features, y_train, y_val, y_test, y_vocab, train_mask, val_mask, test_mask, vocab_mask, _, _ = load_corpus(FLAGS.dataset)
    train_index = np.where(train_mask)[0]
    vocab_index = np.where(vocab_mask)[0]
    tmp_index = list(train_index) + list(vocab_index)
    adj_train = adj[train_index, :][:, tmp_index]
    adj_train_vocab = adj[tmp_index, :][:, tmp_index]
    print(len(train_mask))
    train_mask = train_mask[train_index]
    y_train = y_train[train_index]
    val_index = np.where(val_mask)[0]
    # adj_val = adj[val_index, :][:, val_index]
    val_mask = val_mask[val_index]
    y_val = y_val[val_index]
    test_index = np.where(test_mask)[0]
    # adj_test = adj[test_index, :][:, test_index]
    test_mask = test_mask[test_index]
    y_test = y_test[test_index]


    numNode_train_1 = adj_train.shape[1]
    numNode_train_0 = adj_train.shape[0]
    # print("numNode", numNode)

    # Some preprocessing
    features = nontuple_preprocess_features(features).todense()
    train_features = features[tmp_index]

    if FLAGS.model == 'gcn_appr':
        normADJ_train = nontuple_preprocess_adj(adj_train)
        normADJ_train_vocab = nontuple_preprocess_adj(adj_train_vocab)
        #print(normADJ_train)
        normADJ = nontuple_preprocess_adj(adj)
        # normADJ_val = nontuple_preprocess_adj(adj_val)
        # normADJ_test = nontuple_preprocess_adj(adj_test)

        num_supports = 2
        model_func = GCN_APPRO
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features.shape[-1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy,model.outputs], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    cost_train=[]
    acc_val=[]
    acc_train=[]
    time_total=[]
    p0 = column_prop(normADJ_train)

    # testSupport = [sparse_to_tuple(normADJ), sparse_to_tuple(normADJ)]
    valSupport = [sparse_to_tuple(normADJ), sparse_to_tuple(normADJ[val_index, :])]
    testSupport = [sparse_to_tuple(normADJ), sparse_to_tuple(normADJ[test_index, :])]


    t = time.time()
    # Train model
    for epoch in range(FLAGS.epochs):
        t1 = time.time()

        n = 0
        for batch in iterate_minibatches_listinputs([normADJ_train, y_train, train_mask], batchsize=256, shuffle=True):
            [normADJ_batch, y_train_batch, train_mask_batch] = batch
            if sum(train_mask_batch) < 1:
                continue
            #print(normADJ_batch)
            p1 = column_prop(normADJ_batch)
            #print(p1.shape)
            q1 = np.random.choice(np.arange(numNode_train_1), rank1, p=p1)  # top layer
            # q0 = np.random.choice(np.arange(numNode_train), rank0, p=p0)  # bottom layer
            support1 = sparse_to_tuple(normADJ_batch[:, q1].dot(sp.diags(1.0 / (p1[q1] * rank1))))
            #print(q1)
            p2 = column_prop(normADJ_train_vocab[q1, :])
            #print(p2.shape)
            q0 = np.random.choice(np.arange(numNode_train_1), rank0, p=p2)
            support0 = sparse_to_tuple(normADJ_train_vocab[q1, :][:, q0])
            #print(y_train_batch, train_mask_batch, len(train_mask))
            features_inputs = sp.diags(1.0 / (p2[q0] * rank0)).dot(train_features[q0, :])  # selected nodes for approximation


            # Construct feed dictionary
            feed_dict = construct_feed_dict(features_inputs, [support0, support1], y_train_batch, train_mask_batch,
                                            placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        cost_train.append(outs[1])
        acc_train.append(outs[2])
        # Validation
        cost, acc,out_val, duration = evaluate(features, valSupport, y_val, val_mask, placeholders)
        cost_val.append(cost)
        acc_val.append(acc)
        # # Print results
        time_total.append(time.time()-t1)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t1))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            # print("Early stopping...")
            break

    train_duration = time.time() - t
    #time_total.append(train_duration)

    # Testing
    test_cost, test_acc,out_put, test_duration = evaluate(features, testSupport, y_test, test_mask,
                                                  placeholders)
    print("rank1 = {}".format(rank1), "rank0 = {}".format(rank0), "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "training time per epoch=", "{:.5f}".format(train_duration/epoch))
    print(type(y_test))
    print(len(y_test))
    print(y_test.shape)
    accuracy = []
    sortedIndices = []
    pred_classes = []
    correc_pred=[]
    for ll in out_put:
    	sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    for k in range(1, 10+1):
    	id = 0
    	trueNum = 0
    	for sortedInd in sortedIndices:
            if k==10:
                correc_pred.append(sortedInd[:k])
            pred_classes.append(sortedInd[:k])
            val=(np.where(y_test[id]==1))
            if val[0] in sortedInd[:k]:
                trueNum += 1
            id += 1
    	accuracy.append((float(trueNum) / len(out_put)) * 100)
    print('Test accuracy:', accuracy)

    pd.DataFrame(correc_pred).to_csv('fast/'+name+'pred.csv')

    pd.DataFrame(y_test).to_csv('fast/'+name+'_correct.csv')

'''
    new_file=open('fast/'+name+'_time','a+')
    ac_str=dataset+'\t'+str(time_total)+'\n'
    new_file.write(ac_str)
    new_file.close()

    new_file=open('fast/'+name+'_Accuracy_Normal.txt','a+')
    ac_str=dataset+'\t'+str(accuracy)+'\n'
    new_file.write(ac_str)
    new_file.close()

    new_file=open('fast/'+name+'_cost_train.txt','a+')
    str1=dataset+'\t'+str(cost_train)+'\n'
    new_file.write(str1)
    new_file.close()

    new_file=open('fast/'+name+'_cost_val.txt','a+')
    str1=dataset+'\t'+str(cost_val)+'\n'
    new_file.write(str1)
    new_file.close()

    new_file=open('fast/'+name+'_acc_train.txt','a+')
    str1=dataset+'\t'+str(acc_train)+'\n'
    new_file.write(str1)
    new_file.close()

    new_file=open('fast/'+name+'_acc_val.txt','a+')
    str1=dataset+'\t'+str(acc_val)+'\n'
    new_file.write(str1)
    new_file.close()
'''

if __name__=="__main__":
    print("DATASET:", FLAGS.dataset)
    for k in [600]:
        main(k, k)

    # main(50,50)
    # for k in [50, 100, 200, 400]:
    #     main(k, k)
