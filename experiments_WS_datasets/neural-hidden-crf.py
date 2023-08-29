import sys
sys.path.append('..')
import datetime
import numpy as np
import os
import logging
import torch
import optparse
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.model_neural_hidden_crf import NeuralHiddenCrf
from wrench.seq_labelmodel import SeqLabelModelWrapper
from wrench.labelmodel import MajorityVoting
from wrench.evaluation import SEQ_METRIC


optparser = optparse.OptionParser()
optparser.add_option("--dataset", default="conll", type="str")
optparser.add_option("--patience", default="100", type="int")
optparser.add_option("--result_path", default="../results/", type="str")

optparser.add_option("--batch_size_mv", default="16", type="int")
optparser.add_option("--lr_mv", default="5e-5", type="float")
optparser.add_option("--lr_crf_mv", default="0.005", type="float")
optparser.add_option("--batch_size", default="16", type="int")
optparser.add_option("--lr", default="5e-5", type="float")
optparser.add_option("--lr_crf", default="0.005", type="float")

optparser.add_option("--lr_worker", default="0.001", type="float")
optparser.add_option("--scaling", default="4.0", type="float")

optparser.add_option("--run_on_testset", default=None, type="str")
optparser.add_option("--run_times", default=None, type="str")
opts = optparser.parse_args()[0]

mv_paramters = {"conll_None": {"batch_size_mv": 8, "lr_mv": 3e-5, "lr_crf_mv": 5e-3},
                "conll_True": {"batch_size_mv": 8, "lr_mv": 2e-5, "lr_crf_mv": 0.005},
                "wikiglod_None": {"batch_size_mv": 8, "lr_mv": 2e-5, "lr_crf_mv": 1e-2},
                "wikigold_True": {"batch_size_mv": 8, "lr_mv": 5e-5, "lr_crf_mv": 0.01},
                "mit-restaurants_None": {"batch_size_mv": 32, "lr_mv": 5e-5, "lr_crf_mv": 0.01},
                "mit-restaurants_True": {"batch_size_mv": opts.batch_size, "lr_mv": opts.lr, "lr_crf_mv": opts.lr_crf},
                }
dataset_run_on_testset = opts.dataset + "_" + str(opts.run_on_testset)
opts.batch_size_mv = mv_paramters[dataset_run_on_testset]["batch_size_mv"]
opts.lr_mv = mv_paramters[dataset_run_on_testset]["lr_mv"]
opts.lr_crf_mv = mv_paramters[dataset_run_on_testset]["lr_crf_mv"]

opts.result_path = "./results/" + str(opts.dataset) + "/" + str(opts.batch_size) + \
               str(opts.lr) + str(opts.lr_crf) + \
               str(opts.lr_worker) + str(opts.scaling) + str(opts.run_on_testset) + "_" + opts.run_times
print("opts:", opts)
print('\n')



if not os.path.exists(opts.result_path):
    os.makedirs(opts.result_path)
this_file = os.path.join(opts.result_path, "running_time.npy")
if os.path.exists(this_file):
    print('\n')
    print('HAVE COMPUTED!')
    print('\n')
    exit()
if os.path.exists(os.path.join(opts.result_path, 'begin_compute.npy')):
    print('\n')
    print('This experiment is running elsewhere!')
    print('\n')
    exit()
begin_compute = []
np.save(os.path.join(opts.result_path, 'begin_compute.npy'), begin_compute)
start = datetime.datetime.now()
print("Start running time: %s" % start)


### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
device = torch.device('cuda')



### Load dataset
dataset_path = './datasets/'
data = opts.dataset
train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=False)
if opts.run_on_testset == None:
    crowd_labels = train_data.weak_labels
    train_data_correct_labels_id = train_data.labels
else:
    crowd_labels = test_data.weak_labels
    train_data_correct_labels_id = test_data.labels
workers_number = len(crowd_labels[0][0])
train_data_number = len(crowd_labels)
label2id = train_data.label2id
print(label2id)
O_id = label2id['O']
labels_number = len(label2id)



### Truth inference with Majority Voting
label_model = SeqLabelModelWrapper(
    label_model_class=MajorityVoting,
    # lr=0.01,
    # l2=0.0,
    n_epochs=10
)
metric_fn = SEQ_METRIC['f1_seq']
if opts.run_on_testset == None:
    label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
    mv_labels = label_model.test_return_infered_result(train_data, 'f1_seq')
else:
    label_model.fit(dataset_train=test_data, dataset_valid=valid_data)
    mv_labels = label_model.test_return_infered_result(test_data, 'f1_seq')



### Parameter initialization for weak source transition matrices
pi = np.zeros((workers_number, labels_number, labels_number))
for r in range(workers_number):
    normalizer = np.zeros(labels_number)
    for i in range(len(train_data_correct_labels_id)):
        for j in range(len(train_data_correct_labels_id[i])):
            ground_truth_est = np.zeros(labels_number)
            ground_truth_est[mv_labels[i][j]] = 1
            normalizer += ground_truth_est
            pi[r, :, crowd_labels[i][j][r]] += ground_truth_est
    normalizer[normalizer == 0] = 0.0000001
    pi[r] = pi[r] / normalizer.reshape(labels_number, 1)



### Learning
model = NeuralHiddenCrf(
    batch_size_mv=opts.batch_size_mv,
    batch_size=opts.batch_size,
    test_batch_size=512,
    n_steps=10000,
    lr_mv=opts.lr_mv,
    lr=opts.lr,
    lr_crf_mv=opts.lr_crf_mv,
    lr_crf=opts.lr_crf,
    use_crf=True,
    lr_worker=opts.lr_worker,
    scaling=opts.scaling,
)
if opts.run_on_testset == None:
    _, y_train_crowd_return = model.fit(
        dataset_train=train_data,
        test_data=test_data,
        pi=pi,
        y_train_crowd=crowd_labels,
        y_train_mv=mv_labels,
        dataset_valid=valid_data,
        evaluation_step=10,
        metric='f1_seq',
        patience=opts.patience,
        device=device,
        result_path=opts.result_path
    )
else:
    _, y_train_crowd_return = model.fit(
        dataset_train=test_data,
        test_data=test_data,
        pi=pi,
        y_train_crowd=crowd_labels,
        y_train_mv=mv_labels,
        dataset_valid=valid_data,
        evaluation_step=10,
        metric='f1_seq',
        patience=opts.patience,
        device=device,
        result_path=opts.result_path
    )



### Evaluation/Test
test_result_early_stop, train_result_early_stop = {}, {}
test_result_early_stop_wrench_viterbi = {}

f1 = model.test(test_data, 'f1_seq')
precision = model.test(test_data, 'precision_seq')
recall = model.test(test_data, 'recall_seq')

test_result_early_stop['f1'] = f1
test_result_early_stop['precision'] = precision
test_result_early_stop['recall'] = recall
np.save(os.path.join(opts.result_path, 'test_result_early_stop.npy'), test_result_early_stop)
logger.info(f'end model test f1: {f1}')


if opts.run_on_testset == None:
    f1 = model.test(train_data, 'f1_seq')
    precision = model.test(train_data, 'precision_seq')
    recall = model.test(train_data, 'recall_seq')

    train_result_early_stop['f1'] = f1
    train_result_early_stop['precision'] = precision
    train_result_early_stop['recall'] = recall
    np.save(os.path.join(opts.result_path, 'train_result_early_stop.npy'), train_result_early_stop)
    logger.info(f'end model train f1: {f1}')

    end = datetime.datetime.now()
    print("End running time: %s" % end)
    print('Running time: %s Seconds' % (end - start))


end = datetime.datetime.now()
print("End running time: %s" % end)
print('Running time: %s Seconds' % (end - start))
os.remove(os.path.join(opts.result_path, 'begin_compute.npy'))







