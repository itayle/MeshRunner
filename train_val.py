import os
import time
import sys
from easydict import EasyDict
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import dataset
import params_setting
import walks

import sys, copy
from easydict import EasyDict
import json

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import rnn_model
import utils


def calc_accuracy_test(dataset_expansion=False, logdir=None, labels=None, iter2use='last', classes_indices_to_use=None,
                       dnn_model=None, params=None, min_max_faces2use=[0, 4000], model_fn=None, n_walks_per_model=16, data_augmentation={},
                       attention=False, seq_len=None):
  # Prepare parameters for the evaluation
  if params is None:
    with open(logdir + '/params.txt') as fp:
      params = EasyDict(json.load(fp))
    if model_fn is not None:
      pass
    elif iter2use != 'last':
      model_fn = logdir + '/learned_model2keep__' + iter2use
      model_fn = model_fn.replace('//', '/')
    else:
      model_fn = tf.train.latest_checkpoint(logdir)
  else:
    params = copy.deepcopy(params)
  if logdir is not None:
    params.logdir = logdir
  params.n_walks_per_model = n_walks_per_model
  params.batch_size = 1
  if seq_len:
    params.seq_len = seq_len
  params.classes_indices_to_use = None
  params.classes_indices_to_use = classes_indices_to_use

  # Prepare the dataset
  test_dataset, n_models_to_test = dataset.tf_mesh_dataset(params, dataset_expansion, mode=params.network_task,
                                                           shuffle_size=0, permute_file_names=True, min_max_faces2use=min_max_faces2use,
                                                           must_run_on_all=True, data_augmentation=data_augmentation, use_saliency=params.saliency)


  # If dnn_model is not provided, load it
  if dnn_model is None:
    dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim, model_fn,
                                     model_must_be_load=True, dump_model_visualization=False)

  n_pos_all = 0
  n_classes = 40
  all_confusion = np.zeros((n_classes, n_classes), dtype=np.int)
  pred_per_model_name = {}
  for i, data in tqdm(enumerate(test_dataset), total=n_models_to_test):
    name, ftrs, gt = data
    model_fn = name.numpy()[0].decode()
    model_name, n_faces = utils.get_model_name_from_npz_fn(model_fn)
    assert ftrs.shape[0] == 1, 'Must have one model per batch for test'
    modelss_per_batch = ftrs.shape[0]
    ftrs = tf.reshape(ftrs, ftrs.shape[1:])
    gt = gt.numpy()[0]
    ftr2use = ftrs.numpy()
    predictions = dnn_model(ftr2use, classify=True, training=False).numpy()

    mean_pred = np.mean(predictions, axis=0)
    max_hit = np.argmax(mean_pred)

    if model_name not in pred_per_model_name.keys():
      pred_per_model_name[model_name] = [gt, np.zeros_like(mean_pred)]
    pred_per_model_name[model_name][1] += mean_pred

    all_confusion[int(gt), max_hit] += 1
    n_pos_all += (max_hit == gt)

  n_models = 0
  n_sucesses = 0
  all_confusion_all_faces = np.zeros((n_classes, n_classes), dtype=np.int)
  # counts per mesh, if the averge of all walks is accurate
  for k, v in pred_per_model_name.items():
    gt = v[0]
    pred = v[1]
    max_hit = np.argmax(pred)
    all_confusion_all_faces[gt, max_hit] += 1
    n_models += 1
    n_sucesses += max_hit == gt
  mean_accuracy_all_faces = n_sucesses / n_models

  # Print list of accuracy per model
  for confusion in [all_confusion, all_confusion_all_faces]:
    acc_per_class = []
    for i, name in enumerate(labels):
      this_type = confusion[i]
      n_this_type = this_type.sum()
      accuracy_this_type = this_type[i] / n_this_type
      if n_this_type:
        acc_per_class.append(accuracy_this_type)
      this_type_ = this_type.copy()
      this_type_[i] = -1
  mean_acc_per_class = np.mean(acc_per_class)

  return [mean_accuracy_all_faces, mean_acc_per_class], dnn_model
def get_valid(logdir):
  from train_val import get_params
  utils.config_gpu(True)
  np.random.seed(0)
  tf.random.set_seed(0)
  job = "shrec11"
  job_part = "10-10_A"
  params = get_params(job, job_part)
  accs1, _ = calc_accuracy_test(logdir=logdir, **params.full_accuracy_test, n_walks_per_model=1)
  accs4, _ = calc_accuracy_test(logdir=logdir, **params.full_accuracy_test, n_walks_per_model=4)
  accs16, _ = calc_accuracy_test(logdir=logdir, **params.full_accuracy_test, n_walks_per_model=16)
  # accs32, _ = calc_accuracy_test(logdir=logdir, **params.full_accuracy_test, n_walks_per_model=32)
  return (accs1[0], accs4[0], accs16[0])



def train_val(params,num_of_epochs_for_validtion):
  utils.next_iter_to_keep = 10000
  print(utils.color.BOLD + utils.color.RED + 'params.logdir :::: ', params.logdir, utils.color.END)
  print(utils.color.BOLD + utils.color.RED, os.getpid(), utils.color.END)
  utils.backup_python_files_and_params(params)
  print('1')
  # Set up datasets for training and for test
  # -----------------------------------------
  train_datasets = []
  train_ds_iters = []
  max_train_size = 0
  print('2')
  print(len(params.datasets2use['train']))
  print("----")
  print(params.train_data_augmentation)
  print(4)
  for i in range(len(params.datasets2use['train'])):
    print("*******")
    this_train_dataset, n_trn_items = dataset.tf_mesh_dataset(params, params.datasets2use['train'][i],
                                                              mode=params.network_tasks[i], size_limit=params.train_dataset_size_limit,
                                                              shuffle_size=100, min_max_faces2use=params.train_min_max_faces2use,
                                                              min_dataset_size=128,
                                                              data_augmentation=params.train_data_augmentation)
    print("*#########*")
    print('Train Dataset size:', n_trn_items)
    train_ds_iters.append(iter(this_train_dataset.repeat()))
    train_datasets.append(this_train_dataset)
    max_train_size = max(max_train_size, n_trn_items)
  train_epoch_size = max(8, int(max_train_size / params.n_walks_per_model / params.batch_size))
  print('train_epoch_size:', train_epoch_size)
  if params.datasets2use['test'] is None:
    test_dataset = None
    n_tst_items = 0
  else:
    test_dataset, n_tst_items = dataset.tf_mesh_dataset(params, params.datasets2use['test'][0],
                                                        mode=params.network_tasks[0], size_limit=params.test_dataset_size_limit,
                                                        shuffle_size=100, min_max_faces2use=params.test_min_max_faces2use)
    # -------------------------
    for i, data in tqdm(enumerate(test_dataset), total=n_tst_items):
      name, ftrs, gt = data
      break
    # -----------------------------
  print(' Test Dataset size:', n_tst_items)

  # Set up RNN model and optimizer
  # ------------------------------
  if params.net_start_from_prev_net is not None:
    init_net_using = params.net_start_from_prev_net
  else:
    init_net_using = None

  if params.optimizer_type == 'adam':
    optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate[0], clipnorm=params.gradient_clip_th)
  elif params.optimizer_type == 'cycle':
    @tf.function
    def _scale_fn(x):
      x_th = 500e3 / params.cycle_opt_prms.step_size
      if x < x_th:
        return 1.0
      else:
        return 0.5
    lr_schedule = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params.cycle_opt_prms.initial_learning_rate,
                                                      maximal_learning_rate=params.cycle_opt_prms.maximal_learning_rate,
                                                      step_size=params.cycle_opt_prms.step_size,
                                                      scale_fn=_scale_fn, scale_mode="cycle", name="MyCyclicScheduler")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=params.gradient_clip_th)
  elif params.optimizer_type == 'sgd':
    optimizer = tf.keras.optimizers.SGD(lr=params.learning_rate[0], decay=0, momentum=0.9, nesterov=True, clipnorm=params.gradient_clip_th)
  else:
    raise Exception('optimizer_type not supported: ' + params.optimizer_type)

  if params.net == 'RnnWalkNet':
    dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim, init_net_using, optimizer=optimizer)

  # Other initializations
  # ---------------------
  time_msrs = {}
  time_msrs_names = ['train_step', 'get_train_data', 'test']
  for name in time_msrs_names:
    time_msrs[name] = 0
  seg_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='seg_train_accuracy')

  train_log_names = ['seg_loss']
  train_logs = {name: tf.keras.metrics.Mean(name=name) for name in train_log_names}
  train_logs['seg_train_accuracy'] = seg_train_accuracy

  # Train / test functions
  # ----------------------
  if params.last_layer_actication is None:
    seg_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  else:
    seg_loss = tf.keras.losses.SparseCategoricalCrossentropy()

  @tf.function
  def train_step(model_ftrs_, labels_, one_label_per_model, warmup_stage=False):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    with tf.GradientTape() as tape:
      if one_label_per_model:
        labels = tf.reshape(tf.transpose(tf.stack((labels_,)*params.n_walks_per_model)),(-1,))
        predictions = dnn_model(model_ftrs, warmup_stage=warmup_stage)
      else:
        labels = tf.reshape(labels_, (-1, sp[-2]))
        skip = params.min_seq_len
        predictions = dnn_model(model_ftrs)[:, skip:]
        labels = labels[:, skip + 1:]
      seg_train_accuracy(labels, predictions)
      loss = seg_loss(labels, predictions)
      loss += tf.reduce_sum(dnn_model.losses)

    gradients = tape.gradient(loss, dnn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dnn_model.trainable_variables))

    train_logs['seg_loss'](loss)

    return loss

  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  @tf.function
  def test_step(model_ftrs_, labels_, one_label_per_model):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    if one_label_per_model:
      labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
      predictions = dnn_model(model_ftrs, training=False)
    else:
      labels = tf.reshape(labels_, (-1, sp[-2]))
      skip = params.min_seq_len
      predictions = dnn_model(model_ftrs, training=False)[:, skip:]
      labels = labels[:, skip + 1:]
    best_pred = tf.math.argmax(predictions, axis=-1)
    test_accuracy(labels, predictions)
    confusion = tf.math.confusion_matrix(labels=tf.reshape(labels, (-1,)), predictions=tf.reshape(best_pred, (-1,)),
                                         num_classes=params.n_classes)
    return confusion
  # -------------------------------------

  # Loop over training EPOCHs
  # -------------------------
  one_label_per_model = params.network_task == 'classification'
  next_iter_to_log = 0
  e_time = 0
  accrcy_smoothed = tb_epoch = last_loss = None
  all_confusion = {}
  with tf.summary.create_file_writer(params.logdir).as_default():
    epoch = 0
    while optimizer.iterations.numpy() < params.iters_to_train + train_epoch_size * 2:
      # wandb.tensorflow.log(tf.summary.merge_all())
      epoch += 1
      str_to_print = str(os.getpid()) + ') Epoch' + str(epoch) + ', iter ' + str(optimizer.iterations.numpy())

      # Save some logs & infos
      utils.save_model_if_needed(optimizer.iterations, dnn_model, params)
      if tb_epoch is not None:
        e_time = time.time() - tb_epoch
        tf.summary.scalar('time/one_epoch', e_time, step=optimizer.iterations)
        tf.summary.scalar('time/av_one_trn_itr', e_time / n_iters, step=optimizer.iterations)
        for name in time_msrs_names:
          if time_msrs[name]:  # if there is something to save
            tf.summary.scalar('time/' + name, time_msrs[name], step=optimizer.iterations)
            time_msrs[name] = 0
      tb_epoch = time.time()
      n_iters = 0
      tf.summary.scalar(name="train/learning_rate", data=optimizer._decayed_lr(tf.float32), step=optimizer.iterations)
      tf.summary.scalar(name="mem/free", data=utils.check_mem_and_exit_if_full(), step=optimizer.iterations)
      gpu_tmpr = utils.get_gpu_temprature()
      if gpu_tmpr > 95:
        print('GPU temprature is too high!!!!!')
        exit(0)
      tf.summary.scalar(name="mem/gpu_tmpr", data=gpu_tmpr, step=optimizer.iterations)

      # Train one EPOC
      train_logs['seg_loss'].reset_states()
      tb = time.time()
      
      for iter_db in range(train_epoch_size):

        for dataset_id in range(len(train_datasets)):
          name, model_ftrs, labels = train_ds_iters[dataset_id].next()
          dataset_type = utils.get_dataset_type_from_name(name)
          time_msrs['get_train_data'] += time.time() - tb
          n_iters += 1
          tb = time.time()
          if params.train_loss[dataset_id] == 'cros_entr':
            # %reload_ext tensorboard
            # #%load_ext tensorboard
            warmup_stage = epoch <= 400
            train_step(model_ftrs, labels, one_label_per_model=one_label_per_model, warmup_stage=False)
            loss2show = 'seg_loss'
          else:
            raise Exception('Unsupported loss_type: ' + params.train_loss[dataset_id])
          time_msrs['train_step'] += time.time() - tb
          tb = time.time()
        if iter_db == train_epoch_size - 1:
          str_to_print += ', TrnLoss: ' + str(round(train_logs[loss2show].result().numpy(), 2))

      # Dump training info to tensorboard
      if optimizer.iterations >= next_iter_to_log:
        for k, v in train_logs.items():
          if v.count.numpy() > 0:
            tf.summary.scalar('train/' + k, v.result(), step=optimizer.iterations)
            v.reset_states()
        next_iter_to_log += params.log_freq

      # Run test on part of the test set
      if test_dataset is not None:
        n_test_iters = 0
        tb = time.time()
        for name, model_ftrs, labels in test_dataset:
          n_test_iters += model_ftrs.shape[0]
          if n_test_iters > params.n_models_per_test_epoch:
            break
          confusion = test_step(model_ftrs, labels, one_label_per_model=one_label_per_model)
          dataset_type = utils.get_dataset_type_from_name(name)
          if dataset_type in all_confusion.keys():
            all_confusion[dataset_type] += confusion
          else:
            all_confusion[dataset_type] = confusion
        # Dump test info to tensorboard
        if accrcy_smoothed is None:
          accrcy_smoothed = test_accuracy.result()
        accrcy_smoothed = accrcy_smoothed * .9 + test_accuracy.result() * 0.1
        tf.summary.scalar('test/accuracy_' + dataset_type, test_accuracy.result(), step=optimizer.iterations)
        str_to_print += ', test/accuracy_' + dataset_type + ': ' + str(round(test_accuracy.result().numpy(), 2))
        test_accuracy.reset_states()
        time_msrs['test'] += time.time() - tb

      str_to_print += ', time: ' + str(round(e_time, 1))
      print("original loss:",str_to_print)
      if epoch % num_of_epochs_for_validtion ==0:
        acc = get_valid(params.logdir)
        print("real_loss",acc)
        with open(params.logdir + "/eval_log.txt", 'at') as f:
          f.write("epoch: " + str(epoch) + " acc: " + str(acc) + " loss: " + str(round(train_logs[loss2show].result().numpy(), 2)) )
          f.write("\n")
  return last_loss


def get_params(job, job_part, attention=False, saliency=False, custom_name=None):
  # Classifications
  job = job.lower()

  if job == 'modelnet40' or job == 'modelnet':
    params = params_setting.modelnet_params()

  if job == 'shrec11':
    params = params_setting.shrec11_params(job_part, attention, saliency, custom_name)

  if job == 'cubes':
    params = params_setting.cubes_params()

  # Semantic Segmentations
  if job == 'human_seg':
    params = params_setting.human_seg_params()

  if job == 'coseg':
    params = params_setting.coseg_params(job_part)   #  job_part can be : 'aliens' or 'chairs' or 'vases'
  return params


# optim in ['sgd', 'adam','cycle']
def run_one_job(job, job_part, optim,lr,num_of_epochs_for_validtion, attention=False, saliency=False, custom_name=None):
  print("-"*50)
  print("-"*50)
  print("-"*50)
  print('-'*9, "Itamar, Amit and Itay 3D Project" ,'-'*9)
  print("-"*50)
  print("-"*50)
  print("-"*50)
  params = get_params(job, job_part, attention, saliency, custom_name)
  params.learning_rate = [lr,lr]
  params.optimizer_type = optim
  train_val(params,num_of_epochs_for_validtion)


def get_all_jobs():
  jobs = [
    'shrec11', 'shrec11', 'shrec11',
    'shrec11', 'shrec11', 'shrec11',
    'coseg', 'coseg', 'coseg',
    'human_seg',
    'cubes',
    'modelnet40',
  ][6:]
  job_parts = [
    '10-10_A', '10-10_B', '10-10_C',
    '16-04_A', '16-04_B', '16-04_C',
    'aliens', 'vases', 'chairs',
    None,
    None,
    None,
  ][6:]

  return jobs, job_parts

