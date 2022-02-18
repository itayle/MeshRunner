import sys, copy
from easydict import EasyDict
import json
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse
import rnn_model
import utils
import dataset
import train_val
import pydevd_pycharm


def run():
  parser = argparse.ArgumentParser(allow_abbrev=True)
  parser.add_argument("--gpu", type=str)
  parser.add_argument("--dir", type=str)
  # parser.add_argument("--debug", action="store_true")

  args = parser.parse_args()
  # if args.debug:
  #   pydevd_pycharm.settrace('10.12.9.2', port=35687, stdoutToServer=True, stderrToServer=True)

  np.random.seed(0)
  tf.random.set_seed(0)
  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  utils.config_gpu()
  job = "shrec11"
  job_part = "10-10_A"
  # those params are task wide. more specific params are later overwriten by calc_accuracy_test
  params = train_val.get_params(job, job_part)
  seq_len_list = [25, 50, 100]
  n_walks_list = [1, 4, 16, 32]
  for seq_len in seq_len_list:
    print("---------seq_len:", seq_len, "---------")
    for n_walks in n_walks_list:
      acc, _ = train_val.calc_accuracy_test(logdir=dir, **params.full_accuracy_test, n_walks_per_model=n_walks,
                                            iter2use='00010008.keras', seq_len=seq_len)
      print("n_walks:", n_walks, "acc", acc[0])


if __name__ == '__main__':
  run()
