import numpy as np
import utils
import train_val
import os
import argparse
import pydevd_pycharm
import tensorflow as tf

def run():
  random_num = np.random.randint(9999)
  parser = argparse.ArgumentParser(allow_abbrev=True)
  parser.add_argument("--gpu", type=str)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--att", default=False, action="store_true")
  parser.add_argument("--slurm_job", type=str)
  parser.add_argument("--seed", default=random_num, type=int)

  args = parser.parse_args()
  if args.debug:
    num_of_epochs_for_validtion = 2
    pydevd_pycharm.settrace('10.12.9.2', port=35687, stdoutToServer=True, stderrToServer=True)
  else:
    num_of_epochs_for_validtion = 50

  np.random.seed(args.seed)
  tf.random.set_seed(args.seed)
  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  utils.config_gpu()
  job = "shrec11"
  job_part = "10-10_A"
  if args.slurm_job:
    run_name = args.slurm_job
  else:
    run_name = str(np.random.randint(99999))

  run_name += "seed_" + str(args.seed)
  run_name += "att_" + str(args.att)
  # wandb.init(project="mesh", entity="itaylevy")
  train_val.run_one_job(job=job,
                        job_part=job_part,
                        optim="adam",
                        lr=1e-4,
                        num_of_epochs_for_validtion=num_of_epochs_for_validtion,
                        attention=args.att,
                        custom_name=run_name)


if __name__ == "__main__":
  run()
