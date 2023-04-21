import os
from functools import partial
import argparse

from lib import settings, train, model, utils
from tensorflow.python.eager import profiler
import tensorflow as tf


def main(**kwargs):
  """ Main function for training ESRGAN model
  """

  for physical_device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(physical_device, True)

  sett = settings.Settings(kwargs["config"])
  Stats = settings.Stats(os.path.join(sett.path, "stats.yaml"))
  profiler.start_profiler_server(6009)
  generator = model.RRDBNet(out_channel=3)
  discriminator = model.VGG28()

  training = train.Trainer(
      settings=sett,
      data_dir=kwargs["data_dir"],
      manual=kwargs["manual"])
  phases = list(map(lambda x: x.strip(),
                    kwargs["phases"].lower().split("_")))

  if not Stats["train_step_1"] and "phase1" in phases:
    training.warmup_generator(generator)
    Stats["train_step_1"] = True
  if not Stats["train_step_2"] and "phase2" in phases:
    training.train_gan(generator, discriminator)
    Stats["train_step_2"] = True

  if Stats["train_step_1"] and Stats["train_step_2"]:
  
    interpolated_generator = utils.interpolate_generator(
        partial(model.RRDBNet, out_channel=3),
        discriminator,
        sett["interpolation_parameter"],
        sett["dataset"]["hr_dimension"])
    tf.saved_model.save(interpolated_generator, kwargs["model_dir"])


if __name__ == '_main_':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--config",
      default="config/config.yaml",
      help="path to configuration file. (default: %(default)s)")
  parser.add_argument(
      "--data_dir",
      default=None,
      help="directory to put the data. (default: %(default)s)")
  parser.add_argument(
      "--manual",
      default=False,
      help="specify if data_dir is a manual directory. (default: %(default)s)",
      action="store_true")
  parser.add_argument(
      "--model_dir",
      default=None,
      help="directory to put the model in.")
  parser.add_argument(
      "--phases",
      default="phase1_phase2",
      help="phases to train for seperated by '_'")
  FLAGS, unparsed = parser.parse_known_args()
  main(**vars(FLAGS))