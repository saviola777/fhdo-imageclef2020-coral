import subprocess
import sys
import os
import glob

MODEL_DIR = "/path/to/your/models"
SCRIPT_DIR = "/path/to/this/repository/train_evaluate/"
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_cross_eval.py")
PREDICTIONS_SCRIPT = os.path.join(SCRIPT_DIR, "create_predictions_evaluations.py")

model_subdir = os.path.join(MODEL_DIR, sys.argv[1])

subprocess.call(["mkdir", "-p", model_subdir])
subprocess.call(["cp", TRAIN_SCRIPT, os.path.join(model_subdir, "train.py")])
subprocess.call(["cp", PREDICTIONS_SCRIPT, os.path.join(model_subdir, "create_predictions_evaluations.py")])


print(f"Prepared training for '{model_subdir}'")
