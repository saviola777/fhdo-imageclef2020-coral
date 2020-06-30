import subprocess
import sys
import os
import glob

MODEL_DIR = "/path/to/your/models"
SCRIPT_DIR = "/path/to/this/repository/train_evaluate/"
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_cross_eval.py")
PREDICTIONS_SCRIPT = os.path.join(SCRIPT_DIR, "create_predictions_evaluations.py")

for run_name in sys.argv[1:]:

    model_subdir = os.path.join(MODEL_DIR, run_name)
    subprocess.call(["mkdir", "-p", model_subdir])

    try:
        with open(os.path.join(model_subdir, "train.txt"), "w+") as output:
            subprocess.call(
                ["python3", "-u", os.path.join(model_subdir, "train.py"), run_name],
                stdout=output)
            subprocess.call(
                ["python3", os.path.join(SCRIPT_DIR, "delete_models.py"), run_name],
                stdout=output)
            subprocess.call(
                ["python3", os.path.join(model_subdir, "create_predictions_evaluations.py"),
                 run_name],
                stdout=output)
    except KeyboardInterrupt:
        subprocess.call(["pkill", "python3"])

    print(f"Training for '{model_subdir}' finished")
