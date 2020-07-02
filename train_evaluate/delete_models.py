import os
import glob
import sys
import csv
import subprocess

# Deletes all but the best 5 models plus every 10th model

RUN_NAME = sys.argv[1]
ROOT_DIR = os.path.abspath("/path/to/models")
MODEL_DIR = os.path.join(ROOT_DIR, RUN_NAME)

if not os.path.exists(MODEL_DIR):
    raise ValueError("Invalid model dir")

model_losses = [[], [], [], [], []]

with open(os.path.join(MODEL_DIR, "losses.csv"), 'r') as input_csv:
    reader = csv.reader(input_csv, delimiter=',')
    next(reader)

    for row in reader:
        run_name, split, epoch, val_loss, train_loss = row[:5]

        glob_search = os.path.join(MODEL_DIR,
            "coral" + split) + '/*/mask_rcnn_coral_0' + str(int(epoch)+1).zfill(3) + '.h5'
        current_loss = float(train_loss) * .368 + float(val_loss) * .632
        try:
            model_losses[int(split)].append((current_loss, glob.glob(glob_search)[0]))
        except IndexError:
            print(f"Warning: Didn't find model for glob {glob_search}")
            pass

# change the 5 here if you want to keep more / less
models_to_be_deleted = list(sorted(x, key=lambda k: k[0])[5:] for x in model_losses)
#models_to_be_kept = list(sorted(x, key=lambda k: k[0])[:5] for x in model_losses)

for i in range(5):
    for loss, model_path in models_to_be_deleted[i]:
        if int(model_path.split("/")[-1].split(".")[0].split("_")[-1]) % 10 != 0:
            print(f'Deleting {model_path} ({loss})')
            subprocess.call(["rm", model_path])

    #for loss, model_path in models_to_be_kept[i]:
        #print(f'Would keep {model_path} ({loss})')
