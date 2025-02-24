import time

all_start_time = time.time()

import pynvml
import psutil
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
import matplotlib.pyplot as plt
import zipfile
import pandas as pd
import os
import time
import threading
import torch

# zip_path = "images.zip"
# with zipfile.ZipFile(zip_path, "r") as zip_ref:
#     zip_ref.extractall()

path = "images/images/"
file_names = []
labels = []

# Load dataset
print("Loading dataset...")
for file in sorted((Path(path).glob("*/*/*.*"))):
    label = str(file).split("/")[-2]
    labels.append(label)
    file_names.append(str(file))
print("Loaded dataset")

# Create DataFrame from image path and labels
df = pd.DataFrame.from_dict({"image": file_names, "label": labels})

# Shuffle images
stop_row = 10000
df = df.sample(frac=1).reset_index(drop=True).sample(n=stop_row).reset_index(drop=True)

# Load vision model
print("Loading model...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(
    "dima806/facial_emotions_image_detection"
)
model = AutoModelForImageClassification.from_pretrained(
    "dima806/facial_emotions_image_detection"
).to(device)
print("Loaded model")

# Used to track CPU/ram usage
cpu_usage = []
ram_vals = []
vram_vals = []
vram_usage = []
timestamps = []
start_time = time.time()
pid = os.getpid()
process = psutil.Process(pid)
finish = False

# Misc
checkpoint_row = max(int(len(df) / 10), 1)
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Store predictions
predictions = []


def monitor_usage():
    while not finish:
        cpu_percent = process.cpu_percent(interval=1)
        ram_val = process.memory_info().rss
        cpu_usage.append(cpu_percent)
        ram_vals.append(ram_val)
        timestamps.append(time.time() - start_time)

        vram_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        vram_vals.append(vram_used)


# Start the background thread
usage_thread = threading.Thread(target=monitor_usage, daemon=True)
usage_thread.start()

# Predict
print("Beginning predictions")
for idx, df_row in df.iterrows():
    image = cv2.imread(df_row.image)

    # Run image through model
    inputs = processor(image, return_tensors="pt").to(device)

    logits = model(**inputs).logits
    # Take the most confident emotion
    predicted_label = logits.argmax(-1).item()

    # Convert from id to emotion label
    pred = model.config.id2label[predicted_label]
    predictions.append(pred)
    actual = df_row.label

    # Print statements to see progress
    if idx % checkpoint_row == 0:
        print(f"Progress: {idx}/{len(df)} over {round((time.time() - start_time))}s")

print("Complete!")
finish = True
usage_thread.join()
pynvml.nvmlShutdown()


def format_bytes(size):
    units = ["bytes", "KB", "MB", "GB", "TB", "PB"]

    if size == 0:
        return "0 bytes"

    unit_index = 0
    while size >= 1000 and unit_index < len(units) - 1:
        size /= 1000.0
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


timestamps.pop()
cpu_usage.pop()

print(f"Max RAM usage: {format_bytes(max(ram_vals))}")
print(f"Average RAM usage: {format_bytes(round(sum(ram_vals)/len(ram_vals)))}")
print(f"Max VRAM usage: {format_bytes(max(vram_vals))}")
print(f"Average VRAM usage: {format_bytes(round(sum(vram_vals)/len(vram_vals)))}")
print(f"Average CPU usage: {round(sum(cpu_usage)/len(cpu_usage), ndigits=2)}%")


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(timestamps, cpu_usage, label="CPU Usage (%)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Usage (%)")
ax.set_ylim(bottom=0, top=max(cpu_usage) + 10)
ax.legend()
plt.savefig("usage_graph.png")

accuracy = 0

# Calculate accuracy
for row, pred in enumerate(predictions):
    cur_row = df.iloc[row]
    if pred == cur_row.label:
        accuracy += 1

print(f"Correctly classified {accuracy} out of {len(predictions)} images")
print(f"Accuracy: {round(100.0 * accuracy / len(predictions), ndigits=2)}%")
print(f"Total run time: {round((time.time() - all_start_time))}s")
