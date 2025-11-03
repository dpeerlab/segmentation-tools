import os
import json
from collections import defaultdict
from datetime import datetime
from time import time
from importlib import reload
import skimage
import mirage
import tensorflow as tf
from tensorflow.python.framework import errors

# HELPER FUNCTIONS
def get_max_date(date_strings):
    """Return file with highest data. Used for getting most recent logfile"""
    dates = [datetime.strptime(date, "%Y_%m_%d_%H_%M_%S") for date in date_strings]
    max_date = max(dates)
    max_date_str = max_date.strftime("%Y_%m_%d_%H_%M_%S")
    return max_date_str

def format_bytes(size, unit="MB"):
    """Human readable filesize"""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    assert unit in units, f"Unit {unit} not supported. Choose from {', '.join(units)}"
    
    for u in units:
        if u != unit:
            size /= 1024
        else:
            break
            
    return f"{round(size, 2)} {unit}"

def get_latest_logs(path="logdir_path", filepattern="memory_profile.json"):
    """Get latest json log file, unzip it and load it."""

    # get logfiles (get highest date)
    logdir = os.path.join(path, "plugins/profile")
    files = os.listdir(logdir)
    max_date = get_max_date(files)
    logdir_max = os.path.join(logdir, max_date)

    # get filename
    files = os.listdir(logdir_max)
    try:
        filename = [f for f in files if filepattern in f][0]
    except:
        raise Exception(f"No files found. Filepattern: {filepattern} | Path: {logdir_max} | Files: {', '.join(files)}")
    filepath_gz = os.path.join(logdir_max, filename)
    assert ".json" in filepath_gz, f"File must be a (gzipped) json file. Got {filepath_gz}."

    # unzip memory file
    if filepath_gz.endswith(".gz"):
        os.system(f"gunzip {filepath_gz}")
    filepath = filepath_gz.replace(".gz", "")

    # open most recent
    with open(filepath, "r") as f:
        memory = json.load(f)
    return memory


# TEST FUNCTIONS
def run_experiment(
    img,
    ref,
    width=100,
    pad=48,
    neurons=196,
    layers=3,
    batch_size=64,
    num_steps=12
):
    """
    Run experiment to test impact of parameters on memory and runtime
    
    * Get subset of image
    * Init profiler
    * Build model
    * Train model
    * Evaluate result
    * Return summary
    """
    # meassure time
    begin = time()

    # sample image
    rows = img.shape[0]
    cols = img.shape[1]
    mesh = mirage.tl.Mesh(x=int(rows / 2), y=int(cols / 2), pad=int(width / 2))

    img_mesh = mesh.get_mesh(img)
    ref_mesh = mesh.get_mesh(ref)
    
    # star profiling
    try:
        tf.profiler.experimental.stop()
        print("Restarting profiler")
    except errors.UnavailableError:
        print("Starting profiler")
    tf.profiler.experimental.start('logdir_path')

    try:
        # build model
        mirage_model = mirage.MIRAGE(
            images=ref_mesh,
            references=img_mesh,
            pad=pad,
            offset=pad,
            num_neurons=neurons,  # more for larger images
            num_layers=layers,  # more for larger images
            pool=1, 
            loss="SSIM"
        )

        # train model
        mirage_model.train(batch_size=batch_size, num_steps=num_steps, lr__sched=True, LR=0.005)
    except Exception as e:
        tf.profiler.experimental.stop()
        raise e
    
    # end tracking
    tf.profiler.experimental.stop()
    dur = time() - begin

    # evaluate model
    mirage_model.compute_transform(num_cut=1000)
    ref_mesh_aligned = mirage_model.apply_transform(ref_mesh)
    ssim, ssim_image = skimage.metrics.structural_similarity(
        img_mesh, ref_mesh_aligned, data_range=1, full=True
    )

    # get summary
    summary = {
        "dur": dur,
        "ssim": ssim,
        "ssim_image": ssim_image
    }

    return mirage_model, summary


def get_memory_summary(top_k=5):
    """
    Extract information from memory logs
    
    * Get memory file
    * Summarise results
    * Return summary
    """
    
    # read and asser memory logs
    memory = get_latest_logs()
    if int(memory["numHosts"]) > 1:
        raise Exception("More than one host not yet supported")
    if len(memory["memoryIds"]) > 2:
        raise Exception("More than one GPU not yet supported")
    
    # init memory values
    mems = memory["memoryProfilePerAllocator"]["GPU_0_bfc"]
    peaks = defaultdict(int)

    # extract data
    for m in mems["memoryProfileSnapshots"]:
        op_name = m["activityMetadata"]["tfOpName"]
        peak = m["activityMetadata"]["allocationBytes"]
        peaks[op_name] = max(peaks[op_name], int(peak))

    # get peaks
    peak_ids = dict(sorted(peaks.items(), key=lambda x: x[1], reverse=True)[:top_k])
    peak_ids = {k: format_bytes(v) for k, v in peak_ids.items()}
    
    peak_bytes_use = format_bytes(int(mems["profileSummary"]["peakStats"]["peakBytesInUse"]))
    peak_free = format_bytes(int(mems["profileSummary"]["peakStats"]["freeMemoryBytes"]))
    
    peaks = {
        "ids": peak_ids,
        "max_use": peak_bytes_use,
        "free": peak_free,
    }
    return peaks

