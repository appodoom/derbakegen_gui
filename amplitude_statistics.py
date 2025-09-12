from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def load_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload["all_peaks"] if isinstance(payload, dict) else payload
    return data


def save_json_file(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {"all_peaks": arr.tolist()}, f, ensure_ascii=False, separators=(",", ":")
        )
    return str(path)


def get_amps_peaks(folder_name, output_file_name):
    all_peaks = []
    filepath = "./final_4.wav"
    y, sr = librosa.load(filepath, sr=None)
    y_abs = np.abs(y)

    peaks = librosa.util.peak_pick(
        y_abs,
        pre_max=1000,
        post_max=1000,
        pre_avg=1000,
        post_avg=1000,
        delta=0.02,
        wait=1000,
    )

    file_peaks = y_abs[peaks]
    all_peaks.extend(file_peaks)
    # for file in os.listdir(folder_name):
    #     if file.endswith(".wav"):
    #         filepath = os.path.join(folder_name, file)
    #         y, sr = librosa.load(filepath, sr=None)
    #         y_abs = np.abs(y)

    #         peaks = librosa.util.peak_pick(
    #             y_abs,
    #             pre_max=1000,
    #             post_max=1000,
    #             pre_avg=1000,
    #             post_avg=1000,
    #             delta=0.02,
    #             wait=1000,
    #         )

    #         file_peaks = y_abs[peaks]
    #         all_peaks.extend(file_peaks)
    print(np.percentile(all_peaks, 50))
    save_json_file(all_peaks, output_file_name)
    return all_peaks


def region_stats(values, bounds):
    values = np.asarray(values)
    reps = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i < len(bounds) - 2:
            mask = (values >= lo) & (values < hi)
        else:
            mask = (values >= lo) & (values <= hi)
        x = values[mask]
        if x.size:
            centroid = float(x.mean())
            median = float(np.median(x))
            rms = float(np.sqrt(np.mean(x**2)))
            count = int(x.size)
        else:
            centroid = median = rms = float((lo + hi) / 2)  # mid_point if bin is empty
            count = 0
        reps.append(
            {
                "range": [lo, hi],
                "centroid_mean": centroid,
                "median": median,
                "rms": rms,
                "count": count,
            }
        )
    return reps


def plot_amps_graph(all_peaks_json_file):
    all_peaks = load_json_file(all_peaks_json_file)
    plt.figure(figsize=(8, 5))
    plt.hist(all_peaks, bins=50, color="skyblue", edgecolor="black")
    plt.title("Amplitude Distribution of Hits (All WAV Files)")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# folder_name = "data/first_data"
# output_file = "all_peaks_old.json"
# bounds = np.array([0.0, 0.0742, 0.4125, 0.5987, 1.0])
# if not Path.exists(Path(output_file)):
#     get_amps_peaks(folder_name=folder_name, output_file_name=output_file)
# all_peaks = load_json_file(json_path=output_file)
# stats = region_stats(values=all_peaks, bounds=bounds)
# print(stats)
# range1: 0 -> 0.0742
# range2: 0.0742 ->0.4125
# range3: 0.4125 -> 0.5987
# range4: 0.5987 -> 1
