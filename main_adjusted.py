import numpy as np
from config import get_audio_data
import soundfile as sf
import random
import librosa

# while current_beat < num_of_beats:
#     choices = get_available_choices(
#         current_tempo, initial_tempo, allowed_tempo_deviation
#     )
#     current_choice = random.choice(choices)
#     current_beat_sample = current_beat * beat_length_in_samples
#     if current_choice == 2:  # Increase tempo for this beat
#         deviation_tempo = random.randint(1, allowed_tempo_deviation)
#         current_tempo = initial_tempo + deviation_tempo
#         new_sr = (beat_length_in_samples * current_tempo) / 60

#     elif current_choice == 3:  # Decrease tempo for this beat
#         deviation_tempo = random.randint(-allowed_tempo_deviation, -1)
#         current_tempo = initial_tempo + deviation_tempo
#         new_sr = (beat_length_in_samples * current_tempo) / 60

#     else:
#         current_tempo = initial_tempo

#     current_beat += 1


def get_probability_matrix(matrix, notes):
    return dict(zip(notes, matrix))


def get_random_proba_list(weights):
    output = []
    for weight in weights:
        choice = random.uniform(0, weight)
        output.append(choice)
    return output


def get_window_by_beat(expected_hit_timestamp, beat_len):
    half = int(0.1 * beat_len)
    start_of_window = max(0, expected_hit_timestamp - half)
    end_of_window = expected_hit_timestamp + half
    return (start_of_window, end_of_window)


def get_deviated_sample(
    start_of_window, end_of_window, expected_hit_timestamp, shift_proba
):
    if random.random() >= shift_proba:
        return expected_hit_timestamp
    return int(random.uniform(start_of_window, end_of_window))


# def crossfade_add(dst, seg, start, fade=256):
#     end = start + len(seg)
#     if end > len(dst):
#         pad = end - len(dst)
#         dst.resize(len(dst) + pad, refcheck=False)

#     if fade > 0:
#         f1 = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
#         f0 = np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
#         tail = dst[start : start + fade]
#         seg[:fade] = seg[:fade] * f1 + tail * f0


#     dst[start:end] += seg
#     return end


def skeleton_generator(bpm, skeleton, num_cycles, shift_proba, sr=48000):
    beat_length_in_samples = int((60 / bpm) * sr)
    skeleton_length = len(skeleton)
    num_of_beats_in_audio = num_cycles * sum(x[0] for x in skeleton)

    # [(1, D), (2.5, T), (2, S)]
    length_in_samples = int(
        sum([x[0] * beat_length_in_samples for x in skeleton]) * num_cycles
    )
    skeleton_hits_intervals = []
    y = np.zeros(length_in_samples + beat_length_in_samples)

    curr_beat = i = 0
    while curr_beat <= num_of_beats_in_audio:
        curr_beat += skeleton[i % skeleton_length][0]
        curr_hit = skeleton[i % skeleton_length][1]
        y_hit = get_audio_data(curr_hit, sr)
        expected_hit_timestamp = int(curr_beat * beat_length_in_samples)
        start_of_window, end_of_window = get_window_by_beat(
            expected_hit_timestamp, beat_length_in_samples
        )
        adjusted_hit_timestamp = get_deviated_sample(
            start_of_window, end_of_window, expected_hit_timestamp, shift_proba
        )
        end_of_hit_timestamp = adjusted_hit_timestamp + len(y_hit)

        if end_of_hit_timestamp <= len(y):
            y[adjusted_hit_timestamp:end_of_hit_timestamp] += y_hit
            skeleton_hits_intervals.append(
                (adjusted_hit_timestamp, end_of_hit_timestamp)
            )
        i += 1
    y_without_initial_silence = y[skeleton_hits_intervals[0][0] - 10 :]
    sf.write(
        "./skeleton.wav",
        data=y_without_initial_silence,
        samplerate=sr,
    )
    return (
        y_without_initial_silence,
        beat_length_in_samples,
        skeleton_hits_intervals,
    )


def subdivisions_generator(
    y,
    maxsubd,
    added_hits_intervals,
    beat_length_in_samples,
    hit_probabilities,
):
    added_hits_intervals = sorted(added_hits_intervals, key=lambda x: x[0])
    subdivisions_y = np.zeros(len(y))
    sample_of_curr_subd = 0
    maxsubd_length = int(beat_length_in_samples / maxsubd)
    hits = list(hit_probabilities.keys())
    weights = list(hit_probabilities.values())
    new_added_hits_intervals = []
    while sample_of_curr_subd < len(subdivisions_y):
        remaining = len(subdivisions_y) - sample_of_curr_subd
        random_proba_list = get_random_proba_list(weights)
        chosen_hit = random.choices(hits, weights=random_proba_list, k=1)[0]
        # print(f"choosen hit : {chosen_hit}")
        if chosen_hit == "S":
            sample_of_curr_subd += maxsubd_length
        else:
            hit_y = get_audio_data(chosen_hit)
            add_len = min(len(hit_y), remaining)

            for start, _ in added_hits_intervals:
                if start <= sample_of_curr_subd <= start + maxsubd_length:
                    sample_of_curr_subd += maxsubd_length
                    break
            else:
                subdivisions_y[sample_of_curr_subd : sample_of_curr_subd + add_len] += (
                    hit_y[:add_len]
                )
                new_added_hits_intervals.append(
                    (
                        sample_of_curr_subd,
                        sample_of_curr_subd + add_len,
                    )
                )
                sample_of_curr_subd += maxsubd_length
    y += subdivisions_y
    # sf.write(
    #     f"./generated/t_{maxsubd}.wav",
    #     y,
    #     samplerate=48000,
    # )
    new_added_hits_intervals.extend(added_hits_intervals)
    return y, new_added_hits_intervals


def build_processes(maxsubd, number_of_hits, hits_list, probabilities_matrix):
    processes = []
    for col_index in range(maxsubd):
        current_process = {}
        sum_of_probabilities = 0
        for j in range(number_of_hits):
            current_hit = hits_list[j]
            current_process[current_hit] = probabilities_matrix[current_hit][col_index]
            sum_of_probabilities += probabilities_matrix[current_hit][col_index]
        if sum_of_probabilities > 100:
            raise ValueError(
                f"Column {col_index} probabilities sum to {sum_of_probabilities} (>100). "
                "Reduce one or more values so that the sum ≤ 100."
            )

        current_process["S"] = 100 - sum_of_probabilities
        processes.append(current_process)
    if len(processes) != maxsubd:
        raise ValueError(
            f"Expected {maxsubd} probability columns, got {len(processes)}."
        )
    return processes


def subdivisions_generator_adjusted(
    maxsubd,
    probabilities_matrix,
    bpm,
    skeleton,
    num_cycles,
    sr=48000,
):
    num_of_beats = num_cycles * sum(x[0] for x in skeleton)
    hits_list = list(probabilities_matrix.keys())
    number_of_hits = len(hits_list)
    processes = build_processes(
        maxsubd=maxsubd,
        number_of_hits=number_of_hits,
        hits_list=hits_list,
        probabilities_matrix=probabilities_matrix,
    )
    y, beat_length_in_samples, added_hits_intervals = skeleton_generator(
        bpm=bpm, skeleton=skeleton, num_cycles=num_cycles, shift_proba=0.8, sr=sr
    )
    for subdiv in range(1, maxsubd, 1):
        # col_idx = maxsubd - subdiv
        y, added_hits_intervals = subdivisions_generator(
            y=y,
            maxsubd=subdiv,
            added_hits_intervals=added_hits_intervals,
            beat_length_in_samples=beat_length_in_samples,
            hit_probabilities=processes[subdiv],
        )
    # sf.write(
    #     "./final.wav",
    #     y,
    #     samplerate=48000,
    # )
    return y, num_of_beats, bpm


# def adjust_generated_tempo(
#     allowed_tempo_deviation,
#     y,
#     num_of_beats,
#     initial_tempo,
#     sr,
#     fade=256,
#     hold_prob=1 / 3,
#     down_prob=1 / 3,
# ):
#     beat_length_in_samples = int((60 / initial_tempo) * sr)
#     beat_length_in_samples = int((60.0 / initial_tempo) * sr)
#     out = np.zeros(int(len(y) * 1.5) + sr, dtype=y.dtype)
#     write_cursor = 0

#     for b in range(num_of_beats):
#         start = b * beat_length_in_samples
#         stop = start + beat_length_in_samples
#         if start >= len(y):
#             break
#         chunk = y[start : min(stop, len(y))].copy()
#         r = random.random()
#         if r < down_prob:
#             direction = -1
#         elif r < down_prob + hold_prob:
#             direction = 0
#         else:
#             direction = 1

#         if direction == 0 or allowed_tempo_deviation == 0:
#             target_bpm = initial_tempo
#         else:
#             dev = random.randint(1, allowed_tempo_deviation)
#             target_bpm = initial_tempo + direction * dev
#             target_bpm = max(1.0, target_bpm)

#         rate = float(target_bpm) / float(initial_tempo)

#         if len(chunk) < 2 * fade:
#             stretched = chunk
#         else:
#             stretched = librosa.effects.time_stretch(chunk, rate=rate)

#         write_cursor = crossfade_add(out, stretched, write_cursor, fade=fade)

#     out = out[:write_cursor]
#     sf.write(
#         "./final.wav",
#         out,
#         samplerate=sr,
#     )
#     return out


# def _xfade_append(accum, seg, fade=256):
#     """Append seg to accum with a short crossfade."""
#     if accum.size == 0:
#         return seg.copy()

#     if fade > 0:
#         f = min(fade, len(accum), len(seg))
#         if f > 0:
#             # linear ramps
#             w_out = np.linspace(1.0, 0.0, f, dtype=accum.dtype)
#             w_in = 1.0 - w_out
#             accum[-f:] = accum[-f:] * w_out + seg[:f] * w_in
#             return np.concatenate([accum, seg[f:]], axis=0)

#     # no fade or not enough samples for it
#     return np.concatenate([accum, seg], axis=0)


def get_available_choices(current_tempo, initial_tempo, allowed_tempo_deviation):
    lower = initial_tempo - allowed_tempo_deviation
    upper = initial_tempo + allowed_tempo_deviation
    choices = [1]  # keep
    if current_tempo <= lower:
        choices.append(2)  # increase
    elif current_tempo >= upper:
        choices.append(3)  # decrease
    else:
        choices.extend([2, 3])
    return choices


# def fit_to_length(x, target_len):
#     if len(x) == target_len:
#         return x
#     if len(x) > target_len:
#         return x[:target_len]
#     pad = target_len - len(x)
#     if len(x) == 0:
#         return np.zeros(target_len, dtype=np.float32)
#     return np.pad(x, (0, pad), mode="edge")


def adjust_generated_tempo(allowed_tempo_deviation, y, num_of_beats, initial_tempo, sr):
    beat_length_in_samples = int((60 / initial_tempo) * sr)
    current_beat = 1
    current_tempo = initial_tempo
    new_y=[]
    while current_beat < num_of_beats:
        start = current_beat * beat_length_in_samples
        end = start + beat_length_in_samples
        if start >= len(y):
            break
        choices = get_available_choices(
            current_tempo, initial_tempo, allowed_tempo_deviation
        )
        choice = random.choice(choices)

        if choice == 2:  # Increase
            deviation = random.randint(1, initial_tempo+allowed_tempo_deviation-current_tempo)
            current_tempo = current_tempo + deviation
        elif choice == 3:  # Decrease
            deviation = random.randint(1, initial_tempo+allowed_tempo_deviation-current_tempo)
            current_tempo = current_tempo - deviation
        else:  # Keep
            current_tempo = current_tempo
        new_sr = (beat_length_in_samples * current_tempo) / 60.0
        seg = y[start:end]
        seg_rs = librosa.resample(y=seg, orig_sr=sr, target_sr=new_sr)
        new_y=np.concatenate([new_y,seg_rs])
        current_beat += 1

    sf.write(
        "./final.wav",
        new_y,
        samplerate=sr,
    )

    return new_y


notes = ["D", "OTA", "OTI", "T1", "T2", "RA", "PA2"]
squleton = [(1, "D"), (1, "T1"), (1, "OTA"), (1, "RA")]
matrix = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875],
    [1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875],
    [0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
]


probabilities_matrix = get_probability_matrix(matrix=matrix, notes=notes)
print(probabilities_matrix)
y_generated, num_of_beats, initial_tempo = subdivisions_generator_adjusted(
    maxsubd=8,
    bpm=120,
    probabilities_matrix=probabilities_matrix,
    skeleton=squleton,
    num_cycles=10,
)
adjust_generated_tempo(
    y=y_generated,
    allowed_tempo_deviation=5,
    num_of_beats=num_of_beats,
    initial_tempo=initial_tempo,
    sr=48000,
)
