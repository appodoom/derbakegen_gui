import numpy as np
from config import get_audio_data
import soundfile as sf
import random
# from pathlib import Path
# from amplitude_statistics import plot_amps_graph, get_amps_peaks, load_json_file


def get_probability_matrix(matrix, notes):
    return dict(zip(notes, matrix))


def get_random_proba_list(weights):
    output = []
    for weight in weights:
        choice = random.uniform(0, weight)
        output.append(choice)
    return output


def get_window_by_beat(expected_hit_timestamp, beat_len):
    half = int(0.05 * beat_len)
    start_of_window = max(0, expected_hit_timestamp - half)
    end_of_window = expected_hit_timestamp + half
    return (start_of_window, end_of_window)


def get_deviated_sample(
    start_of_window, end_of_window, expected_hit_timestamp, shift_proba
):
    if random.random() >= shift_proba:
        return expected_hit_timestamp
    return int(random.uniform(start_of_window, end_of_window))


def skeleton_generator(amplitude, skeleton, num_cycles, tempos, shift_proba, sr=48000):
    beat_length_in_samples = int((60 / tempos[0]) * sr)
    skeleton_length = len(skeleton)
    num_of_beats_in_audio = num_cycles * sum(x[0] for x in skeleton)

    # [(1, D), (2.5, T), (2, S)]
    skeleton_hits_intervals = []
    y = np.zeros(0, dtype=np.float32)
    expected_hit_timestamp = 0
    curr_beat = i = 0
    while curr_beat <= num_of_beats_in_audio:
        beat = skeleton[i % skeleton_length][0]
        curr_beat += beat
        if curr_beat % 1 == 0:
            beat_length_in_samples = int((60 / tempos[int(curr_beat - 1)]) * sr)
        curr_hit = skeleton[i % skeleton_length][1]
        y_hit = np.asarray(get_audio_data(curr_hit, sr), dtype=np.float32)
        expected_hit_timestamp += int(beat * beat_length_in_samples)
        if i == 0:
            y = np.zeros(expected_hit_timestamp, dtype=np.float32)
        start_of_window, end_of_window = get_window_by_beat(
            expected_hit_timestamp, beat_length_in_samples
        )
        adjusted_hit_timestamp = get_deviated_sample(
            start_of_window, end_of_window, expected_hit_timestamp, shift_proba
        )
        end_of_hit_timestamp = adjusted_hit_timestamp + len(y_hit)
        if end_of_hit_timestamp > y.size:
            pad_len = end_of_hit_timestamp - y.size
            y = np.pad(y, (0, pad_len), mode="constant")
        print(adjusted_hit_timestamp, end_of_hit_timestamp)
        np.concatenate((y, np.zeros(end_of_hit_timestamp - adjusted_hit_timestamp + 1)))
        y[adjusted_hit_timestamp:end_of_hit_timestamp] += y_hit
        skeleton_hits_intervals.append((adjusted_hit_timestamp, end_of_hit_timestamp))
        i += 1
    y_without_initial_silence = amplitude * y[skeleton_hits_intervals[0][0] - 10 :]
    # sf.write(
    #     "./generated/skeleton1.wav",
    #     data=y_without_initial_silence,
    #     samplerate=sr,
    # )
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
    subdiv_proba,
    amplitudes,
    amplitudes_proba_list,
    tempos,
    sr=48000,
):
    subdiv_array = []
    for i in range(len(subdiv_proba)):
        subdiv_array.append(i)
    maxsubdi = random.choices(population=subdiv_array, weights=subdiv_proba, k=1)[0]
    added_hits_intervals = sorted(added_hits_intervals, key=lambda x: x[0])
    subdivisions_y = np.zeros(len(y))
    sample_of_curr_subd = 0
    beat_length_in_samples = int(60 * sr / tempos[0])
    maxsubd_length = int(beat_length_in_samples / (maxsubd - maxsubdi))
    hits = list(hit_probabilities[maxsubdi].keys())
    weights = list(hit_probabilities[maxsubdi].values())
    new_added_hits_intervals = []
    j = 0
    while sample_of_curr_subd < len(subdivisions_y):
        if sample_of_curr_subd % beat_length_in_samples == 0:
            maxsubdi = random.choices(
                population=subdiv_array, weights=subdiv_proba, k=1
            )[0]
            maxsubd_length = int(beat_length_in_samples / (maxsubd - maxsubdi))
            hits = list(hit_probabilities[maxsubdi].keys())
            weights = list(hit_probabilities[maxsubdi].values())
            j += 1
            beat_length_in_samples = int(60 * sr / tempos[j])
        remaining = len(subdivisions_y) - sample_of_curr_subd
        random_proba_list = get_random_proba_list(weights)
        chosen_hit = random.choices(hits, weights=random_proba_list, k=1)[0]
        choosen_amplitude = random.choices(
            population=amplitudes, weights=amplitudes_proba_list, k=1
        )[0]
        # print(f"choosen hit : {chosen_hit}")
        if chosen_hit == "S":
            sample_of_curr_subd += maxsubd_length
        else:
            hit_y = np.asarray(get_audio_data(chosen_hit, sr), dtype=np.float32)
            add_len = min(len(hit_y), remaining)

            for start, _ in added_hits_intervals:
                if start <= sample_of_curr_subd <= start + maxsubd_length:
                    sample_of_curr_subd += maxsubd_length
                    break
            else:
                subdivisions_y[sample_of_curr_subd : sample_of_curr_subd + add_len] += (
                    choosen_amplitude * (hit_y[:add_len])
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
    subdiv_proba,
    amplitudes,
    amplitudes_proba_list,
    cycle_length,
    shift_proba,
    sr=48000,
):
    num_of_beats = num_cycles * sum(x[0] for x in skeleton)
    tempos = get_tempos(
        number_of_beats=num_of_beats, initial_tempo=bpm, allowed_tempo_deviation=10
    )
    hits_list = list(probabilities_matrix.keys())
    number_of_hits = len(hits_list)
    processes = build_processes(
        maxsubd=maxsubd,
        number_of_hits=number_of_hits,
        hits_list=hits_list,
        probabilities_matrix=probabilities_matrix,
    )
    y, beat_length_in_samples, added_hits_intervals = skeleton_generator(
        shift_proba=shift_proba,
        amplitude=amplitudes[-1],
        skeleton=skeleton,
        num_cycles=num_cycles,
        sr=sr,
        tempos=tempos,
    )
    y, added_hits_intervals = subdivisions_generator(
        y=y,
        maxsubd=maxsubd,
        amplitudes=amplitudes,
        amplitudes_proba_list=amplitudes_proba_list,
        added_hits_intervals=added_hits_intervals,
        beat_length_in_samples=beat_length_in_samples,
        hit_probabilities=processes,
        subdiv_proba=subdiv_proba,
        tempos=tempos,
    )
    sf.write(
        "./final_80.wav",
        y,
        samplerate=48000,
    )
    return y, num_of_beats, bpm


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


def get_tempos(number_of_beats, initial_tempo, allowed_tempo_deviation):
    # tempos = []
    # current_tempo = initial_tempo
    # i = 0
    # while i <= number_of_beats:
    #     choices = get_available_choices(
    #         current_tempo, initial_tempo, allowed_tempo_deviation
    #     )
    #     choice = random.choice(choices)
    #     if choice == 2:  # Increase
    #         deviation = random.randint(
    #             0, initial_tempo + allowed_tempo_deviation - current_tempo
    #         )
    #         tempos.append(current_tempo + deviation)
    #     elif choice == 3:  # Decrease
    #         deviation = random.randint(
    #             0, initial_tempo + allowed_tempo_deviation - current_tempo
    #         )
    #         tempos.append(current_tempo - deviation)
    #     else:  # Keep
    #         tempos.append(current_tempo)
    #     i += 1
    # # tempos.extend([initial_tempo] * 30)
    return [initial_tempo] * (number_of_beats + 2)


notes = ["D", "OTA", "OTI", "T1", "T2", "RA", "PA2"]
skeleton = [(1, "D"), (1, "PA2"), (1, "D"), (1, "PA2")]
matrix = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [100, 100, 100, 100, 100, 100, 100, 100],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
subdiv_proba = [100, 0, 0, 0, 0, 0, 0, 0]
amplitudes = [
    0.052183534022625,
    0.227138053760854,
    0.493612215184329,
    0.712676925659180,
]
amplitudes_proba_list = [0, 0, 0, 1]
# amplitudes_proba_list = [0.5, 0.5]
# folder_name = "data/first_data"
# output_file = "all_peaks_generated.json"
# bounds = np.array([0.0, 0.0742, 0.4125, 0.5987, 1.0])
# if not Path.exists(Path(output_file)):
#     get_amps_peaks(folder_name=folder_name, output_file_name=output_file)
# all_peaks = load_json_file(json_path=output_file)
# plot_amps_graph(output_file)

probabilities_matrix = get_probability_matrix(matrix=matrix, notes=notes)
print(probabilities_matrix)
y_generated, num_of_beats, initial_tempo = subdivisions_generator_adjusted(
    amplitudes=amplitudes,
    amplitudes_proba_list=amplitudes_proba_list,
    shift_proba=0,
    maxsubd=8,
    bpm=10,
    probabilities_matrix=probabilities_matrix,
    skeleton=skeleton,
    num_cycles=15,
    subdiv_proba=subdiv_proba,
    cycle_length=4,
)
