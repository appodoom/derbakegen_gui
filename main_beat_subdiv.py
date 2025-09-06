import numpy as np
from config import get_audio_data
import soundfile as sf
import random
import librosa

def get_probability_matrix(matrix, notes):
    return dict(zip(notes, matrix))


def get_random_proba_list(weights):
    output = []
    for weight in weights:
        choice = random.uniform(0, weight)
        output.append(choice)
    return output


def squeleton_generator(bpm, squeleton, num_cycles, sr=48000):
    beat_length_in_samples = int((60 / bpm) * sr)
    skeleton_length = len(squeleton)
    num_of_beats_in_audio = num_cycles * sum(x[0] for x in squeleton)

    # [(1, D), (2.5, T), (2, S)]
    length_in_samples = int(
        sum([x[0] * beat_length_in_samples for x in squeleton]) * num_cycles
    )
    squeleton_hits_intervals = []
    y = np.zeros(length_in_samples + beat_length_in_samples)

    curr_beat = i = 0
    while curr_beat <= num_of_beats_in_audio:
        curr_beat += squeleton[i % skeleton_length][0]
        curr_hit = squeleton[i % skeleton_length][1]
        y_hit = get_audio_data(curr_hit, sr)
        hit_timestamp = int(curr_beat * beat_length_in_samples)
        end_of_hit_timestamp = hit_timestamp + len(y_hit)

        if end_of_hit_timestamp <= len(y):
            y[hit_timestamp:end_of_hit_timestamp] += y_hit
            squeleton_hits_intervals.append((hit_timestamp, end_of_hit_timestamp))
        i += 1
    y_without_initial_silence = y[squeleton_hits_intervals[0][0] - 10 :]
    # sf.write(
    #     "./generated/squeleton1.wav",
    #     data=y_without_initial_silence,
    #     samplerate=sr,
    # )
    return (
        y_without_initial_silence,
        beat_length_in_samples,
        squeleton_hits_intervals,
    )


def subdivisions_generator(
    y,
    maxsubd,
    added_hits_intervals,
    beat_length_in_samples,
    hit_probabilities,
    subdiv_proba
):  
    subdiv_array=[]
    for i in range(len(subdiv_proba)):
        subdiv_array.append(i)
    maxsubdi=random.choices(population=subdiv_array,weights=subdiv_proba,k=1)[0]
    added_hits_intervals = sorted(added_hits_intervals, key=lambda x: x[0])
    subdivisions_y = np.zeros(len(y))
    sample_of_curr_subd = 0
    maxsubd_length = int(beat_length_in_samples / (maxsubd-maxsubdi))
    hits = list(hit_probabilities[maxsubdi].keys())
    weights = list(hit_probabilities[maxsubdi].values())
    new_added_hits_intervals = []
    while sample_of_curr_subd < len(subdivisions_y):
        if sample_of_curr_subd%beat_length_in_samples==0:
            maxsubdi=random.choices(population=subdiv_array,weights=subdiv_proba,k=1)[0]
            maxsubd_length = int(beat_length_in_samples / (maxsubd-maxsubdi))
            hits = list(hit_probabilities[maxsubdi].keys())
            weights = list(hit_probabilities[maxsubdi].values())
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
    squeleton,
    num_cycles,
    subdiv_proba,
    sr=48000,
):
    num_of_beats = num_cycles * sum(x[0] for x in squeleton)
    hits_list = list(probabilities_matrix.keys())
    number_of_hits = len(hits_list)
    processes = build_processes(
        maxsubd=maxsubd,
        number_of_hits=number_of_hits,
        hits_list=hits_list,
        probabilities_matrix=probabilities_matrix,
    )
    y, beat_length_in_samples, added_hits_intervals = squeleton_generator(
        bpm=bpm, squeleton=squeleton, num_cycles=num_cycles, sr=sr
    )
    y, added_hits_intervals = subdivisions_generator(
            y=y,
            maxsubd=maxsubd,
            added_hits_intervals=added_hits_intervals,
            beat_length_in_samples=beat_length_in_samples,
            hit_probabilities=processes,
            subdiv_proba=subdiv_proba
        )
    sf.write(
        "./final_4.wav",
        y,
        samplerate=48000,
    )
    return y, num_of_beats, bpm


# def get_available_choices(current_tempo, initial_tempo, allowed_tempo_deviation):
#     lower = initial_tempo - allowed_tempo_deviation
#     upper = initial_tempo + allowed_tempo_deviation
#     choices = [1]  # keep
#     if current_tempo <= lower:
#         choices.append(2)  # increase
#     elif current_tempo >= upper:
#         choices.append(3)  # decrease
#     else:
#         choices.extend([2, 3])
#     return choices


# def adjust_generated_tempo(allowed_tempo_deviation, y, num_of_beats, initial_tempo, sr):
#     beat_length_in_samples = int((60 / initial_tempo) * sr)
#     current_beat = 1
#     current_tempo = initial_tempo
#     new_y=[]
#     while current_beat < num_of_beats:
#         start = current_beat * beat_length_in_samples
#         end = start + beat_length_in_samples
#         if start >= len(y):
#             break
#         choices = get_available_choices(
#             current_tempo, initial_tempo, allowed_tempo_deviation
#         )
#         choice = random.choice(choices)

#         if choice == 2:  # Increase
#             deviation = random.randint(0, initial_tempo+allowed_tempo_deviation-current_tempo)
#             current_tempo = current_tempo + deviation
#         elif choice == 3:  # Decrease
#             deviation = random.randint(0, initial_tempo+allowed_tempo_deviation-current_tempo)
#             current_tempo = current_tempo - deviation
#         else:  # Keep
#             current_tempo = current_tempo
#         new_sr = (beat_length_in_samples * current_tempo) / 60.0
#         seg = y[start:end]
#         seg_rs = librosa.resample(y=seg, orig_sr=sr, target_sr=new_sr)
#         new_y=np.concatenate([new_y,seg_rs])
#         current_beat += 1

#     sf.write(
#         "./final_1.wav",
#         new_y,
#         samplerate=sr,
#     )

#     return new_y

notes = ["D", "OTA", "OTI", "T1", "T2", "RA", "PA2"]
squleton = [(1, "D"), (1, "T1")]
matrix = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [20, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875],
    [1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875],
    [0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
]
subdiv_proba=[40,7.5,7.5,7.5,7.5,7.5,7.5,7.5]


probabilities_matrix = get_probability_matrix(matrix=matrix, notes=notes)
print(probabilities_matrix)
y_generated, num_of_beats, initial_tempo =subdivisions_generator_adjusted(
    maxsubd=8,
    bpm=70,
    probabilities_matrix=probabilities_matrix,
    squeleton=squleton,
    num_cycles=10,
    subdiv_proba=subdiv_proba
)

# adjust_generated_tempo(
#     y=y_generated,
#     allowed_tempo_deviation=5,
#     num_of_beats=num_of_beats,
#     initial_tempo=initial_tempo,
#     sr=48000,
# )
