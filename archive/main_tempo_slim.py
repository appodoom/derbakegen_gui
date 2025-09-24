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
    subdiv_proba,
    cycle_length,
    number_of_beats,
    sr=48000,
    beat_tempo_vari_proba=0.1
):  
    subdiv_array=[]
    for i in range(len(subdiv_proba)):
        subdiv_array.append(i)
    maxsubdi=random.choices(population=subdiv_array,weights=subdiv_proba,k=1)[0]
    added_hits_intervals = sorted(added_hits_intervals, key=lambda x: x[0])
    subdivisions_y = []
    sample_of_curr_subd = 0
    maxsubd_length = int(beat_length_in_samples / (maxsubd-maxsubdi))
    hits = list(hit_probabilities[maxsubdi].keys())
    weights = list(hit_probabilities[maxsubdi].values())
    new_added_hits_intervals = []
    current_tempo=60*sr/beat_length_in_samples
    initial_tempo=current_tempo
    cycle_tempo=current_tempo
    # number_of_beats=(len(y))/beat_length_in_samples
    for i in range(int(number_of_beats)):
        if i%(cycle_length)==0:
            cycle_tempo=get_tempo(current_tempo=current_tempo, initial_tempo=initial_tempo,allowed_tempo_deviation=5)
        current_tempo=cycle_tempo
        sample_of_curr_subd = 0
        beat_length_in_samples=60*sr/current_tempo
        if random.random()<beat_tempo_vari_proba:
                current_tempo=get_tempo(current_tempo=current_tempo, initial_tempo=initial_tempo, allowed_tempo_deviation=5)
                beat_length_in_samples=60*sr/current_tempo
        maxsubdi=random.choices(population=subdiv_array,weights=subdiv_proba,k=1)[0]
        hits = list(hit_probabilities[maxsubdi].keys())
        weights = list(hit_probabilities[maxsubdi].values())
        maxsubd_length = int(beat_length_in_samples / (maxsubd-maxsubdi))
        # remaining = len(subdivisions_y) - sample_of_curr_subd
        while sample_of_curr_subd < beat_length_in_samples:
            random_proba_list = get_random_proba_list(weights)
            chosen_hit = random.choices(hits, weights=random_proba_list, k=1)[0]
            # print(f"choosen hit : {chosen_hit}")
            if chosen_hit == "S":
                sample_of_curr_subd += maxsubd_length
            else:
                hit_y = get_audio_data(chosen_hit)
                add_len = len(hit_y)

                for start, _ in added_hits_intervals:
                    if start <= sample_of_curr_subd <= start + maxsubd_length:
                        sample_of_curr_subd += maxsubd_length
                        break
                else:
                    subdivisions_y.extend(
                        hit_y[:add_len]
                    )
                    new_added_hits_intervals.append(
                        (
                            sample_of_curr_subd,
                            sample_of_curr_subd + add_len,
                        )
                    )
                    sample_of_curr_subd += maxsubd_length
    # sf.write(
    #     f"./generated/t_{maxsubd}.wav",
    #     y,
    #     samplerate=48000,
    # )
    new_added_hits_intervals.extend(added_hits_intervals)
    return np.array(subdivisions_y), new_added_hits_intervals


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
    cycle_length,
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
            subdiv_proba=subdiv_proba,
            cycle_length=cycle_length,
            sr=sr,
            number_of_beats=num_of_beats
        )
    sf.write(
        "./final_4.wav",
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

def get_tempo(current_tempo, initial_tempo, allowed_tempo_deviation):
    choices = get_available_choices(
                current_tempo, initial_tempo, allowed_tempo_deviation
            )
    choice = random.choice(choices)
    if choice == 2:  # Increase
        deviation = random.randint(0, initial_tempo+allowed_tempo_deviation-current_tempo)
        return current_tempo + deviation
    elif choice == 3:  # Decrease
        deviation = random.randint(0, initial_tempo+allowed_tempo_deviation-current_tempo)
        return current_tempo - deviation
    else:  # Keep
        return current_tempo







notes = ["D", "OTA", "OTI", "T1", "T2", "RA", "PA2"]
squleton = [(1, "D"), (0.5, "OTA"),(1,"OTA"),(0.5,"D"),(1,"OTA")]
matrix = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [20, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875],
    [1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875],
    [0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
]
subdiv_proba=[30,7.5,7.5,7.5,7.5,7.5,7.5,7.5]


probabilities_matrix = get_probability_matrix(matrix=matrix, notes=notes)
print(probabilities_matrix)
y_generated, num_of_beats, initial_tempo =subdivisions_generator_adjusted(
    maxsubd=8,
    bpm=120,
    probabilities_matrix=probabilities_matrix,
    squeleton=squleton,
    num_cycles=20,
    subdiv_proba=subdiv_proba,
    cycle_length=4
    
)