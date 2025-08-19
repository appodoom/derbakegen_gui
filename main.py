import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from config import get_audio_data
import soundfile as sf
import random
import math
def squeleton_generator(bpm, squeleton, num_cycles, sr=48000):
    print(squeleton)
    beat_length_in_samples = int((60 / bpm) * sr)
    skeleton_length = len(squeleton)
    num_of_beats_in_audio = num_cycles * skeleton_length

    length_in_samples = int(
        sum([x[0] * beat_length_in_samples for x in squeleton]) * num_cycles
    )
    squeleton_samples_indices = []
    y = np.zeros(length_in_samples + beat_length_in_samples)

    accumulator = i = 0
    while accumulator <= num_of_beats_in_audio:
        accumulator += squeleton[i % skeleton_length][0]
        curr_beat = squeleton[i % skeleton_length][1]
        y_hit = get_audio_data(curr_beat, sr)
        hit_timestamp = int(accumulator * beat_length_in_samples)
        end_index = hit_timestamp + len(y_hit)

        # place curr_beat on accumulator
        if end_index <= len(y):
            y[hit_timestamp:end_index] += y_hit
            squeleton_samples_indices.append((hit_timestamp, end_index))
        i += 1
    sf.write("squeleton.wav", data=y, samplerate=sr)
    return y[squeleton_samples_indices[0][0]-10:], beat_length_in_samples, skeleton_length, squeleton_samples_indices

def subdivisions_generator(
    y,
    maxsubd,
    squeleton_samples_indices,
    beat_length_in_samples,
    hit_probabilities,
    even_subdivisions_percentage,
):
    subdivisions_y = np.zeros(len(y))
    index_of_current_slot_samples = 0
    duration_in_sample_by_maxsub = int(beat_length_in_samples / maxsubd)
    hits = list(hit_probabilities.keys())
    weights = list(hit_probabilities.values())
    added_hits_indicies_in_samples = []
    while index_of_current_slot_samples < len(subdivisions_y):
        if random.random() >= even_subdivisions_percentage:
            index_of_current_slot_samples += duration_in_sample_by_maxsub
            continue

        remaining = len(subdivisions_y) - index_of_current_slot_samples
        hit_choosen = random.choices(hits, weights=weights, k=1)[0]

        hit_y = get_audio_data(hit_choosen)
        add_len = min(len(hit_y), remaining)

        if hit_choosen == "S":
            index_of_current_slot_samples += duration_in_sample_by_maxsub
        else:
            for sk_start, sk_end in squeleton_samples_indices:
                if (
                    index_of_current_slot_samples >= sk_start
                    and (index_of_current_slot_samples + add_len)
                    <= sk_start + duration_in_sample_by_maxsub
                ):
                    index_of_current_slot_samples += duration_in_sample_by_maxsub
                    break
            else:
                subdivisions_y[
                    index_of_current_slot_samples : index_of_current_slot_samples
                    + add_len
                ] += hit_y[:add_len]
                added_hits_indicies_in_samples.append(
                    (
                        index_of_current_slot_samples,
                        index_of_current_slot_samples + add_len,
                    )
                )
                index_of_current_slot_samples += duration_in_sample_by_maxsub
    y += subdivisions_y
    sf.write(
        f"generated.wav",
        y,
        samplerate=48000,
    )
    return y, added_hits_indicies_in_samples


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cycle Parameters")
        self.root.geometry("500x400")
        self.root.configure(padx=20, pady=20)
        
        # Create input fields
        self.create_input("Number of cycles in the output:", "cycles")
        self.create_input("Tempo (in bpm):", "tempo")
        self.create_input("Cycle length (in beats):", "cycle_length")
        self.create_input("Smallest note (0-1):", "smallest_note", self.validate_0_to_1)
        self.create_input("Percentage of even notes:", "variation_percent", self.validate_0_to_100)
        self.create_input("Percentage of Doom", "percent_doom", self.validate_0_to_100)
        self.create_input("Percentage of Open Tak", "percent_ota", self.validate_0_to_100)
        self.create_input("Percentage of Open Tik", "percent_oti", self.validate_0_to_100)
        self.create_input("Percentage of Tik1", "percent_t1", self.validate_0_to_100)
        self.create_input("Percentage of Tik2", "percent_t2", self.validate_0_to_100)
        self.create_input("Percentage of Ra2", "percent_ra", self.validate_0_to_100)
        self.create_input("Percentage of Pa2", "percent_pa2", self.validate_0_to_100)


        self.map = {
            "Silence": "S",
            "Doom": "D",
            "Open Tak": "OTA",
            "Open Tik": "OTI",
            "Pa2": "PA2",
            "Ra2": "RA",
            "Tik1": "T1",
            "Tik2": "T2"
        }
        
        # Submit button
        self.submit_btn = tk.Button(
            self.root, 
            text="Next", 
            command=self.validate_inputs,
            width=10
        )
        self.submit_btn.pack(pady=(20, 10))
        
        self.root.mainloop()
    
    def create_input(self, label_text, name, validation_func=None):
        """Create labeled input fields with validation"""
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill=tk.X)
        
        label = tk.Label(frame, text=label_text, width=30, anchor='w')
        label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add validation to entry
        if validation_func:
            vcmd = (self.root.register(validation_func), '%P')
        else:
            vcmd = (self.root.register(self.validate_number), '%P')
            
        entry = tk.Entry(
            frame, 
            validate="key", 
            validatecommand=vcmd
        )
        entry.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(10, 0))
        setattr(self, f"{name}_entry", entry)
    
    def validate_number(self, new_value):
        """Validate that the input is a number (or empty string)"""
        if new_value == "":
            return True
        try:
            float(new_value)
            return True
        except ValueError:
            return False
    
    def validate_0_to_1(self, new_value):
        """Validate that input is between 0 and 1"""
        if new_value == "":
            return True
        try:
            val = float(new_value)
            return 0 <= val <= 1
        except ValueError:
            return False
    
    def validate_0_to_100(self, new_value):
        """Validate that input is between 0 and 100"""
        if new_value == "":
            return True
        try:
            val = float(new_value)
            return 0 <= val <= 100
        except ValueError:
            return False
    
    def validate_inputs(self):
        """Validate all inputs when submit button is pressed"""
        try:
            # Get values
            self.cycles = int(float(self.cycles_entry.get()))
            self.tempo = float(self.tempo_entry.get())
            self.cycle_length = int(float(self.cycle_length_entry.get()))
            self.smallest_note = float(self.smallest_note_entry.get())
            self.variation_percent = float(self.variation_percent_entry.get())
            self.percentages = [self.percent_doom_entry.get(), self.percent_ota_entry.get(), self.percent_oti_entry.get(), self.percent_t1_entry.get(), self.percent_t2_entry.get(), self.percent_ra_entry.get(), self.percent_pa2_entry.get()]
            self.percentages = [float(i)/100 for i in self.percentages]
            # Validate cycles
            if self.cycles <= 0:
                raise ValueError("Number of cycles must be positive")
                
            # Validate tempo (50-150 BPM)
            if self.tempo < 50 or self.tempo > 150:
                raise ValueError("Tempo must be between 50 and 150 BPM")
                
            # Validate cycle length
            if self.cycle_length <= 0:
                raise ValueError("Cycle length must be positive")
                
            # Validate smallest note
            if not 0 <= self.smallest_note <= 1:
                raise ValueError("Smallest note must be between 0 and 1")
                
            # Validate variation percentage
            if not 0 <= self.variation_percent <= 100:
                raise ValueError("Variation percentage must be between 0 and 100%")
            
            if sum(self.percentages) > 1:
                raise ValueError("Percentages must be less than 100%")
            # If validation succeeds, create the cycle points window
            self.create_cycle_points_window()
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
    
    def create_cycle_points_window(self):
        """Create window with true infinite subdivision capability"""
        self.root.withdraw()
        self.cycle_window = tk.Toplevel()
        self.cycle_window.title("Cycle Points")
        self.cycle_window.geometry("800x600") 
        self.cycle_window.configure(padx=20, pady=20)

        # Main container frame
        main_frame = tk.Frame(self.cycle_window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Initialize with integer points (1.0, 2.0, etc.)
        self.points = []
        for i in range(1, self.cycle_length + 1):
            self.points.append({
                'value': float(i),
                'var': tk.StringVar(value="Silence"),
                'frame': None,
                'label': None,
                'dropdown': None
            })

        self.redraw_all_points()

        # Submit button
        submit_btn = tk.Button(
            main_frame,
            text="Submit",
            command=self.on_cycle_window_submit,
            width=15
        )
        submit_btn.pack(pady=(20, 10))

        self.cycle_window.protocol("WM_DELETE_WINDOW", self.on_cycle_window_close)

    def redraw_all_points(self):
        """Completely redraw the interface with current points and subdivision buttons"""
        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Sort points by value
        self.points.sort(key=lambda x: x['value'])

        # Draw each point and subdivision buttons
        for i, point in enumerate(self.points):
            # Create point frame
            point['frame'] = tk.Frame(self.scrollable_frame)
            point['frame'].pack(fill=tk.X, pady=5)

            # Point label
            point['label'] = tk.Label(point['frame'], text=f"{point['value']:.5f}:", width=10, anchor='w')
            point['label'].pack(side=tk.LEFT)

            # Sound selection dropdown
            sound_options = ["Silence", "Doom", "Open Tak", "Open Tik", "Pa2", "Ra2", "Tik1", "Tik2"]
            point['dropdown'] = ttk.Combobox(
                point['frame'],
                textvariable=point['var'],
                values=sound_options,
                state="readonly",
                width=15
            )
            point['dropdown'].pack(side=tk.LEFT, padx=10)

            # Add subdivision button if not last point
            if i < len(self.points) - 1:
                next_point = self.points[i + 1]
                midpoint = (point['value'] + next_point['value']) / 2
                
                # Always show button - no gap size restriction
                btn_frame = tk.Frame(self.scrollable_frame)
                btn_frame.pack(fill=tk.X, pady=2)

                # Spacer to align with dropdowns
                spacer = tk.Frame(btn_frame, width=80)
                spacer.pack(side=tk.LEFT)

                btn = tk.Button(
                    btn_frame,
                    text=f"Add subdivision at {midpoint:.5f}",
                    command=lambda mid=midpoint: self.add_subdivision(mid),
                    width=30
                )
                btn.pack(side=tk.LEFT, padx=10)

    def add_subdivision(self, value):
        """Add a new subdivision point at the specified value"""
        # Check if point already exists (with floating point tolerance)
        if not any(abs(point['value'] - value) < 1e-5 for point in self.points):
            # Add the new point
            self.points.append({
                'value': value,
                'var': tk.StringVar(value="Silence"),
                'frame': None,
                'label': None,
                'dropdown': None
            })
            # Redraw the entire interface
            self.redraw_all_points()

    def on_cycle_window_submit(self):
        """Handle submission of cycle points"""
        selected_values = []
        for point in sorted(self.points, key=lambda x: x['value']):
            selected_values.append((point['value'], point['var'].get()))
        
        message = "\n".join([f"{value:.5f}: {sound}" for value, sound in selected_values])
        messagebox.showinfo("Selected Values", message)

        # Save the second window preferences
        self.selected_points = selected_values
        squeleton = []
        acc = 0
        for pos, val in selected_values:
            new_val = self.map.get(val, "S")
            if new_val == "S" and pos != self.cycle_length: continue
            pos = pos - acc
            acc += pos
            squeleton.append((pos, new_val))

        squeleton_y, beat_length_in_samples, skeleton_length, squeleton_samples_indices = squeleton_generator(float(self.tempo), squeleton ,float(self.cycles))
        
        hit_probabilities = {
            "D": self.percentages[0],
            "OTA": self.percentages[1],
            "OTI": self.percentages[2],
            "T1": self.percentages[3],
            "T2": self.percentages[4],
            "RA": self.percentages[5],
            "PA2": self.percentages[6],
            "S": 1-sum(self.percentages)
        }

        maxsubd = 1/self.smallest_note

        y, added_hits_indicies_in_samples = subdivisions_generator(
            hit_probabilities=hit_probabilities,
            y=squeleton_y,
            squeleton_samples_indices=squeleton_samples_indices,
            beat_length_in_samples=beat_length_in_samples,
            maxsubd=maxsubd,
            even_subdivisions_percentage=self.variation_percent/100,
        )

        smallest_odd_note = 0

        for i in range(math.floor(maxsubd), -1, -1):
            if i % 3 == 0:
                smallest_odd_note = i
                break
        generated_y = y
        if smallest_odd_note > 0:
            final_y, _ = subdivisions_generator(
                hit_probabilities=hit_probabilities,
                y=squeleton_y,
                squeleton_samples_indices=added_hits_indicies_in_samples
                + squeleton_samples_indices,
                beat_length_in_samples=beat_length_in_samples,
                maxsubd=smallest_odd_note,
                even_subdivisions_percentage=(100-self.variation_percent)/100,
            )
            generated_y = final_y
        
        sf.write("generated.wav", data=generated_y, samplerate=48000)

        self.cycle_window.destroy()
        self.root.deiconify()
    
    def on_cycle_window_close(self):
        """Handle closing the cycle points window"""
        self.cycle_window.destroy()
        self.root.deiconify()

# Create and run the GUI
app = GUI()