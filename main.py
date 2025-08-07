import tkinter as tk
from tkinter import messagebox, ttk

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
        self.create_input("Percentage of power two notes:", "variation_percent", self.validate_0_to_100)
        
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
                'var': tk.StringVar(value=""),
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
            sound_options = ["", "Doom", "Open Tak", "Open Tik", "Pa2", "Ra2", "Tik1", "Tik2"]
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
                'var': tk.StringVar(value=""),
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
        self.cycle_window.destroy()
        self.root.deiconify()
    
    def on_cycle_window_close(self):
        """Handle closing the cycle points window"""
        self.cycle_window.destroy()
        self.root.deiconify()

# Create and run the GUI
app = GUI()