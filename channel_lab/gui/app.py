import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg") #Use Tkinter GUI toolkit as a backend, Agg is Anti-Grain Geometry - a fast non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from codes.uncoded import Uncoded
from codes.repetition import RepetitionCode
from codes.ldpc import LDPCCode
from simulation.ber import ber_curve_awgn, ber_curve_bsc

CODE_CHOICES = ["uncoded", "rep", "LDPC"]
CHANNEL_CHOICES = ["bi-awgn", "bsc", "bec"]

# Get the code instance with its methods
def make_code(name: str, n: int):
    if name == "uncoded":
        return Uncoded()
    elif name == "rep":
        return RepetitionCode(n=n)
    elif name == "LDPC":
        return LDPCCode()
    else:
        raise ValueError("Unknown code")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Channel Coding Platform")
        self.geometry("980x640")

        # Maximum N lines can stay  
        self.lines = []      # list of (line_artist, label)
        self.max_lines = 30
        self.run_idx = 1     # for labeling: Run 1, Run 2, ...

        # Controls
        ## Frame
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        ## Option 1: Select Code
        ttk.Label(ctrl, text="Code").pack(anchor="w")
        self.code_var = tk.StringVar(value="uncoded")
        ttk.Combobox(ctrl, textvariable=self.code_var, values=CODE_CHOICES, state="readonly").pack(fill=tk.X)

        ## Option 2: Repetition Params
        ttk.Label(ctrl, text="Repetition n (odd)").pack(anchor="w", pady=(8,0))
        self.rep_n_var = tk.IntVar(value=3)
        ttk.Entry(ctrl, textvariable=self.rep_n_var).pack(fill=tk.X)

        ## Option 3: Select Channel
        ttk.Label(ctrl, text="Channel").pack(anchor="w", pady=(12,0))
        self.channel_var = tk.StringVar(value="bi-awgn")
        ttk.Combobox(ctrl, textvariable=self.channel_var, values=CHANNEL_CHOICES, state="readonly").pack(fill=tk.X)

        ## Option 4: AWGN params
        awgn_frame = ttk.LabelFrame(ctrl, text="AWGN Params")
        awgn_frame.pack(fill=tk.X, pady=(8,0))
        self.ebn0_start = tk.DoubleVar(value=0.0)
        self.ebn0_stop  = tk.DoubleVar(value=6.0)
        self.ebn0_points= tk.IntVar(value=7)
        for lbl, var in [("Eb/N0 start (dB)", self.ebn0_start),
                         ("Eb/N0 stop (dB)", self.ebn0_stop),
                         ("Points", self.ebn0_points)]:
            row = ttk.Frame(awgn_frame); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=lbl).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT)

        ## Option 5: BSC params
        bsc_frame = ttk.LabelFrame(ctrl, text="BSC Params")
        bsc_frame.pack(fill=tk.X, pady=(8,0))
        self.bsc_p_start = tk.DoubleVar(value=0.0)
        self.bsc_p_stop  = tk.DoubleVar(value=0.2)
        self.bsc_p_points= tk.IntVar(value=11)
        for lbl, var in [("p start", self.bsc_p_start),
                         ("p stop", self.bsc_p_stop),
                         ("Points", self.bsc_p_points)]:
            row = ttk.Frame(bsc_frame); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=lbl).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT)

        ## Option 6: Common params
        common = ttk.LabelFrame(ctrl, text="Common Params")
        common.pack(fill=tk.X, pady=(8,0))
        self.bits = tk.IntVar(value=10000)
        self.seed = tk.StringVar(value="")  # empty = None
        for lbl, var in [("Bits per point", self.bits),
                         ("Seed", self.seed)]:
            row = ttk.Frame(common); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=lbl).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.RIGHT)

        ## Buttons (Run + Erase)
        btn_row = ttk.Frame(ctrl); btn_row.pack(fill=tk.X, pady=(12,4))
        ttk.Button(btn_row, text="Run Simulation", command=self.run_sim).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,4))
        ttk.Button(btn_row, text="Erase Plot", command=self.erase_plot).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ## Figure/axes
        self.fig = plt.Figure(figsize=(6.4, 5.0), dpi=100, constrained_layout=True) #constrained_layout=True to make shure the legend box is contained inside the window
        self.ax = self.fig.add_subplot(111)
        # self.fig.subplots_adjust(right=0.78) #activate this line to keep legends box outside
        self.ax.set_yscale("log")
        self.ax.set_xlabel("Eb/N0 (dB) or p")
        self.ax.set_ylabel("BER")
        self.ax.grid(True, which="both")
        self.ax.set_title("Channel Coding BER")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Wiping the entire lines
    def erase_plot(self):
        for ln, _ in self.lines:
            try:
                ln.remove()
            except Exception:
                pass
        self.lines.clear()
        self.run_idx = 1
        self.ax.cla()
        self.ax.set_yscale("log")
        self.ax.set_xlabel("Eb/N0 (dB) or p")
        self.ax.set_ylabel("BER")
        self.ax.grid(True, which="both")
        self.ax.set_title("Channel Coding BER")
        if self.ax.get_legend():
            self.ax.legend().remove()
        self.canvas.draw()

    # Popping the oldest line
    def _ensure_capacity(self):
        if len(self.lines) >= self.max_lines:
            oldest, _ = self.lines.pop(0)
            try:
                oldest.remove()
            except Exception:
                pass

    # Logic part
    def run_sim(self):
        try:
            ## getting the essential data
            code_name = self.code_var.get()
            n = int(self.rep_n_var.get())
            ch = self.channel_var.get()
            bits = int(self.bits.get())
            seed_val = self.seed.get().strip()
            seed = int(seed_val) if seed_val != "" else None

            ## generating code instance
            code = make_code(code_name, n=n)

            if ch == "bi-awgn": ### Branch: channel (bi-awgn, bsc)
                xs = np.linspace(float(self.ebn0_start.get()),
                                 float(self.ebn0_stop.get()),
                                 int(self.ebn0_points.get()))
                snr, ber = ber_curve_awgn(code, xs, n_bits=bits, seed=seed) # Simulating the bits with the code instance
                self._ensure_capacity()
                (line,) = self.ax.semilogy(snr, ber, marker="o",
                                           label=f"Line {self.run_idx}: {code.name()} AWGN")
                self.lines.append((line, f"Line {self.run_idx}"))
                self.ax.set_xlabel("Eb/N0 (dB)")
            else:
                xs = np.linspace(float(self.bsc_p_start.get()),
                                 float(self.bsc_p_stop.get()),
                                 int(self.bsc_p_points.get()))
                p, ber = ber_curve_bsc(code, xs, n_bits=bits, seed=seed) # Simulating the bits with the code instance
                self._ensure_capacity()
                (line,) = self.ax.semilogy(p, ber, marker="o",
                                           label=f"Line {self.run_idx}: {code.name()} BSC")
                self.lines.append((line, f"Line {self.run_idx}"))
                self.ax.set_xlabel("BSC crossover prob p")

            self.ax.set_ylabel("BER")
            self.ax.grid(True, which="both")
            self.ax.set_title("Channel Coding BER")
            self.ax.legend(loc="center left", bbox_to_anchor=(1,0.5))  #loc="best" to keep legend box inside the plot
            self.run_idx += 1

            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    App().mainloop()
