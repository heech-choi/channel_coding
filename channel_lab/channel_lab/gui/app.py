
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from codes.uncoded import Uncoded
from codes.repetition import RepetitionCode
from simulation.ber import ber_curve_awgn, ber_curve_bsc

CODE_CHOICES = ["uncoded", "rep"]
CHANNEL_CHOICES = ["awgn", "bsc"]

def make_code(name: str, n: int):
    if name == "uncoded":
        return Uncoded()
    elif name == "rep":
        return RepetitionCode(n=n)
    else:
        raise ValueError("Unknown code")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Channel Coding Lab")
        self.geometry("900x600")

        # Controls frame
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Code selection
        ttk.Label(ctrl, text="Code").pack(anchor="w")
        self.code_var = tk.StringVar(value="uncoded")
        ttk.Combobox(ctrl, textvariable=self.code_var, values=CODE_CHOICES, state="readonly").pack(fill=tk.X)

        ttk.Label(ctrl, text="Repetition n (odd)").pack(anchor="w", pady=(8,0))
        self.rep_n_var = tk.IntVar(value=3)
        ttk.Entry(ctrl, textvariable=self.rep_n_var).pack(fill=tk.X)

        # Channel selection
        ttk.Label(ctrl, text="Channel").pack(anchor="w", pady=(12,0))
        self.channel_var = tk.StringVar(value="awgn")
        ttk.Combobox(ctrl, textvariable=self.channel_var, values=CHANNEL_CHOICES, state="readonly").pack(fill=tk.X)

        # AWGN params
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

        # BSC params
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

        # Common params
        common = ttk.LabelFrame(ctrl, text="Common Params")
        common.pack(fill=tk.X, pady=(8,0))
        self.bits = tk.IntVar(value=20000)
        self.seed = tk.StringVar(value="")  # empty = None
        for lbl, var in [("Bits per point", self.bits),
                         ("Seed", self.seed)]:
            row = ttk.Frame(common); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=lbl).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.RIGHT)

        run_btn = ttk.Button(ctrl, text="Run Simulation", command=self.run_sim)
        run_btn.pack(fill=tk.X, pady=(12,4))

        # Figure
        self.fig = plt.Figure(figsize=(5.5, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_yscale("log")
        self.ax.set_xlabel("Eb/N0 (dB) or p")
        self.ax.set_ylabel("BER")
        self.ax.grid(True, which="both")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def run_sim(self):
        try:
            code_name = self.code_var.get()
            n = int(self.rep_n_var.get())
            ch = self.channel_var.get()
            bits = int(self.bits.get())
            seed_val = self.seed.get().strip()
            seed = int(seed_val) if seed_val != "" else None

            code = make_code(code_name, n=n)

            self.ax.cla()
            self.ax.set_yscale("log")
            self.ax.set_xlabel("Eb/N0 (dB) or p")
            self.ax.set_ylabel("BER")
            self.ax.grid(True, which="both")

            if ch == "awgn":
                xs = np.linspace(float(self.ebn0_start.get()),
                                 float(self.ebn0_stop.get()),
                                 int(self.ebn0_points.get()))
                snr, ber = ber_curve_awgn(code, xs, n_bits=bits, seed=seed)
                self.ax.semilogy(snr, ber, marker="o")
                self.ax.set_xlabel("Eb/N0 (dB)")
                self.ax.set_title(f"{code.name()} over AWGN (hard)")
            else:
                xs = np.linspace(float(self.bsc_p_start.get()),
                                 float(self.bsc_p_stop.get()),
                                 int(self.bsc_p_points.get()))
                p, ber = ber_curve_bsc(code, xs, n_bits=bits, seed=seed)
                self.ax.semilogy(p, ber, marker="o")
                self.ax.set_xlabel("BSC crossover prob p")
                self.ax.set_title(f"{code.name()} over BSC")

            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    App().mainloop()
