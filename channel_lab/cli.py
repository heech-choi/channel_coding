
import argparse
import numpy as np
import matplotlib.pyplot as plt

from codes.uncoded import Uncoded
from codes.repetition import RepetitionCode
from simulation.ber import ber_curve_awgn, ber_curve_bsc

CODE_MAP = {
    "uncoded": lambda **kw: Uncoded(),
    "rep": lambda n=3, **kw: RepetitionCode(n=n),
}

def main():
    parser = argparse.ArgumentParser(description="Channel coding BER simulator (CLI)")
    parser.add_argument("--code", choices=CODE_MAP.keys(), default="uncoded")
    parser.add_argument("--n", type=int, default=3, help="Repetition length for rep code (odd)")
    parser.add_argument("--channel", choices=["awgn", "bsc"], default="awgn")
    parser.add_argument("--ebn0_start", type=float, default=0.0)
    parser.add_argument("--ebn0_stop", type=float, default=6.0)
    parser.add_argument("--ebn0_points", type=int, default=7)
    parser.add_argument("--bsc_p_start", type=float, default=0.0)
    parser.add_argument("--bsc_p_stop", type=float, default=0.2)
    parser.add_argument("--bsc_p_points", type=int, default=11)
    parser.add_argument("--bits", type=int, default=20000, help="information bits per SNR point")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_plot", action="store_true", help="skip plotting; just print numbers")

    args = parser.parse_args()
    code = CODE_MAP[args.code](n=args.n)

    if args.channel == "awgn":
        xs = np.linspace(args.ebn0_start, args.ebn0_stop, args.ebn0_points)
        snr, ber = ber_curve_awgn(code, xs, n_bits=args.bits, seed=args.seed)
        print("Eb/N0(dB)  BER")
        for s, b in zip(snr, ber):
            print(f"{s:.2f}  {b:.6g}")
        if not args.no_plot:
            plt.semilogy(snr, ber, marker="o")
            plt.xlabel("Eb/N0 (dB)")
            plt.ylabel("BER")
            plt.title(f"{code.name()} over AWGN (hard)")
            plt.grid(True, which="both")
            plt.show()
    else:
        xs = np.linspace(args.bsc_p_start, args.bsc_p_stop, args.bsc_p_points)
        p, ber = ber_curve_bsc(code, xs, n_bits=args.bits, seed=args.seed)
        print("p  BER")
        for s, b in zip(p, ber):
            print(f"{s:.4f}  {b:.6g}")
        if not args.no_plot:
            plt.semilogy(p, ber, marker="o")
            plt.xlabel("BSC crossover prob p")
            plt.ylabel("BER")
            plt.title(f"{code.name()} over BSC")
            plt.grid(True, which="both")
            plt.show()

if __name__ == "__main__":
    main()
