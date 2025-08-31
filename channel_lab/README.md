# Channel Lab (Python) — Minimal Dependencies

**Purpose:** Modular platform to test channel codes over AWGN/BSC, plot BER, with both GUI (tkinter) and CLI.
Dependencies: `numpy`, `matplotlib`, `tkinter` (usually included with Python on desktop).

## Install
```bash
pip install numpy matplotlib
```

## Run GUI
```bash
python -m gui.app
```

## Run CLI (examples)
```bash
# Uncoded over AWGN, 0..6 dB
python cli.py --code uncoded --channel awgn --ebn0_start 0 --ebn0_stop 6 --ebn0_points 7 --bits 30000

# Repetition(3) over AWGN
python cli.py --code rep --n 3 --channel awgn --ebn0_start 0 --ebn0_stop 6 --ebn0_points 7 --bits 30000

# Repetition(5) over BSC
python cli.py --code rep --n 5 --channel bsc --bsc_p_start 0 --bsc_p_stop 0.2 --bsc_p_points 11 --bits 50000
```

## Extend
- Add new codes under `codes/` by subclassing `BaseCode` and implementing `encode/decode/rate`.
- Add new channels under `channels/` and a corresponding BER driver in `simulation/ber.py`.
```python
class MyCode(BaseCode):
    ...
```

## Pyinstaller -> exe file
py -m PyInstaller -F -w -n ChannelLabGUI channel_lab/gui/app.py ^
  --collect-submodules matplotlib ^
  --collect-data matplotlib
