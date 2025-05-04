import glob
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Regex to match both with/without "queries/sec"
pattern = re.compile(
    r'Interval\s+(\d+):\s+QPS\s*=\s*([-\d\.]+)(?:\s*queries/sec)?\s*[, ]+\s*Latency\s*=\s*([-\d\.]+)\s*ms',
    re.IGNORECASE
)

# Scan and process each file
for filepath in glob.glob('dynamic_updates_*.txt'):
    dbname = os.path.splitext(os.path.basename(filepath))[0].replace('dynamic_updates_', '')
    intervals, qps_vals, latencies = [], [], []
    events = {-1: [], -2: [], -3: [], -4: []}

    with open(filepath) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            i = int(m.group(1))
            q = float(m.group(2))
            lat = float(m.group(3))
            if q < 0:
                events[int(q)].append(i)
            else:
                intervals.append(i)
                qps_vals.append(q)
                latencies.append(lat)

    # Pair up start/end markers
    delete_regions = list(zip(events[-1], events[-2]))
    insert_regions = list(zip(events[-3], events[-4]))

    # Plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Plot lines with distinct colors
    qps_line, = ax1.plot(intervals, qps_vals, color='tab:green', label='QPS')
    latency_line, = ax2.plot(intervals, latencies, color='tab:orange', label='Latency')

    # Shade delete (red) and insert (blue) regions
    for start, end in delete_regions:
        ax1.axvspan(start, end, color='red', alpha=0.3)
    for start, end in insert_regions:
        ax1.axvspan(start, end, color='blue', alpha=0.2)

    # Create legend patches for regions
    delete_patch = mpatches.Patch(color='red', alpha=0.3, label='Delete Region')
    insert_patch = mpatches.Patch(color='blue', alpha=0.2, label='Insert Region')

    # Combine all legend entries
    handles = [qps_line, latency_line, delete_patch, insert_patch]
    labels = ['QPS', 'Latency', 'Delete Region', 'Insert Region']
    ax1.legend(handles, labels, loc='upper left')

    ax1.set_xlabel('Interval')
    ax1.set_ylabel('QPS')
    ax2.set_ylabel('Latency (ms)')
    plt.title(f'{dbname}: QPS & Latency with Delete/Insert Regions')
    plt.tight_layout()

    # Save output
    outname = f'{dbname}_qps_latency.png'
    plt.savefig(outname)
    plt.close(fig)
    print(f'Saved graph: {outname}')
