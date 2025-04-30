import os
import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plots_dir = "plots"
pattern = re.compile(r'dynamic_updates_(\w+)\.txt')

# Data structure: {db_name: {update_pct: {latency: [], qps: []}}}
data = {}

# Parse files
for filename in os.listdir(plots_dir):
    match = pattern.match(filename)
    if not match:
        continue

    db_name = match.group(1)
    filepath = os.path.join(plots_dir, filename)

    with open(filepath, 'r') as f:
        content = f.read()

    if db_name not in data:
        data[db_name] = {}

    for update_pct in ['25%', '75%']:
        section = re.search(rf'--- {update_pct} Update ---\n(.*?)(?=\n---|\Z)', content, re.DOTALL)
        if not section:
            continue

        lines = section.group(1).strip().split('\n')[:5]

        latencies = []
        qps_values = []

        for line in lines:
            qps_match = re.search(r'QPS = ([\d.]+)', line)
            latency_match = re.search(r'Latency = ([\d.]+)', line)
            if qps_match and latency_match:
                qps_values.append(float(qps_match.group(1)))
                latencies.append(float(latency_match.group(1)))

        data[db_name][update_pct] = {'latency': latencies, 'qps': qps_values}

# Plotting function
def plot_combined(metric_key, ylabel, filename):
    plt.figure(figsize=(10, 6))
    markers = {'25%': 'o', '75%': 's'}  # circle and square
    marker_size = 8
    db_colors = {}

    for i, (db_name, updates) in enumerate(data.items()):
        color = f"C{i}"
        db_colors[db_name] = color
        for update_pct in ['25%', '75%']:
            if update_pct not in updates:
                continue
            y_vals = updates[update_pct][metric_key]
            plt.plot(
                range(1, 6), y_vals,
                label=f"{db_name} - {update_pct}",
                marker=markers[update_pct],
                markersize=marker_size,
                linestyle='-',
                color=color
            )

    plt.xlabel("Interval")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over First 5 Intervals")
    plt.grid(True)

    # Custom legends
    db_legend = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='-', label=db)
        for db, color in db_colors.items()
    ]
    shape_legend = [
        mlines.Line2D([], [], color='black', marker=markers[up], linestyle='None', markersize=marker_size, label=f"{up} Update")
        for up in markers
    ]

    # Create both legends stacked in the same corner
    first_legend = plt.legend(handles=db_legend, title="Database", loc="upper right", bbox_to_anchor=(1.0, 1.0))
    plt.gca().add_artist(first_legend)

    plt.legend(handles=shape_legend, title="Update %", loc="upper right", bbox_to_anchor=(1.0, 0.55))

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(plots_dir), filename)
    plt.savefig(output_path)

plot_combined('latency', "Latency (ms)", "combined_latency_plot.pdf")
plot_combined('qps', "QPS (queries/sec)", "combined_qps_plot.pdf")
