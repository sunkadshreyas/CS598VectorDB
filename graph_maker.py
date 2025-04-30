import os
import re
import matplotlib.pyplot as plt

# Set the path to your plots folder
plots_dir = "plots"
pattern = re.compile(r'dynamic_updates_(\w+)\.txt')

# Data structure to store latency and QPS for each DB and update percentage
data = {
    '25%': {},
    '75%': {}
}

# Parse each file in the plots directory
for filename in os.listdir(plots_dir):
    match = pattern.match(filename)
    if not match:
        continue

    db_name = match.group(1)
    filepath = os.path.join(plots_dir, filename)

    with open(filepath, 'r') as file:
        content = file.read()

    for update_pct in ['25%', '75%']:
        # Extract section for the current update percentage
        update_section = re.search(rf'--- {update_pct} Update ---\n(.*?)(?=\n---|\Z)', content, re.DOTALL)
        if not update_section:
            continue

        intervals = update_section.group(1).strip().split('\n')[:5]

        latencies = []
        qps_values = []

        for line in intervals:
            qps_match = re.search(r'QPS = ([\d.]+)', line)
            latency_match = re.search(r'Latency = ([\d.]+)', line)
            if qps_match and latency_match:
                qps_values.append(float(qps_match.group(1)))
                latencies.append(float(latency_match.group(1)))

        data[update_pct][db_name] = {
            'latency': latencies,
            'qps': qps_values
        }

# Plotting
def plot_metric(metric_key, ylabel, filename):
    for update_pct in ['25%', '75%']:
        plt.figure(figsize=(8, 6))
        for db_name, metrics in data[update_pct].items():
            plt.plot(range(1, 6), metrics[metric_key], marker='o', label=db_name)

        plt.xlabel("Interval")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} over First 5 Intervals ({update_pct} Updates)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_path = os.path.join(os.path.dirname(plots_dir), f"{metric_key}_plot_{update_pct.replace('%', '')}.pdf")
        plt.savefig(output_path)

plot_metric('latency', "Latency (ms)", "latency_plot.pdf")
plot_metric('qps', "QPS (queries/sec)", "qps_plot.pdf")
