import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--logdirs',
#     nargs='+',
#     required=True,
#     help="List of log folders, each containing slo_timeouts_per_second.csv"
# )
args = parser.parse_args()

logdirs = [
    "logs_analysis_Tung/logs_8qps",
    "logs_analysis_Tung/logs_16qps",
    "logs_analysis_Tung/logs_24qps",
    "logs_analysis_Tung/logs_32qps"
]

saved_dir = 'plots'
exp_name = 'static'
y_limit = (0.0, 0.3)
os.makedirs(saved_dir, exist_ok=True)

end_point = 355
chunk_size = 5
time_frame = np.arange(0, end_point, chunk_size)*10 # we accelerate model inference by 10x

def scale_to_nearest_005(value):
    return round(np.round(value * 20) / 20, 3)

def closest_index(_list, value):
    return min(range(len(_list)), key=lambda i: abs(_list[i] - value))

plt.figure(figsize=(10, 6))
plt.ylim(y_limit)
for log_folder in logdirs:
    slo_file = os.path.join(log_folder, 'slo_timeouts_per_second.csv')
    slo_data = pd.read_csv(slo_file)[:end_point]
    
    timeout_per_second = slo_data['timeout'].groupby(slo_data.index // chunk_size).sum()
    drop_per_second = slo_data['drop'].groupby(slo_data.index // chunk_size).sum()
    failed_per_second = timeout_per_second + drop_per_second
    total_per_second = slo_data['total'].groupby(slo_data.index // chunk_size).sum().replace(0, 1)
    slo_rate = failed_per_second / total_per_second
    
    smoothed = np.convolve(slo_rate, np.ones(chunk_size) / chunk_size, mode='valid')
    mean_slo = np.mean(smoothed)
    print(f"{log_folder} Mean SLO violation rate: {mean_slo:.4f}")
    
    plt.plot(
        time_frame[:len(smoothed)],
        smoothed,
        label=os.path.basename(log_folder)
    )

plt.title('SLO violation ratio')
plt.xlabel('Time (s)')
plt.ylabel('Violation ratio')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(saved_dir, f'slo_violation_ratio_{exp_name}.png'))
plt.show()
