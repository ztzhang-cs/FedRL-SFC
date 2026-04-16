import numpy as np
import os

def print_table_data(npz_path, tail=30):
    # 检查文件是否存在
    if not os.path.exists(npz_path):
        print(f"❌ 找不到文件: {npz_path} (请确认路径是否正确)")
        return
        
    data = np.load(npz_path)
    methods = [('Heuristic', 'h'), ('Local RL', 'l'), ('FedRL', 'f'), ('FedGreedy', 'g')]
    
    print(f"\n===== Data for {npz_path} =====")
    for name, suffix in methods:
        # 提取最后 tail 个 episode 的数据
        acc = data[f'acc_{suffix}'][-tail:]
        delay = data[f'delay_{suffix}'][-tail:]
        util = data[f'avg_util_{suffix}'][-tail:]
        block = data[f'block_{suffix}'][-tail:]
        
        print(f"{name}:")
        # 使用 \\pm 来避免 Python 的转义报错，同时直接生成可以直接粘贴到 LaTeX 的格式
        print(f"  Acceptance: {np.mean(acc):.4f} \\pm {np.std(acc):.4f}")
        print(f"  E2E Delay:  {np.mean(delay):.2f} \\pm {np.std(delay):.2f}")
        print(f"  Avg Util:   {np.mean(util):.4f} \\pm {np.std(util):.4f}")
        print(f"  Blocking:   {np.mean(block):.4f} \\pm {np.std(block):.4f}")

# 加上了 'results/' 文件夹前缀
print_table_data('results/results_random.npz')
print_table_data('results/results_toy.npz')
