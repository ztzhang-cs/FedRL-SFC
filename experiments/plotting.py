# experiments/plotting.py
import numpy as np
import matplotlib.pyplot as plt


def smooth_curve(arr, window: int = 7):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    smooth = np.convolve(arr, kernel, mode="valid")
    return smooth


def cumulative_average(arr):
    arr = np.asarray(arr, dtype=float)
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


def _safe_arr(x):
    if x is None:
        return None
    a = np.asarray(x, dtype=float).reshape(-1)
    return a


def _legend_outside_right():
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def _tight_layout_with_legend_space():
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))


def _align_len(*arrs):
    """裁剪到同一长度（以第一个非 None 的长度为基准）"""
    base = None
    for a in arrs:
        if a is not None:
            base = len(a)
            break
    if base is None:
        return arrs
    out = []
    for a in arrs:
        out.append(None if a is None else a[:base])
    return tuple(out)


# ===================== 曲线图 =====================

def plot_convergence_delays(
    h, l, f, g=None,
    d=None,                 # optional DFSC eval delay
    p=None,                 # optional FLPredict eval delay
    save_path: str = "results/convergence_delay.png",
    l_train=None,
    f_train=None,
    g_train=None,
    show_train: bool = True,
):
    """
    支持 4 条主曲线：Heuristic / Local / FedRL / FedGreedy（g 可选）
    仍兼容旧的 d/p（可选上界方法）。
    """
    h = _safe_arr(h); l = _safe_arr(l); f = _safe_arr(f)
    g = _safe_arr(g)
    d = _safe_arr(d); p = _safe_arr(p)

    h_s = smooth_curve(cumulative_average(h), window=21)
    l_s = smooth_curve(cumulative_average(l), window=21)
    f_s = smooth_curve(cumulative_average(f), window=21)
    g_s = smooth_curve(cumulative_average(g), window=21) if g is not None else None

    d_s = smooth_curve(cumulative_average(d), window=21) if d is not None else None
    p_s = smooth_curve(cumulative_average(p), window=21) if p is not None else None

    h_s, l_s, f_s, g_s, d_s, p_s = _align_len(h_s, l_s, f_s, g_s, d_s, p_s)
    n = len(h_s)
    episodes = range(1, n + 1)

    plt.figure(figsize=(9.5, 5))
    plt.plot(episodes, h_s, label="Heuristic", linewidth=2)
    plt.plot(episodes, l_s, label="Local RL", linewidth=2)
    plt.plot(episodes, f_s, label="FedRL", linewidth=2)
    if g_s is not None:
        plt.plot(episodes, g_s, label="FedGreedy", linewidth=2)
    if d_s is not None:
        plt.plot(episodes, d_s, label="DFSC (eval)", linewidth=2)
    if p_s is not None:
        plt.plot(episodes, p_s, label="FL-Predict (eval)", linewidth=2)

    if show_train:
        if l_train is not None:
            l_train = _safe_arr(l_train)
            l_train_s = smooth_curve(cumulative_average(l_train), window=21)
            m = min(len(l_train_s), n)
            plt.plot(range(1, m + 1), l_train_s[:m], linestyle="--", alpha=0.35,
                     linewidth=1.8, label="Local RL (train)")
        if f_train is not None:
            f_train = _safe_arr(f_train)
            f_train_s = smooth_curve(cumulative_average(f_train), window=21)
            m = min(len(f_train_s), n)
            plt.plot(range(1, m + 1), f_train_s[:m], linestyle="--", alpha=0.35,
                     linewidth=1.8, label="FedRL (train)")
        if g_train is not None:
            g_train = _safe_arr(g_train)
            g_train_s = smooth_curve(cumulative_average(g_train), window=21)
            m = min(len(g_train_s), n)
            plt.plot(range(1, m + 1), g_train_s[:m], linestyle="--", alpha=0.35,
                     linewidth=1.8, label="FedGreedy (train)")

    plt.xlabel("Episode")
    plt.ylabel("End-to-end delay (cumulative avg, smoothed)")
    _legend_outside_right()
    plt.grid(alpha=0.3, linestyle="--")
    _tight_layout_with_legend_space()
    plt.savefig(save_path, dpi=300)
    print(f"Delay convergence figure saved to {save_path}")


def plot_acceptance(
    acc_h, acc_l, acc_f, acc_g=None,
    acc_d=None,
    acc_p=None,
    save_path: str = "results/acceptance.png"
):
    acc_h = _safe_arr(acc_h); acc_l = _safe_arr(acc_l); acc_f = _safe_arr(acc_f)
    acc_g = _safe_arr(acc_g)
    acc_d = _safe_arr(acc_d); acc_p = _safe_arr(acc_p)

    h_s = smooth_curve(acc_h, window=21)
    l_s = smooth_curve(acc_l, window=21)
    f_s = smooth_curve(acc_f, window=21)
    g_s = smooth_curve(acc_g, window=21) if acc_g is not None else None

    d_s = smooth_curve(acc_d, window=21) if acc_d is not None else None
    p_s = smooth_curve(acc_p, window=21) if acc_p is not None else None

    h_s, l_s, f_s, g_s, d_s, p_s = _align_len(h_s, l_s, f_s, g_s, d_s, p_s)
    n = len(h_s)
    episodes = range(1, n + 1)

    plt.figure(figsize=(9.5, 5))
    plt.plot(episodes, h_s, label="Heuristic", linewidth=2)
    plt.plot(episodes, l_s, label="Local RL", linewidth=2)
    plt.plot(episodes, f_s, label="FedRL", linewidth=2)
    if g_s is not None:
        plt.plot(episodes, g_s, label="FedGreedy", linewidth=2)
    if d_s is not None:
        plt.plot(episodes, d_s, label="DFSC", linewidth=2)
    if p_s is not None:
        plt.plot(episodes, p_s, label="FL-Predict", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Acceptance ratio (smoothed)")

    all_list = [h_s, l_s, f_s]
    if g_s is not None: all_list.append(g_s)
    if d_s is not None: all_list.append(d_s)
    if p_s is not None: all_list.append(p_s)
    all_vals = np.concatenate(all_list)
    ymin = max(0.0, float(all_vals.min()) - 0.05)
    ymax = min(1.0, float(all_vals.max()) + 0.05)
    if ymin == ymax:
        ymax = ymin + 0.1
    plt.ylim(ymin, ymax)

    _legend_outside_right()
    plt.grid(alpha=0.3, linestyle="--")
    _tight_layout_with_legend_space()
    plt.savefig(save_path, dpi=300)
    print(f"Acceptance figure saved to {save_path}")


def plot_blocking(
    block_h, block_l, block_f, block_g=None,
    block_d=None,
    block_p=None,
    save_path: str = "results/blocking.png"
):
    block_h = _safe_arr(block_h); block_l = _safe_arr(block_l); block_f = _safe_arr(block_f)
    block_g = _safe_arr(block_g)
    block_d = _safe_arr(block_d); block_p = _safe_arr(block_p)

    h_s = smooth_curve(block_h, window=21)
    l_s = smooth_curve(block_l, window=21)
    f_s = smooth_curve(block_f, window=21)
    g_s = smooth_curve(block_g, window=21) if block_g is not None else None

    d_s = smooth_curve(block_d, window=21) if block_d is not None else None
    p_s = smooth_curve(block_p, window=21) if block_p is not None else None

    h_s, l_s, f_s, g_s, d_s, p_s = _align_len(h_s, l_s, f_s, g_s, d_s, p_s)
    n = len(h_s)
    episodes = range(1, n + 1)

    plt.figure(figsize=(9.5, 5))
    plt.plot(episodes, h_s, label="Heuristic", linewidth=2)
    plt.plot(episodes, l_s, label="Local RL", linewidth=2)
    plt.plot(episodes, f_s, label="FedRL", linewidth=2)
    if g_s is not None:
        plt.plot(episodes, g_s, label="FedGreedy", linewidth=2)
    if d_s is not None:
        plt.plot(episodes, d_s, label="DFSC", linewidth=2)
    if p_s is not None:
        plt.plot(episodes, p_s, label="FL-Predict", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Blocking ratio (smoothed)")
    plt.ylim(0.0, 1.0)

    _legend_outside_right()
    plt.grid(alpha=0.3, linestyle="--")
    _tight_layout_with_legend_space()
    plt.savefig(save_path, dpi=300)
    print(f"Blocking figure saved to {save_path}")


def plot_utilization(
    max_h, max_l, max_f, max_g=None,
    avg_h=None, avg_l=None, avg_f=None, avg_g=None,
    avg_d=None,
    avg_p=None,
    save_path_max: str = "results/max_util.png",
    save_path_avg: str = "results/avg_util.png",
):
    """
    你原函数只画 avg-util（max-util 参数没用到），这里补全：
    - 画 max-util 曲线：Heuristic/Local/Fed/FedGreedy
    - 画 avg-util 曲线：Heuristic/Local/Fed/FedGreedy
    仍兼容旧的 avg_d/avg_p。
    """
    max_h = _safe_arr(max_h); max_l = _safe_arr(max_l); max_f = _safe_arr(max_f)
    max_g = _safe_arr(max_g)

    avg_h = _safe_arr(avg_h); avg_l = _safe_arr(avg_l); avg_f = _safe_arr(avg_f)
    avg_g = _safe_arr(avg_g)

    avg_d = _safe_arr(avg_d); avg_p = _safe_arr(avg_p)

    # ---------- max util ----------
    max_h_s = smooth_curve(max_h, window=21)
    max_l_s = smooth_curve(max_l, window=21)
    max_f_s = smooth_curve(max_f, window=21)
    max_g_s = smooth_curve(max_g, window=21) if max_g is not None else None

    max_h_s, max_l_s, max_f_s, max_g_s = _align_len(max_h_s, max_l_s, max_f_s, max_g_s)
    n = len(max_h_s)
    episodes = range(1, n + 1)

    plt.figure(figsize=(9.5, 5))
    plt.plot(episodes, max_h_s, label="Heuristic", linewidth=2)
    plt.plot(episodes, max_l_s, label="Local RL", linewidth=2)
    plt.plot(episodes, max_f_s, label="FedRL", linewidth=2)
    if max_g_s is not None:
        plt.plot(episodes, max_g_s, label="FedGreedy", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Max node utilization (smoothed)")
    plt.ylim(0.0, 1.05)

    _legend_outside_right()
    plt.grid(alpha=0.3, linestyle="--")
    _tight_layout_with_legend_space()
    plt.savefig(save_path_max, dpi=300)
    print(f"Max-util figure saved to {save_path_max}")

    # ---------- avg util ----------
    avg_h_s = smooth_curve(avg_h, window=21)
    avg_l_s = smooth_curve(avg_l, window=21)
    avg_f_s = smooth_curve(avg_f, window=21)
    avg_g_s = smooth_curve(avg_g, window=21) if avg_g is not None else None

    d_s = smooth_curve(avg_d, window=21) if avg_d is not None else None
    p_s = smooth_curve(avg_p, window=21) if avg_p is not None else None

    avg_h_s, avg_l_s, avg_f_s, avg_g_s, d_s, p_s = _align_len(avg_h_s, avg_l_s, avg_f_s, avg_g_s, d_s, p_s)
    n = len(avg_h_s)
    episodes = range(1, n + 1)

    plt.figure(figsize=(9.5, 5))
    plt.plot(episodes, avg_h_s, label="Heuristic", linewidth=2)
    plt.plot(episodes, avg_l_s, label="Local RL", linewidth=2)
    plt.plot(episodes, avg_f_s, label="FedRL", linewidth=2)
    if avg_g_s is not None:
        plt.plot(episodes, avg_g_s, label="FedGreedy", linewidth=2)
    if d_s is not None:
        plt.plot(episodes, d_s, label="DFSC", linewidth=2)
    if p_s is not None:
        plt.plot(episodes, p_s, label="FL-Predict", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Average node utilization (smoothed)")
    plt.ylim(0.0, 1.05)

    _legend_outside_right()
    plt.grid(alpha=0.3, linestyle="--")
    _tight_layout_with_legend_space()
    plt.savefig(save_path_avg, dpi=300)
    print(f"Avg-util figure saved to {save_path_avg}")


# ===================== 热力图 =====================

def plot_attention_heatmap(
    attn_mat,
    x_labels=None,
    y_labels=None,
    title="FedRL attention heatmap",
    save_path="results/attn_heatmap.png",
):
    M = np.asarray(attn_mat, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"attn_mat must be 2D, got shape={M.shape}")

    # 处理 NaN / inf
    finite = np.isfinite(M)
    if finite.any():
        fill = float(np.nanmean(M[finite]))
    else:
        fill = 0.0
    M = np.nan_to_num(M, nan=fill, posinf=fill, neginf=fill)
    T, D = M.shape

    # ==========================================
    # ✅ 核心修复：将矩阵的列和 x 轴标签同步强制排序！
    # ==========================================
    if x_labels is not None and len(x_labels) == D:
        # 提取出数字 (比如从 'D0' 或 0 中提取出整数)
        numeric_ids = []
        for label in x_labels:
            if isinstance(label, str):
                numeric_ids.append(int(''.join(filter(str.isdigit, label))))
            else:
                numeric_ids.append(int(label))
        
        # 找到正确的排序索引（从小到大：0, 1, 2, 3, 4, 5）
        sort_indices = np.argsort(numeric_ids)
        
        # 1. 对矩阵的列进行重新排列，使其数据对齐
        M = M[:, sort_indices]
        
        # 2. 生成规范的论文标签 (D1, D2, D3...)，这里 +1 是为了契合你正文从 1 开始的称呼
        sorted_ids = np.array(numeric_ids)[sort_indices]
        x_labels = [f"D{i}" for i in sorted_ids]
    else:
        # 兜底：如果没有传入标签，默认它是顺序的
        x_labels = [f"D{i}" for i in range(D)]
    # ==========================================

    plt.figure(figsize=(12, 6))
    im = plt.imshow(M, aspect="auto", interpolation="nearest")

    plt.title(title)
    plt.xlabel("Target (domains)")
    plt.ylabel("Episode")

    # 强制打印所有的 X 轴刻度，绝不跳号
    plt.xticks(np.arange(D), x_labels, rotation=45, ha="right")

    if y_labels is not None and len(y_labels) == T:
        plt.yticks(np.arange(T), y_labels)

    plt.colorbar(im, fraction=0.025, pad=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Attention heatmap saved to {save_path}")



# ===================== 柱状图（最后 tail 个 episode 的均值） =====================

def _tail_mean(arr, tail: int):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return float("nan")
    t = min(int(tail), int(a.size))
    return float(np.mean(a[-t:]))


def plot_delay_bar(delay_dict: dict, tail: int = 30, save_path: str = "results/delay_bar.png"):
    """
    delay_dict: {"FedRL": [..], "Local": [..], ...}
    """
    names = list(delay_dict.keys())
    vals = [_tail_mean(delay_dict[k], tail) for k in names]

    plt.figure(figsize=(9.5, 5))
    plt.bar(names, vals)
    plt.title("Delay comparison (lower is better)")
    plt.ylabel(f"Avg delay (last {tail} eps)")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Delay bar saved to {save_path}")


def plot_acceptance_bar(acc_dict: dict, tail: int = 30, save_path: str = "results/acceptance_bar.png"):
    """
    acc_dict: {"FedRL": [..], "Local": [..], ...}
    """
    names = list(acc_dict.keys())
    vals = [_tail_mean(acc_dict[k], tail) for k in names]

    plt.figure(figsize=(9.5, 5))
    plt.bar(names, vals)
    plt.title("Acceptance comparison (higher is better)")
    plt.ylabel(f"Acceptance (last {tail} eps)")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Acceptance bar saved to {save_path}")
