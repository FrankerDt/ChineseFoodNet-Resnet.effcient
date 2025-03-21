import re
import pandas as pd
import matplotlib.pyplot as plt

# 数据准备部分保持不变
with open("model_test_res/test_result.txt", "r", encoding="utf-8") as f:
    text = f.read()

pattern = re.compile(
    r"Model:\s+(resmodel-[\d\-\.]+\.pt).*?"
    r"Test Top 1 Accuracy:\s+([\d.]+)%.*?"
    r"Test Top 5 Accuracy:\s+([\d.]+)%.*?"
    r"Val Top 1 Accuracy:\s+([\d.]+)%.*?"
    r"Val Top 5 Accuracy:\s+([\d.]+)%",
    re.DOTALL
)

data = pd.DataFrame(
    pattern.findall(text),
    columns=["Model", "Test Top 1", "Test Top 5", "Val Top 1", "Val Top 5"]
)
numeric_cols = ["Test Top 1", "Test Top 5", "Val Top 1", "Val Top 5"]
data[numeric_cols] = data[numeric_cols].astype(float)
data["Main Model ID"] = data["Model"].str.extract(r"resmodel-(\d+)-\d+").astype(int)
data = data.sort_values("Main Model ID").reset_index(drop=True)

# 创建2x2子图布局
fig, axs = plt.subplots(2, 2, figsize=(20, 16))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# 分离绘图参数和字体参数
plot_style = {
    "linewidth": 2.5,
    "markersize": 8
}

font_style = {
    "title": {"fontsize": 16},
    "label": {"fontsize": 14},
    "tick": {"labelsize": 12}
}

# 图1: Test集TOP1 vs TOP5
axs[0,0].plot(data["Main Model ID"], data["Test Top 1"],
            marker='o', linestyle='-', color='steelblue',
            label='Test Top 1', **plot_style)
axs[0,0].plot(data["Main Model ID"], data["Test Top 5"],
            marker='s', linestyle='-', color='deepskyblue',
            label='Test Top 5', **plot_style)
axs[0,0].set_title('Test Set Accuracy Comparison', **font_style["title"])
axs[0,0].set_ylabel('Accuracy (%)', **font_style["label"])

# 图2: Val集TOP1 vs TOP5
axs[0,1].plot(data["Main Model ID"], data["Val Top 1"],
            marker='^', linestyle='--', color='crimson',
            label='Val Top 1', **plot_style)
axs[0,1].plot(data["Main Model ID"], data["Val Top 5"],
            marker='D', linestyle='--', color='lightcoral',
            label='Val Top 5', **plot_style)
axs[0,1].set_title('Validation Set Accuracy Comparison', **font_style["title"])

# 图3: TOP1对比（Test vs Val）
axs[1,0].plot(data["Main Model ID"], data["Test Top 1"],
            marker='o', linestyle='-', color='steelblue',
            label='Test Top 1', **plot_style)
axs[1,0].plot(data["Main Model ID"], data["Val Top 1"],
            marker='^', linestyle='--', color='crimson',
            label='Val Top 1', **plot_style)
axs[1,0].set_title('Top-1 Accuracy Comparison', **font_style["title"])
axs[1,0].set_xlabel('Main Model Generation', **font_style["label"])
axs[1,0].set_ylabel('Accuracy (%)', **font_style["label"])

# 图4: TOP5对比（Test vs Val）
axs[1,1].plot(data["Main Model ID"], data["Test Top 5"],
            marker='s', linestyle='-', color='deepskyblue',
            label='Test Top 5', **plot_style)
axs[1,1].plot(data["Main Model ID"], data["Val Top 5"],
            marker='D', linestyle='--', color='lightcoral',
            label='Val Top 5', **plot_style)
axs[1,1].set_title('Top-5 Accuracy Comparison', **font_style["title"])
axs[1,1].set_xlabel('Main Model Generation', **font_style["label"])

# 统一设置坐标轴
for ax in axs.flat:
    ax.set_ylim(60, 100)
    ax.set_xticks(data["Main Model ID"].unique()[::5])  # 每5个模型显示一个刻度
    ax.tick_params(**font_style["tick"])
    ax.grid(True, alpha=0.3)
    ax.legend(prop={'size': font_style["label"]["fontsize"]})
    ax.label_outer()

plt.tight_layout()
plt.show()
# 数据处理部分保持不变（与之前相同）
# ...

# 定义通用样式配置
plot_config = {
    "linewidth": 2.5,
    "markersize": 8,
    "dpi": 300,
    "figsize": (10, 6),
    "font_settings": {
        "title": {"fontsize": 16, "pad": 15},
        "axis": {"fontsize": 14},
        "legend": {"fontsize": 12}
    }
}


# 图表1: Test集TOP1 vs TOP5
def save_test_comparison():
    plt.figure(figsize=plot_config["figsize"])
    plt.plot(data["Main Model ID"], data["Test Top 1"],
             marker='o', linestyle='-', color='steelblue',
             linewidth=plot_config["linewidth"],
             markersize=plot_config["markersize"],
             label='Test Top 1')
    plt.plot(data["Main Model ID"], data["Test Top 5"],
             marker='s', linestyle='-', color='deepskyblue',
             linewidth=plot_config["linewidth"],
             markersize=plot_config["markersize"],
             label='Test Top 5')

    plt.title('Test Set Accuracy Comparison', **plot_config["font_settings"]["title"])
    plt.xlabel('Main Model Generation', **plot_config["font_settings"]["axis"])
    plt.ylabel('Accuracy (%)', **plot_config["font_settings"]["axis"])
    plt.legend(**plot_config["font_settings"]["legend"])
    plt.ylim(60, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("test_comparison.png", dpi=plot_config["dpi"], bbox_inches='tight')
    plt.close()


# 图表2: Val集TOP1 vs TOP5
def save_val_comparison():
    plt.figure(figsize=plot_config["figsize"])
    plt.plot(data["Main Model ID"], data["Val Top 1"],
             marker='^', linestyle='--', color='crimson',
             linewidth=plot_config["linewidth"],
             markersize=plot_config["markersize"],
             label='Val Top 1')
    plt.plot(data["Main Model ID"], data["Val Top 5"],
             marker='D', linestyle='--', color='lightcoral',
             linewidth=plot_config["linewidth"],
             markersize=plot_config["markersize"],
             label='Val Top 5')

    plt.title('Validation Set Accuracy Comparison', **plot_config["font_settings"]["title"])
    plt.xlabel('Main Model Generation', **plot_config["font_settings"]["axis"])
    plt.ylabel('Accuracy (%)', **plot_config["font_settings"]["axis"])
    plt.legend(**plot_config["font_settings"]["legend"])
    plt.ylim(60, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("val_comparison.png", dpi=plot_config["dpi"], bbox_inches='tight')
    plt.close()


# 图表3: TOP1对比（Test vs Val）
def save_top1_comparison():
    plt.figure(figsize=plot_config["figsize"])
    plt.plot(data["Main Model ID"], data["Test Top 1"],
             marker='o', linestyle='-', color='steelblue',
             linewidth=plot_config["linewidth"],
             markersize=plot_config["markersize"],
             label='Test Top 1')
    plt.plot(data["Main Model ID"], data["Val Top 1"],
             marker='^', linestyle='--', color='crimson',
             linewidth=plot_config["linewidth"],
             markersize=plot_config["markersize"],
             label='Val Top 1')

    plt.title('Top-1 Accuracy Comparison', **plot_config["font_settings"]["title"])
    plt.xlabel('Main Model Generation', **plot_config["font_settings"]["axis"])
    plt.ylabel('Accuracy (%)', **plot_config["font_settings"]["axis"])
    plt.legend(**plot_config["font_settings"]["legend"])
    plt.ylim(60, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("top1_comparison.png", dpi=plot_config["dpi"], bbox_inches='tight')
    plt.close()


# 图表4: TOP5对比（Test vs Val）
def save_top5_comparison():
    plt.figure(figsize=plot_config["figsize"])
    plt.plot(data["Main Model ID"], data["Test Top 5"],
             marker='s', linestyle='-', color='deepskyblue',
             linewidth=plot_config["linewidth"],
             markersize=plot_config["markersize"],
             label='Test Top 5')
    plt.plot(data["Main Model ID"], data["Val Top 5"],
             marker='D', linestyle='--', color='lightcoral',
             linewidth=plot_config["linewidth"],
             markersize=plot_config["markersize"],
             label='Val Top 5')

    plt.title('Top-5 Accuracy Comparison', **plot_config["font_settings"]["title"])
    plt.xlabel('Main Model Generation', **plot_config["font_settings"]["axis"])
    plt.ylabel('Accuracy (%)', **plot_config["font_settings"]["axis"])
    plt.legend(**plot_config["font_settings"]["legend"])
    plt.ylim(60, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("top5_comparison.png", dpi=plot_config["dpi"], bbox_inches='tight')
    plt.close()


# 执行保存函数
save_test_comparison()
save_val_comparison()
save_top1_comparison()
save_top5_comparison()

