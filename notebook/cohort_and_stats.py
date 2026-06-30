"""
电商用户行为分析 — 同期群留存分析 + 统计检验
==============================================
在原有 notebook 基础上补充：
  1. 同期群首购留存矩阵 (Cohort Retention Matrix)
  2. 统计检验：相关性、差异显著性、集中度
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway, chi2_contingency
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── 全局绘图设置 ─────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

OUTPUT_IMAGES = "../images"
REPORT_OUTPUT = "../report/report.md"

# ====================================================================
# 1. 数据加载与预处理（沿用 notebook 逻辑 + 修复）
# ====================================================================
print("=" * 60)
print("1. 数据加载…")
print("=" * 60)

data = pd.read_csv("../data/ecommerce_data.csv")
data["event_time"] = pd.to_datetime(data["event_time"], utc=True)

# 剔除时间异常的 1970 年记录
data = data[data["event_time"].dt.year >= 2019].copy()
data["category_code"] = data["category_code"].fillna("missing")
data["brand"] = data["brand"].fillna("missing")
data["year"] = data["event_time"].dt.year
data["month"] = data["event_time"].dt.month
data["day"] = data["event_time"].dt.day
data["hour"] = data["event_time"].dt.hour
data["date"] = data["event_time"].dt.date  # 用于同期群
data["cohort_month"] = data["event_time"].dt.to_period("M")

print(f"  有效行数: {len(data):,}")
print(f"  用户数:   {data['user_id'].nunique():,}")
print(f"  时间范围: {data['event_time'].min()} → {data['event_time'].max()}")

# ====================================================================
# 2. 同期群首购留存分析
# ====================================================================
print("\n" + "=" * 60)
print("2. 同期群留存分析…")
print("=" * 60)

# 2a. 每个用户的首购月份
user_first_purchase = data.groupby("user_id")["cohort_month"].min().reset_index()
user_first_purchase.columns = ["user_id", "first_cohort"]

# 2b. merge 回原表
data_cohort = data.merge(user_first_purchase, on="user_id")
data_cohort["cohort_index"] = (
    data_cohort["cohort_month"].astype(int) - data_cohort["first_cohort"].astype(int)
)

# 2c. 构建留存矩阵
cohort_counts = data_cohort.groupby(["first_cohort", "cohort_index"])["user_id"].nunique().unstack()
cohort_sizes = cohort_counts.iloc[:, 0]  # 第 0 期 = 首购用户数
retention = cohort_counts.divide(cohort_sizes, axis=0)

print(f"\n  同期群数量: {len(cohort_sizes)}")
print(f"  首月用户数范围: {cohort_sizes.min():,} ~ {cohort_sizes.max():,}")
print(f"\n  留存率矩阵 (前 5 同期群 × 前 6 期):")
print(retention.iloc[:5, :6].to_string(float_format=lambda x: f"{x:.1%}"))

# 2d. 留存率热力图
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 左图: 留存率 (%) 热力图
ax1 = axes[0]
sns.heatmap(
    retention * 100,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    ax=ax1,
    vmin=0,
    vmax=100,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "留存率 (%)"},
)
ax1.set_title("同期群首购留存率 (%)", fontsize=16, pad=12)
ax1.set_xlabel("距离首次购买的月份", fontsize=13)
ax1.set_ylabel("首次购买月份", fontsize=13)

# 右图: 留存人数
ax2 = axes[1]
sns.heatmap(
    cohort_counts,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    ax=ax2,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "活跃用户数"},
)
ax2.set_title("同期群各月活跃用户数", fontsize=16, pad=12)
ax2.set_xlabel("距离首次购买的月份", fontsize=13)
ax2.set_ylabel("首次购买月份", fontsize=13)

plt.tight_layout()
plt.savefig(f"{OUTPUT_IMAGES}/同期群留存热力图.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 同期群留存热力图已保存")

# 2e. 留存衰减曲线
fig, ax = plt.subplots(figsize=(11, 7))

for cohort in retention.index[:7]:  # 选前 7 个同期群画线
    ax.plot(
        retention.columns[:8],
        retention.loc[cohort].values[:8] * 100,
        marker="o",
        linewidth=2,
        markersize=6,
        label=str(cohort),
    )

# 均值线
avg_retention = retention.mean(axis=0) * 100
ax.plot(
    retention.columns[:8],
    avg_retention.values[:8],
    marker="D",
    linewidth=3,
    markersize=8,
    color="black",
    linestyle="--",
    label="所有同期群均值",
)

ax.set_title("同期群留存衰减曲线", fontsize=18)
ax.set_xlabel("距离首次购买的月份", fontsize=14)
ax.set_ylabel("留存率 (%)", fontsize=14)
ax.legend(fontsize=10, loc="upper right")
ax.set_xticks(range(0, 8))
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(f"{OUTPUT_IMAGES}/同期群留存衰减曲线.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 同期群留存衰减曲线已保存")

# 2f. 关键留存指标
avg_retention_series = retention.mean(axis=0)
print(f"\n  平均次日留存率 (month+1):  {avg_retention_series.iloc[1]:.1%}")
print(f"  平均次月留存率 (month+2):  {avg_retention_series.iloc[2]:.1%}")
print(f"  平均 3 月留存率 (month+3):  {avg_retention_series.iloc[3]:.1%}")
if len(avg_retention_series) > 6:
    print(f"  平均 6 月留存率 (month+6):  {avg_retention_series.iloc[6]:.1%}")

# ====================================================================
# 3. 统计检验
# ====================================================================
print("\n" + "=" * 60)
print("3. 统计检验…")
print("=" * 60)

# 准备：用户级别的消费数据
user_agg = data.groupby("user_id").agg(
    order_count=("order_id", "nunique"),
    total_amount=("price", "sum"),
    avg_price=("price", "mean"),
    first_date=("event_time", "min"),
    last_date=("event_time", "max"),
    sex=("sex", "first"),
    age=("age", "first"),
    local=("local", "first"),
).reset_index()

# 年龄段分组
bins = [10, 20, 30, 40, 50]
labels = ["10-20", "20-30", "30-40", "40-50"]
user_agg["age_group"] = pd.cut(user_agg["age"], bins=bins, labels=labels)

# 只保留有消费的用户
user_agg = user_agg[user_agg["total_amount"] > 0]

# ── 3a. 消费次数与消费金额的相关性 ──
print("\n  3a. 消费次数 vs 消费金额")
r_pearson, p_pearson = pearsonr(user_agg["order_count"], user_agg["total_amount"])
r_spearman, p_spearman = spearmanr(user_agg["order_count"], user_agg["total_amount"])
print(f"    Pearson  r = {r_pearson:.4f},  p = {p_pearson:.2e}")
print(f"    Spearman ρ = {r_spearman:.4f},  p = {p_spearman:.2e}")

# 相关性散点图 + 回归线
fig, ax = plt.subplots(figsize=(11, 7))

# log-log 散点图减少长尾扭曲
x = user_agg["order_count"]
y = user_agg["total_amount"]

ax.scatter(x, y, alpha=0.3, s=8, edgecolors="none")
ax.set_xlabel("消费次数 (去重订单数)", fontsize=14)
ax.set_ylabel("消费金额 (元)", fontsize=14)
ax.set_title(
    f"消费次数与消费金额的关系\n"
    f"Pearson r = {r_pearson:.3f} (p < 0.001)   Spearman ρ = {r_spearman:.3f} (p < 0.001)",
    fontsize=15,
)

# 添加低通平滑趋势线
from numpy.polynomial.polynomial import Polynomial
z = np.polyfit(x, y, 1)
p_line = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 200)
ax.plot(x_line, p_line(x_line), "r-", linewidth=2.5, label="线性拟合")
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(f"{OUTPUT_IMAGES}/消费次数与金额相关性.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 消费次数与金额相关性图已保存")

# ── 3b. 性别与消费金额的差异 ──
print("\n  3b. 性别 vs 消费金额")
male = user_agg[user_agg["sex"] == "男"]["total_amount"]
female = user_agg[user_agg["sex"] == "女"]["total_amount"]

# 先做方差齐性检验
levene_stat, levene_p = stats.levene(male, female)
print(f"    Levene 方差齐性检验: stat={levene_stat:.2f}, p={levene_p:.2e}")

# 根据方差是否齐性选择 Welch t-test 或 Student t-test
if levene_p < 0.05:
    t_stat, t_p = ttest_ind(male, female, equal_var=False)
    t_method = "Welch's t-test (不等方差)"
else:
    t_stat, t_p = ttest_ind(male, female, equal_var=True)
    t_method = "Student's t-test (等方差)"

print(f"    {t_method}: t={t_stat:.4f}, p={t_p:.2e}")
print(f"    男 平均消费: ¥{male.mean():.2f} (±{male.std():.2f})")
print(f"    女 平均消费: ¥{female.mean():.2f} (±{female.std():.2f})")
print(f"    均值差 (男-女): ¥{male.mean() - female.mean():.2f}")

# Cohen's d 效应量
pooled_std = np.sqrt((male.std() ** 2 + female.std() ** 2) / 2)
cohens_d = (male.mean() - female.mean()) / pooled_std
print(f"    Cohen's d (效应量): {cohens_d:.4f}")

# ── 3c. 性别与消费次数的差异 ──
print("\n  3c. 性别 vs 消费次数")
male_orders = user_agg[user_agg["sex"] == "男"]["order_count"]
female_orders = user_agg[user_agg["sex"] == "女"]["order_count"]
t_stat_f, t_p_f = ttest_ind(male_orders, female_orders, equal_var=False)
print(f"    Welch's t-test: t={t_stat_f:.4f}, p={t_p_f:.2e}")
print(f"    男 平均消费次数: {male_orders.mean():.2f}")
print(f"    女 平均消费次数: {female_orders.mean():.2f}")

# 可视化：性别对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左: 消费金额 箱线图
ax1 = axes[0]
bp1 = ax1.boxplot(
    [male.values, female.values],
    labels=["男", "女"],
    patch_artist=True,
    widths=0.4,
    showfliers=False,  # 隐藏离群点使图可读
)
bp1["boxes"][0].set_facecolor("#66B2FF")
bp1["boxes"][1].set_facecolor("#FF9999")
ax1.set_title("不同性别用户消费金额分布", fontsize=14)
ax1.set_ylabel("消费金额 (元)", fontsize=12)
# 添加均值标注
ax1.annotate(
    f"均值 ¥{male.mean():.0f}",
    xy=(1, male.mean()),
    xytext=(1.3, male.mean() * 1.2),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=11,
)
ax1.annotate(
    f"均值 ¥{female.mean():.0f}",
    xy=(2, female.mean()),
    xytext=(2.3, female.mean() * 1.2),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=11,
)

# 右: 消费次数 箱线图
ax2 = axes[1]
bp2 = ax2.boxplot(
    [male_orders.values, female_orders.values],
    labels=["男", "女"],
    patch_artist=True,
    widths=0.4,
    showfliers=False,
)
bp2["boxes"][0].set_facecolor("#66B2FF")
bp2["boxes"][1].set_facecolor("#FF9999")
ax2.set_title("不同性别用户消费次数分布", fontsize=14)
ax2.set_ylabel("消费次数", fontsize=12)

plt.tight_layout()
plt.savefig(f"{OUTPUT_IMAGES}/性别消费差异箱线图.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 性别消费差异箱线图已保存")

# ── 3d. 年龄段与消费金额的差异 (ANOVA) ──
print("\n  3d. 年龄段 vs 消费金额")

age_groups_data = [
    user_agg[user_agg["age_group"] == label]["total_amount"].values
    for label in labels
]
f_stat_age, p_age = f_oneway(*age_groups_data)
print(f"    One-way ANOVA: F={f_stat_age:.4f}, p={p_age:.2e}")

# 事后检验: Tukey HSD (如果 ANOVA 显著)
if p_age < 0.05:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    tukey = pairwise_tukeyhsd(user_agg["total_amount"], user_agg["age_group"], alpha=0.05)
    print(f"    Tukey HSD 事后检验:")
    print(tukey)

# 计算 eta-squared 效应量
grand_mean = user_agg["total_amount"].mean()
ss_between = sum(
    len(group) * (group.mean() - grand_mean) ** 2
    for group in age_groups_data
    if len(group) > 0
)
ss_total = ((user_agg["total_amount"] - grand_mean) ** 2).sum()
eta_sq = ss_between / ss_total
print(f"    η² (效应量): {eta_sq:.4f}")

# 分组统计
print(f"\n    各年龄段消费金额统计:")
for label in labels:
    subset = user_agg[user_agg["age_group"] == label]
    print(f"      {label}: n={len(subset):,}, 均值=¥{subset['total_amount'].mean():.2f}, "
          f"中位数=¥{subset['total_amount'].median():.2f}, "
          f"人均消费次数={subset['order_count'].mean():.1f}")

# ── 3e. 省份与消费金额的差异 (ANOVA) ──
print("\n  3e. 省份 vs 消费金额")

province_groups = [
    user_agg[user_agg["local"] == loc]["total_amount"].values
    for loc in sorted(user_agg["local"].unique())
]
f_stat_prov, p_prov = f_oneway(*province_groups)
print(f"    One-way ANOVA: F={f_stat_prov:.4f}, p={p_prov:.2e}")

# 计算 eta-squared
ss_between_prov = sum(
    len(group) * (group.mean() - grand_mean) ** 2
    for group in province_groups
    if len(group) > 0
)
eta_sq_prov = ss_between_prov / ss_total
print(f"    η² (效应量): {eta_sq_prov:.4f}")

# 省份统计汇总
province_stats = user_agg.groupby("local").agg(
    user_count=("user_id", "count"),
    avg_amount=("total_amount", "mean"),
    median_amount=("total_amount", "median"),
    total_amount=("total_amount", "sum"),
    avg_orders=("order_count", "mean"),
).sort_values("avg_amount", ascending=False)

print(f"\n    各省份用户消费统计 (Top 5 按人均):")
print(province_stats.head(5).to_string(formatters={"avg_amount": "¥{:.2f}".format}))

# 省份对比图
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(province_stats)))
ax.barh(province_stats.index, province_stats["avg_amount"].values, color=colors)
ax.set_title(
    f"不同省份用户人均消费金额\n"
    f"One-way ANOVA: F={f_stat_prov:.2f}, p={p_prov:.2e}, η²={eta_sq_prov:.4f}",
    fontsize=14,
)
ax.set_xlabel("人均消费金额 (元)", fontsize=13)
ax.set_ylabel("省份", fontsize=13)
for i, (v, n) in enumerate(zip(province_stats["avg_amount"].values, province_stats["user_count"].values)):
    ax.text(v + 5, i, f"¥{v:.0f}  (n={n:,})", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_IMAGES}/省份人均消费统计检验.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 省份人均消费统计检验图已保存")

# ── 3f. 小时与订单分布的拟合优度检验 ──
print("\n  3f. 各小时订单分布均匀性检验")

hour_counts = data.groupby("hour")["order_id"].nunique()
# 卡方拟合优度: 检验是否与均匀分布有显著差异
chi2_hour, p_hour = stats.chisquare(hour_counts.values)
print(f"    χ² 拟合优度检验 (均匀分布): χ²={chi2_hour:.2f}, p={p_hour:.2e}")
print(f"    订单最多的小时: {hour_counts.idxmax()} 时 ({hour_counts.max():,} 单)")
print(f"    订单最少的小时: {hour_counts.idxmin()} 时 ({hour_counts.min():,} 单)")
print(f"    极差比: {hour_counts.max() / hour_counts.min():.2f}x")

# ── 3g. 品牌集中度分析 ──
print("\n  3g. 品牌集中度 (CRn & HHI)")

brand_orders = data[data["price"] > 0].groupby("brand")["order_id"].nunique().sort_values(ascending=False)
total_brand_orders = brand_orders.sum()

for n in [3, 5, 10, 20]:
    cr_n = brand_orders.head(n).sum() / total_brand_orders
    print(f"    CR{n} (Top {n} 品牌集中度): {cr_n:.1%}")

# HHI 指数 (Herfindahl-Hirschman Index)
hhi = ((brand_orders / total_brand_orders) ** 2).sum()
print(f"    HHI (市场集中度指数): {hhi:.4f}  ({'高集中度' if hhi > 0.25 else '中集中度' if hhi > 0.15 else '低集中度'})")

# 品牌 Lorenz 曲线 (累积份额)
brand_cumsum = brand_orders.cumsum() / total_brand_orders
brand_pct = np.arange(1, len(brand_cumsum) + 1) / len(brand_cumsum)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(brand_pct * 100, brand_cumsum.values * 100, "b-", linewidth=2)
ax.plot([0, 100], [0, 100], "r--", linewidth=1.5, label="完全均等线")
ax.fill_between(brand_pct * 100, brand_pct * 100, brand_cumsum.values * 100, alpha=0.3, color="blue")
ax.set_title(f"品牌订单份额 Lorenz 曲线\nCR5={brand_orders.head(5).sum()/total_brand_orders:.1%},  HHI={hhi:.4f}", fontsize=14)
ax.set_xlabel("品牌累计百分比 (%)", fontsize=13)
ax.set_ylabel("订单累计份额 (%)", fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f"{OUTPUT_IMAGES}/品牌集中度洛伦兹曲线.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 品牌集中度 Lorenz 曲线已保存")

# ── 3h. 品类集中度 ──
print("\n  3h. 品类集中度")
cat_orders = data[data["price"] > 0].groupby("category_code")["order_id"].nunique().sort_values(ascending=False)
total_cat_orders = cat_orders.sum()
for n in [3, 5, 10]:
    cr_n_cat = cat_orders.head(n).sum() / total_cat_orders
    print(f"    CR{n}: {cr_n_cat:.1%}")
hhi_cat = ((cat_orders / total_cat_orders) ** 2).sum()
print(f"    HHI: {hhi_cat:.4f}")

# ====================================================================
# 4. 汇总关键统计指标
# ====================================================================
print("\n" + "=" * 60)
print("4. 关键指标汇总")
print("=" * 60)

total_users = user_agg["user_id"].nunique()
active_users = user_agg[user_agg["order_count"] > 1]["user_id"].nunique()
repeat_rate = active_users / total_users

print(f"  总用户数: {total_users:,}")
print(f"  复购用户数 (>1单): {active_users:,} ({repeat_rate:.1%})")
print(f"  人均消费次数: {user_agg['order_count'].mean():.2f}")
print(f"  人均消费金额: ¥{user_agg['total_amount'].mean():.2f}")
print(f"  客单价均值: ¥{data[data['price']>0]['price'].mean():.2f}")

# 同期群核心指标
print(f"  平均首月存留率 (month+1): {avg_retention_series.iloc[1]:.1%}")
print(f"  平均 3 月存留率:           {avg_retention_series.iloc[3]:.1%}")
print(f"  平均 6 月存留率:           {avg_retention_series.iloc[6]:.1%}" if len(avg_retention_series) > 6 else "")

print("\n✅ 分析完成。所有图片已保存至 images/ 目录。")
