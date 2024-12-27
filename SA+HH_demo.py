import os
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
from lifelines import CoxPHFitter
import matplotlib
import statsmodels.api as sm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 建议：若没有 result 文件夹，先创建
os.makedirs('result', exist_ok=True)

#####################
#  数据清洗 & 整合  #
#####################

def add_dates_and_redemptions(base_df, campaign_desc_path: str, coupon_redempt_path: str):
    """
    Adds issuance, expiration, and redemption dates to the base coupon table.
    Excludes TypeA campaigns from redemption data.
    """
    typeA_campaigns = [8, 13, 18, 26, 30]

    campaign_desc = pd.read_csv(campaign_desc_path)
    with_dates = base_df.merge(
        campaign_desc[['CAMPAIGN', 'START_DAY', 'END_DAY']],
        on='CAMPAIGN',
        how='left'
    ).rename(columns={'START_DAY': 'day_issued', 'END_DAY': 'day_expired'})

    coupon_redempt = pd.read_csv(coupon_redempt_path)
    coupon_redempt_filtered = coupon_redempt[~coupon_redempt['CAMPAIGN'].isin(typeA_campaigns)]

    coupon_redempt_filtered = coupon_redempt_filtered.sort_values('DAY').drop_duplicates(
        subset=['household_key', 'CAMPAIGN', 'COUPON_UPC'],
        keep='first'
    )

    final_df = with_dates.merge(
        coupon_redempt_filtered[['household_key', 'CAMPAIGN', 'COUPON_UPC', 'DAY']],
        on=['household_key', 'CAMPAIGN', 'COUPON_UPC'],
        how='left'
    ).rename(columns={'DAY': 'day_redeemed'})

    return final_df


def create_complete_dataset():
    """
    Creates the complete coupon dataset, excluding TypeA campaigns.
    """
    typeA_campaigns = [8, 13, 18, 26, 30]

    base_df = create_coupon_base_table(
        'data/coupon.csv',
        'data/campaign_desc.csv',
        'data/campaign_table.csv'
    )

    base_df_filtered = base_df[~base_df['CAMPAIGN'].isin(typeA_campaigns)]

    final_df = add_dates_and_redemptions(
        base_df_filtered,
        'data/campaign_desc.csv',
        'data/coupon_redempt.csv'
    )

    return final_df


def check_cox_assumptions(df):
    """
    全面检查 Cox 模型的统计假设。这个函数进行三个主要检验：
    1. Schoenfeld 残差检验
    2. 时间交互项检验（使用对数、线性和平方根三种时间变换）
    3. 多重共线性检验（通过 VIF 和相关系数）

    Args:
        df: 包含生存数据的 DataFrame，必须包含用于 Cox 模型的所有必要变量

    Returns:
        cph: 拟合好的 Cox 比例风险模型
    """
    print("开始检验 Cox 模型的统计假设...")

    # 1. 准备数据并拟合初始 Cox 模型
    cox_data = prepare_cox_data(df)
    cox_data_no_na = cox_data.dropna(axis=0)

    # 使用弱惩罚项来提高数值稳定性
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        cox_data_no_na,
        duration_col='duration',
        event_col='event',
        show_progress=True
    )

    # 2. 比例风险假设检验函数
    def check_proportional_hazards(cph, data):
        """
        检验比例风险假设，通过 Schoenfeld 残差和时间交互项两种方法
        """
        print("\n=== 比例风险假设检验 ===")

        # 2.1 Schoenfeld 残差检验
        try:
            schoenfeld_test = cph.check_assumptions(data, show_plots=True)

            plt.figure(figsize=(15, 10))
            for i, (var, ax) in enumerate(schoenfeld_test.axes.items(), 1):
                ax.set_title(f'Schoenfeld Residuals for {var}')
                ax.grid(True)
            plt.tight_layout()
            plt.savefig('result/schoenfeld_residuals.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"警告：Schoenfeld 残差检验失败，可能是版本兼容性问题。错误信息：{str(e)}")
            print("继续进行其他检验...")

        # 2.2 时间交互检验
        print("\n开始进行时间交互检验...")
        time_transform = {
            'log': lambda x: np.log(x + 1),  # 加1避免取0的对数
            'identity': lambda x: x,
            'sqrt': lambda x: np.sqrt(x)
        }

        for transform_name, transform_func in time_transform.items():
            print(f"\n使用 {transform_name} 时间变换检验比例风险假设:")
            cox_time = CoxPHFitter()

            test_data = data.copy()
            for col in data.columns:
                if col not in ['duration', 'event']:
                    # 标准化原始变量以提高数值稳定性
                    test_data[col] = (test_data[col] - test_data[col].mean()) / test_data[col].std()
                    test_data[f'{col}_time_{transform_name}'] = \
                        test_data[col] * transform_func(test_data['duration'])

            try:
                cox_time.fit(test_data, duration_col='duration', event_col='event')
                time_vars = [col for col in cox_time.params_.index
                             if f'_time_{transform_name}' in col]

                print("\n时间交互项的显著性检验结果:")
                print("（如果交互项显著，说明可能违反比例风险假设）")
                summary = cox_time.print_summary()
                for var in time_vars:
                    p_value = float(summary.loc[var, 'p'])
                    coef = float(summary.loc[var, 'coef'])
                    print(f"{var:40s}: coef = {coef:8.3f}, p = {p_value:8.3f}")

                    if p_value < 0.05:
                        print(f"警告：{var} 的时间交互项显著 (p < 0.05)，可能违反比例风险假设")

            except Exception as e:
                print(f"警告：{transform_name} 时间交互检验失败。错误信息：{str(e)}")
                continue

    # 3. 无共线性假设检验函数
    def check_multicollinearity(data):
        """
        检验无共线性假设，通过计算 VIF 和相关系数矩阵
        """
        print("\n=== 无共线性假设检验 ===")

        # 3.1 准备数据
        X = data.drop(['duration', 'event'], axis=1)
        X = X.select_dtypes(include=[np.number]).astype(float)

        # 3.2 相关系数矩阵分析
        correlation_matrix = X.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        plt.title('特征相关系数矩阵')
        plt.tight_layout()
        plt.savefig('result/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3.3 VIF 分析
        # 添加常数项用于 VIF 计算
        X_with_const = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_const.columns

        # 逐个计算 VIF
        vif_values = []
        for i in range(X_with_const.shape[1]):
            if i == 0:  # 跳过常数项
                vif_values.append(np.nan)
                continue
            try:
                vif = variance_inflation_factor(X_with_const.values, i)
                vif_values.append(vif)
            except:
                vif_values.append(np.nan)

        vif_data["VIF"] = vif_values
        print("\nVIF 值 (>5 可能存在共线性问题):")
        print(vif_data.sort_values('VIF', ascending=False))

        # 3.4 识别高度相关的特征对
        high_correlation = np.where(np.abs(correlation_matrix) > 0.7)
        high_correlation = [(correlation_matrix.index[x],
                             correlation_matrix.columns[y],
                             correlation_matrix.iloc[x, y])
                            for x, y in zip(*high_correlation)
                            if x != y and x < y]

        if high_correlation:
            print("\n高度相关的特征对 (|相关系数| > 0.7):")
            for var1, var2, corr in high_correlation:
                print(f"{var1} - {var2}: {corr:.3f}")

    # 执行所有检验
    check_proportional_hazards(cph, cox_data_no_na)
    check_multicollinearity(cox_data_no_na)

    return cph


def fit_stratified_cox_with_time_interactions(df):
    """
    拟合带有时间交互项的分层Cox模型。

    这个函数会：
    1. 对优惠券类型(is_typeC)进行分层
    2. 为发券时间和有效期添加时间交互项
    3. 拟合模型并返回结果
    """
    # 1. 数据准备：创建时间交互项
    df_with_interactions = df.copy()

    # 为数值变量创建对数时间交互项
    # 加1是为了避免取log(0)
    df_with_interactions['day_issued_time_log'] = (
            df_with_interactions['day_issued_std'] *
            np.log(df_with_interactions['duration'] + 1)
    )
    df_with_interactions['validity_period_time_log'] = (
            df_with_interactions['validity_period_std'] *
            np.log(df_with_interactions['duration'] + 1)
    )

    # 2. 拟合分层Cox模型
    # 注意：我们现在直接在fit中指定列名
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        df_with_interactions,
        duration_col='duration',
        event_col='event',
        strata=['is_typeC'],  # 按优惠券类型分层
        show_progress=True,
        # 不再使用covariates参数，而是事先选择需要的列
    )

    # 3. 输出模型结果
    print("\n=== 模型摘要 ===")
    print(cph.print_summary())

    # 4. 可视化基线生存函数
    plt.figure(figsize=(10, 6))
    for strata in df_with_interactions['is_typeC'].unique():
        cph.plot_partial_effects('day_issued_std', values=[0],
                                 strata=strata,
                                 label=f'Type {"C" if strata else "B"}')
    plt.title('不同类型优惠券的基线生存函数')
    plt.xlabel('时间（天）')
    plt.ylabel('生存概率')
    plt.grid(True)
    plt.savefig('result/baseline_survival.png', dpi=300, bbox_inches='tight')
    plt.close()

    return cph, df_with_interactions


def evaluate_model_performance(cph, df_with_interactions):
    """
    评估模型性能并生成诊断图
    """
    # 1. 计算C-index
    c_index = cph.concordance_index_
    print(f"\nC-index: {c_index:.3f}")

    # 2. 计算每个分层的AIC
    aic = cph.AIC_partial_
    print(f"Partial AIC: {aic:.2f}")

    # 3. 计算预测风险值
    predictions = cph.predict_partial_hazard(df_with_interactions)

    # 4. 绘制预测风险分布图
    plt.figure(figsize=(10, 6))
    for is_type_c in [0, 1]:
        mask = df_with_interactions['is_typeC'] == is_type_c
        plt.hist(
            predictions[mask],
            bins=30,
            alpha=0.5,
            label=f'Type {"C" if is_type_c else "B"}',
            density=True
        )
    plt.title('预测风险值分布')
    plt.xlabel('预测风险值')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True)
    plt.savefig('result/risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


# 主函数：运行整个建模过程
def run_stratified_cox_analysis(df):
    """
    运行分层Cox分析流程，包含数据预处理、模型拟合和结果保存

    处理步骤：
    1. 准备基础的生存数据（duration和event）
    2. 创建Cox模型所需的所有变量
    3. 处理缺失值
    4. 添加时间交互项
    5. 拟合分层Cox模型
    6. 保存分析结果
    """
    # 1. 首先准备基础的生存数据
    print("\n=== 开始数据预处理 ===")
    df_survival = prepare_basic_survival_data(df)

    # 2. 创建Cox模型所需的变量
    print("准备Cox模型变量...")
    cox_data = prepare_cox_data(df_survival)

    # 3. 诊断和展示数据情况
    print("\n=== 数据完整性分析 ===")
    missing_pct = (cox_data.isnull().sum() / len(cox_data) * 100).round(2)
    print("\n各变量的缺失比例(%):")
    print(missing_pct)

    # 4. 保留完整数据的观察值
    cox_data_complete = cox_data.dropna()

    # 5. 创建时间交互项
    df_with_interactions = cox_data_complete.copy()
    df_with_interactions['day_issued_time_log'] = (
            df_with_interactions['day_issued_std'] *
            np.log(df_with_interactions['duration'] + 1)
    )
    df_with_interactions['validity_period_time_log'] = (
            df_with_interactions['validity_period_std'] *
            np.log(df_with_interactions['duration'] + 1)
    )

    # 6. 打印样本特征
    print("\n=== 样本特征分析 ===")
    print(f"总样本数: {len(cox_data)}")
    print(f"完整样本数: {len(cox_data_complete)}")
    print(f"兑换事件数: {cox_data_complete['event'].sum()}")
    print(f"平均兑换率: {(cox_data_complete['event'].mean() * 100):.2f}%")

    # 7. 拟合分层Cox模型
    print("\n=== 拟合Cox模型 ===")
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        df_with_interactions,
        duration_col='duration',
        event_col='event',
        strata=['is_typeC'],
        show_progress=True
    )

    # 8. 输出模型结果
    print("\n=== Cox 模型结果 ===")
    summary = cph.print_summary()

    # 9. 评估模型性能
    print("\n=== 模型性能指标 ===")
    performance_metrics = {
        "Concordance Index": f"{cph.concordance_index_:.3f}",
        "Partial AIC": f"{cph.AIC_partial_:.2f}",
        "Log Likelihood": f"{cph.log_likelihood_:.2f}"
    }
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")

    # 10. 绘制生存曲线
    plt.figure(figsize=(10, 6))
    for strata in [0, 1]:  # TypeB 和 TypeC
        mask = df_with_interactions['is_typeC'] == strata
        kmf = KaplanMeierFitter()
        kmf.fit(
            df_with_interactions[mask]['duration'],
            df_with_interactions[mask]['event'],
            label=f'Type {"C" if strata else "B"}'
        )
        kmf.plot()

    plt.title('不同类型优惠券的生存函数')
    plt.xlabel('时间（天）')
    plt.ylabel('生存概率')
    plt.grid(True)
    plt.savefig('result/survival_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 11. 保存详细结果到PDF
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # 注册中文字体（使用更通用的字体）
    try:
        # 尝试使用Arial Unicode MS（通常支持中文和特殊字符）
        pdfmetrics.registerFont(TTFont('Arial', '/Library/Fonts/Arial Unicode.ttf'))
        font_name = 'Arial'
    except:
        try:
            # 备选方案：尝试使用苹果系统自带的中文字体
            pdfmetrics.registerFont(TTFont('PingFang', '/System/Library/Fonts/PingFang.ttc'))
            font_name = 'PingFang'
        except:
            # 如果都失败了，使用默认字体，但可能不能显示中文
            font_name = 'Helvetica'

    # 创建支持中文的样式
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ChineseHeading1',
        fontName=font_name,
        fontSize=24,
        leading=30,
        spaceAfter=30
    ))
    styles.add(ParagraphStyle(
        name='ChineseHeading2',
        fontName=font_name,
        fontSize=18,
        leading=24,
        spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        name='ChineseBody',
        fontName=font_name,
        fontSize=12,
        leading=16
    ))

    doc = SimpleDocTemplate("result/cox_analysis_results.pdf", pagesize=letter)
    elements = []

    # 添加标题和内容，使用支持中文的样式
    elements.append(Paragraph("Cox 比例风险模型分析结果", styles['ChineseHeading1']))
    elements.append(Spacer(1, 12))

    # 数据概况
    elements.append(Paragraph("数据概况", styles['ChineseHeading2']))
    elements.append(Spacer(1, 6))
    data_summary = [
        ["指标", "数值"],
        ["总样本量", f"{len(cox_data):,}"],
        ["有效样本量", f"{len(cox_data_complete):,}"],
        ["事件数量", f"{cox_data_complete['event'].sum():,}"],
        ["事件率", f"{(cox_data_complete['event'].mean() * 100):.2f}%"]
    ]

    # 创建表格样式
    table_style = TableStyle([
        ('FONT', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6)
    ])

    summary_table = Table(data_summary)
    summary_table.setStyle(table_style)
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # 模型系数表
    elements.append(Paragraph("模型系数", styles['ChineseHeading2']))
    elements.append(Spacer(1, 6))

    # 准备模型系数数据
    coef_data = [["变量", "系数", "风险比", "标准误", "P值"]]
    for var in cph.params_.index:
        coef_data.append([
            var,  # 变量名
            f"{cph.params_[var]:.3f}",  # 系数
            f"{np.exp(cph.params_[var]):.3f}",  # 风险比
            f"{cph.standard_errors_[var]:.3f}",  # 标准误
            f"{cph.summary.loc[var, 'p']:.3f}"  # P值
        ])

    coef_table = Table(coef_data)
    coef_table.setStyle(table_style)
    elements.append(coef_table)
    elements.append(Spacer(1, 12))

    # 模型性能指标
    elements.append(Paragraph("模型性能指标", styles['ChineseHeading2']))
    elements.append(Spacer(1, 6))

    performance_data = [
        ["指标", "数值"],
        ["C-index (一致性指数)", f"{cph.concordance_index_:.3f}"],
        ["偏AIC", f"{cph.AIC_partial_:.2f}"],
        ["对数似然值", f"{cph.log_likelihood_:.2f}"]
    ]

    performance_table = Table(performance_data)
    performance_table.setStyle(table_style)
    elements.append(performance_table)
    elements.append(Spacer(1, 12))

    # 生存曲线
    elements.append(Paragraph("生存曲线", styles['ChineseHeading2']))
    elements.append(Spacer(1, 6))
    elements.append(Image('result/survival_curves.png', width=400, height=300))

    # 生成PDF
    doc.build(elements)
    print("\n完整结果已保存至 result/cox_analysis_results.pdf")

    return cph, df_with_interactions


def plot_partial_dependence_rsf(
    rsf,
    X,
    feature,
    times=[30],
    n_points=20,
    sample_size=None,
    figure_path="result/pdp_{feature}.png"
):
    """
    绘制对随机生存森林 (RSF) 的部分依赖图 (Partial Dependence Plot),
    展示当“feature”从低到高变化时, 在给定 times (一组时间点) 下,
    模型预测的生存概率(平均)如何变化。

    Args:
        rsf: 训练好的 RandomSurvivalForest 模型
        X:   特征 DataFrame, 每行对应一个样本
        feature: 要分析的特征列名 (str)
        times:   列表, 指定想查看的时间点, 比如 [30, 60, 90]
        n_points: 对 feature 坐标取多少个网格点
        sample_size: 若数据量大, 可以先抽样多少行再画, 减少计算量. 为 None 则不抽样
        figure_path: 图片保存路径, 默认 result/pdp_{feature}.png
                     其中 {feature} 会被替换为实际特征名

    Note:
        - 对于连续特征, 我们在 min ~ max 之间均匀取 n_points 个值
        - 对于离散特征, 也可以取 unique 值.
        - 这里做的是一条“平均”曲线, 对所有样本, 只修改该 feature, 其它特征保持原样.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 若需要抽样, 以减少计算量
    if sample_size is not None and sample_size < len(X):
        X_sample = X.sample(sample_size, random_state=42).copy()
    else:
        X_sample = X.copy()

    # 计算 feature 的取值网格
    x_min, x_max = X_sample[feature].min(), X_sample[feature].max()
    grid_values = np.linspace(x_min, x_max, n_points)

    # 对每个 time, 计算 PDP 曲线(网格长度= n_points)
    pd_curves = {}  # time -> [平均生存概率...]
    for t in times:
        pd_curves[t] = []

    # 主循环：在每个 grid_val 上, 替换 X_sample[feature] -> grid_val, 预测生存概率
    for grid_val in grid_values:
        X_modified = X_sample.copy()
        X_modified[feature] = grid_val  # 替换目标特征

        # 调用 RSF 的 predict_survival_function(), 得到每行的生存函数
        surv_fns = rsf.predict_survival_function(X_modified)
        # surv_fns 是个 list, 每个元素是个可调用, 你可以 surv_fns[i](time)
        # scikit-survival 会返回 array-like 的生存函数

        # 对所有样本, 计算在指定时间点 times[i] 下的生存概率, 然后取平均
        n_sample = len(X_modified)
        # 循环 times
        for t in times:
            # 对每个样本 i, surv_fns[i](t), 最后平均
            surv_probs = [fn(t) for fn in surv_fns]
            mean_prob = np.mean(surv_probs)
            pd_curves[t].append(mean_prob)

    # 绘图
    plt.figure(figsize=(8, 6))
    for t in times:
        plt.plot(
            grid_values,
            pd_curves[t],
            label=f"Time={t} days"
        )

    plt.title(f"Partial Dependence of '{feature}' on Survival Probability")
    plt.xlabel(f"{feature} value")
    plt.ylabel("Average Survival Probability")
    plt.grid(True)
    plt.legend()
    save_path = figure_path.format(feature=feature)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PDP for '{feature}' saved as {save_path}")


def run_rsf_analysis(df):
    X, y, scaler = prepare_rsf_data(df)

    na_mask = X.isna().any(axis=1)
    print("Number of rows with NaN in X:", na_mask.sum())
    X_no_na = X[~na_mask].copy()
    y_no_na = y[~na_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_no_na, y_no_na,
        test_size=0.2,
        random_state=42
    )

    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_leaf=15,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    print("\nFitting Random Survival Forest...")
    rsf.fit(X_train, y_train)

    train_score = rsf.score(X_train, y_train)
    test_score = rsf.score(X_test, y_test)
    print(f"\nModel Performance:")
    print(f"Training C-index: {train_score:.3f}")
    print(f"Testing C-index: {test_score:.3f}")

    r = permutation_importance(
        rsf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    importance_df = pd.DataFrame({
        'feature': X_no_na.columns,
        'importance_mean': r.importances_mean,
        'importance_std': r.importances_std
    }).sort_values('importance_mean', ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

    plot_feature_importance(importance_df)

    print("\nGenerating survival curves for different scenarios...")
    plot_survival_curves(rsf, y_train, X_no_na)

    print("\nFeature Summary Statistics:")
    print(X_no_na.describe())

    # 额外：对前 5 个最重要的特征作 PDP
    top_features = importance_df['feature'].head(5).tolist()  # 取排名前5
    for f in top_features:
        # 在 30, 60, 90 三个时间点画 PDP
        plot_partial_dependence_rsf(
            rsf,
            X_train,
            feature=f,
            times=[30, 60, 90],
            n_points=20,
            sample_size=200,
            figure_path="result/pdp_{feature}.png"
        )

    return rsf, importance_df, (X_train, X_test, y_train, y_test)

############################
#   保存模型结果 (PDF)     #
############################

def save_model_results(cph, rsf, importance_df, file_path='result/model_results.pdf'):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    import numpy as np

    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Survival Analysis Results", styles['Heading1']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Cox Proportional Hazards Model Results", styles['Heading2']))
    elements.append(Spacer(1, 12))

    cox_stats = [
        ['Number of observations', str(cph._n_examples)],
        ['Number of events', str(cph.event_observed.sum())],
        ['Concordance', f"{cph.concordance_index_:.3f}"],
        ['Partial AIC', f"{cph.AIC_partial_:.2f}"],
        ['Log-likelihood ratio test', f"{cph.log_likelihood_ratio_test().test_statistic:.2f}"],
        ['Number of parameters', str(len(cph.params_))]
    ]

    coef_df = pd.DataFrame({
        'coef': cph.params_,
        'exp(coef)': np.exp(cph.params_),
        'se(coef)': cph.standard_errors_,
        'z': cph.summary['z'],
        'p': cph.summary['p'],
    }).round(3)

    coef_data = [['Variable', 'Coef', 'Exp(coef)', 'SE', 'z', 'p-value']] + \
                [[idx] + list(row) for idx, row in coef_df.iterrows()]

    def create_table_with_style(data, header=True):
        if header:
            style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]
        else:
            style = [
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]
        return Table(data, style=TableStyle(style))

    elements.append(create_table_with_style(cox_stats, header=False))
    elements.append(Spacer(1, 12))
    elements.append(create_table_with_style(coef_data))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Random Survival Forest Results", styles['Heading2']))
    elements.append(Spacer(1, 12))

    importance_data = [['Feature', 'Importance', 'Std']] + importance_df.values.tolist()
    elements.append(create_table_with_style(importance_data))

    doc.build(elements)
    print(f"Results saved to {file_path}")


def create_coupon_base_table(coupon_path: str, campaign_desc_path: str, campaign_table_path: str):
    """
    Creates the base coupon issuance table with unique IDs for each coupon issuance.
    """
    df_coupon = pd.read_csv(coupon_path)
    df_campaign_desc = pd.read_csv(campaign_desc_path)
    df_campaign_table = pd.read_csv(campaign_table_path)

    # 只取 TypeB / TypeC
    campaign_list = df_campaign_desc[df_campaign_desc['DESCRIPTION'].isin(['TypeB', 'TypeC'])]['CAMPAIGN'].tolist()

    # 对 coupon 做分组
    campaign_bundle = df_coupon.groupby('CAMPAIGN')['COUPON_UPC'].apply(set).apply(list).to_dict()
    campaign_bundle = {k: v for k, v in campaign_bundle.items() if k in campaign_list}

    # household -> campaign
    household_bundle = df_campaign_table.groupby('household_key')['CAMPAIGN'].apply(list).to_dict()

    # household -> (campaign -> coupon list)
    UID_bundle = {}
    for key, value in household_bundle.items():
        temp = {}
        for campaign in value:
            if campaign not in campaign_bundle:
                continue
            temp[campaign] = campaign_bundle[campaign]
        UID_bundle[key] = temp

    # 转为 DataFrame
    df_UID = pd.DataFrame(
        {
            'household_key': household_key,
            'CAMPAIGN': campaign,
            'COUPON_UPC': coupon
        }
        for household_key, campaign_bundle in UID_bundle.items()
        for campaign, coupons in campaign_bundle.items()
        for coupon in coupons
    )

    df_UID['coupon_uid'] = df_UID.index + 1
    df_UID = df_UID[['coupon_uid', 'household_key', 'CAMPAIGN', 'COUPON_UPC']]

    return df_UID

############################
#  有序 vs 无序的处理函数  #
############################

def prepare_basic_survival_data(df, hh_demographic_path='data/hh_demographic.csv'):
    """
    计算基础生存时间 & 事件指示符，并合并人口统计数据
    """
    hh_demo = pd.read_csv(hh_demographic_path)

    # duration
    df['duration'] = df.apply(
        lambda x: x['day_redeemed'] - x['day_issued'] if pd.notnull(x['day_redeemed'])
        else x['day_expired'] - x['day_issued'],
        axis=1
    )
    df['event'] = df['day_redeemed'].notna().astype(int)

    # 合并人口统计
    df = df.merge(hh_demo, on='household_key', how='left')
    return df

def extract_age_group(s):
    """ 'Age GroupX' -> X """
    if not isinstance(s, str):
        return np.nan
    return int(s.replace("Age Group", "").strip())

def extract_level(s):
    """ 'LevelN' -> N """
    if not isinstance(s, str):
        return np.nan
    return int(s.replace("Level", "").strip())

def parse_class_4(val):
    """
    classification_4: 可能有 '5+', '2', '3' 等
    如果带 '+', 则 +1
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str.endswith('+'):
        base = val_str.replace('+', '')
        try:
            return int(base) + 1
        except:
            return np.nan
    else:
        try:
            return int(val_str)
        except:
            return np.nan

def parse_class_5(val):
    """ 'Group1' -> 1 ... 'Group6' -> 6 """
    if pd.isna(val):
        return np.nan
    classification_5_map = {
        'Group1': 1, 'Group2': 2, 'Group3': 3,
        'Group4': 4, 'Group5': 5, 'Group6': 6
    }
    return classification_5_map.get(val, np.nan)


############################
#   prepare_cox_data()    #
############################

def parse_kid(val):
    """
    'None/Unknown'->0, '1'->1, '2'->2, '3+'->3 (或4)
    """
    kid_map = {
        'None/Unknown': 0,
        '1': 1,
        '2': 2,
        '3+': 3
    }
    return kid_map.get(val, np.nan)


############################
#   prepare_rsf_data()    #
############################

def prepare_cox_data(df):
    """
    1) 先把 classification_1, 3, 4, 5, KID_CATEGORY_DESC 做有序处理
    2) 把 classification_2, HOMEOWNER_DESC 用 get_dummies 处理
    3) 标准化 day_issued, validity_period
    4) 生成最终 Cox DataFrame
    """
    # 先手动做有序列
    # classification_1 -> Age Group
    if 'classification_1' in df.columns:
        df['classification_1_ordinal'] = df['classification_1'].apply(extract_age_group)

    # classification_3 -> Level
    if 'classification_3' in df.columns:
        df['classification_3_ordinal'] = df['classification_3'].apply(extract_level)

    # classification_4
    if 'classification_4' in df.columns:
        df['classification_4_ordinal'] = df['classification_4'].apply(parse_class_4)

    # classification_5
    if 'classification_5' in df.columns:
        df['classification_5_ordinal'] = df['classification_5'].apply(parse_class_5)

    # KID_CATEGORY_DESC
    if 'KID_CATEGORY_DESC' in df.columns:
        df['KID_CATEGORY_DESC_ordinal'] = df['KID_CATEGORY_DESC'].apply(parse_kid)

    # classification_2 & HOMEOWNER_DESC -> dummies
    cat_cols_for_dummies = []
    if 'classification_2' in df.columns:
        cat_cols_for_dummies.append('classification_2')
    if 'HOMEOWNER_DESC' in df.columns:
        cat_cols_for_dummies.append('HOMEOWNER_DESC')

    # 生成 dummy
    dummy_dfs = []
    for col in cat_cols_for_dummies:
        dummy_dfs.append(pd.get_dummies(df[col], prefix=col, drop_first=True))

    # is_typeC
    typeC_campaigns = [3, 6, 14, 15, 20, 27]
    df['is_typeC'] = df['CAMPAIGN'].isin(typeC_campaigns).astype(int)

    # validity_period
    df['validity_period'] = df['day_expired'] - df['day_issued']

    # 标准化
    df['day_issued_std'] = (df['day_issued'] - df['day_issued'].mean()) / df['day_issued'].std()
    df['validity_period_std'] = (df['validity_period'] - df['validity_period'].mean()) / df['validity_period'].std()

    # 拼合
    base_cols = ['duration', 'event', 'is_typeC', 'day_issued_std', 'validity_period_std',
                 'classification_1_ordinal','classification_3_ordinal',
                 'classification_4_ordinal','classification_5_ordinal','KID_CATEGORY_DESC_ordinal']
    cox_data_parts = [df[base_cols]]  # 主体

    if len(dummy_dfs) > 0:
        cox_data_parts += dummy_dfs

    cox_data = pd.concat(cox_data_parts, axis=1)

    return cox_data

#####################
#  KM / Log Rank   #
#####################

def prepare_rsf_data(df):
    """
    1) classification_1, 3, 4, 5, KID_CATEGORY_DESC -> ordinal
    2) classification_2, HOMEOWNER_DESC -> dummy
    3) 标准化 day_issued, validity_period
    4) 构造 X, y
    """
    # 同上做法
    if 'classification_1' in df.columns:
        df['classification_1_ordinal'] = df['classification_1'].apply(extract_age_group)
    if 'classification_3' in df.columns:
        df['classification_3_ordinal'] = df['classification_3'].apply(extract_level)
    if 'classification_4' in df.columns:
        df['classification_4_ordinal'] = df['classification_4'].apply(parse_class_4)
    if 'classification_5' in df.columns:
        df['classification_5_ordinal'] = df['classification_5'].apply(parse_class_5)
    if 'KID_CATEGORY_DESC' in df.columns:
        df['KID_CATEGORY_DESC_ordinal'] = df['KID_CATEGORY_DESC'].apply(parse_kid)

    # dummies
    cat_cols_for_dummies = []
    if 'classification_2' in df.columns:
        cat_cols_for_dummies.append('classification_2')
    if 'HOMEOWNER_DESC' in df.columns:
        cat_cols_for_dummies.append('HOMEOWNER_DESC')

    dummy_dfs = []
    for col in cat_cols_for_dummies:
        dummy_dfs.append(pd.get_dummies(df[col], prefix=col, drop_first=True))

    # is_typeC
    typeC_campaigns = [3, 6, 14, 15, 20, 27]
    df['is_typeC'] = df['CAMPAIGN'].isin(typeC_campaigns).astype(int)

    # validity_period
    df['validity_period'] = df['day_expired'] - df['day_issued']

    # 标准化
    scaler = StandardScaler()
    features_to_scale = ['day_issued', 'validity_period']
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # 组装 X
    base_feature_cols = [
        'is_typeC', 'day_issued', 'validity_period',
        'classification_1_ordinal',
        'classification_3_ordinal',
        'classification_4_ordinal',
        'classification_5_ordinal',
        'KID_CATEGORY_DESC_ordinal'
    ]
    X_parts = [df_scaled[base_feature_cols]]  # 主体
    if len(dummy_dfs) > 0:
        X_parts += dummy_dfs

    X = pd.concat(X_parts, axis=1)

    # 构造 y
    y = np.zeros(len(df), dtype=[('event', bool), ('time', float)])
    y['event'] = df['event'].astype(bool)
    y['time'] = df['duration']

    return X, y, scaler


def plot_km_curve(df, group_col=None):
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    if group_col is None:
        kmf.fit(df['duration'], df['event'], label='Overall')
        kmf.plot()
    else:
        for group in df[group_col].unique():
            mask = df[group_col] == group
            kmf.fit(df[mask]['duration'], df[mask]['event'], label=f'{group_col}={group}')
            kmf.plot()

    plt.title('Coupon Survival Curve')
    plt.xlabel('Days since coupon issued')
    plt.ylabel('Survival probability')
    plt.grid(True)
    return plt


def basic_survival_analysis(df):
    import matplotlib
    matplotlib.use('TkAgg')

    df = prepare_basic_survival_data(df)

    print("Basic Survival Statistics:")
    print(f"Total coupons: {len(df)}")
    print(f"Redemption events: {df['event'].sum()}")
    print(f"Redemption rate: {df['event'].mean():.2%}")
    print("\nDuration Statistics (days):")
    print(df['duration'].describe())

    # Plot overall
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    kmf.fit(df['duration'], df['event'], label='Overall')
    kmf.plot(ax=ax1)
    ax1.set_title('Overall Coupon Survival Curve')
    ax1.set_xlabel('Days since coupon issued')
    ax1.set_ylabel('Survival probability')
    ax1.grid(True)
    fig1.savefig('result/overall_survival.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot campaign-specific
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for campaign in sorted(df['CAMPAIGN'].unique()):
        mask = df['CAMPAIGN'] == campaign
        if mask.sum() > 0:
            kmf.fit(df[mask]['duration'], df[mask]['event'], label=f'Campaign {campaign}')
            kmf.plot(ax=ax2)

    ax2.set_title('Coupon Survival Curves by Campaign')
    ax2.set_xlabel('Days since coupon issued')
    ax2.set_ylabel('Survival probability')
    ax2.grid(True)
    ax2.legend()
    fig2.savefig('result/campaign_survival.png', dpi=300, bbox_inches='tight')
    plt.show()

    campaign_stats = df.groupby('CAMPAIGN').agg({
        'event': ['count', 'sum', 'mean'],
        'duration': ['mean', 'std', 'min', 'max']
    }).round(4)
    print("\nCampaign-level Statistics:")
    print(campaign_stats)

    return df

############################
#   Cox 分析 + 可视化     #
############################

def analyze_campaign_types(df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    typeC_campaigns = [3, 6, 14, 15, 20, 27]
    df['campaign_type'] = df['CAMPAIGN'].apply(lambda x: 'TypeC' if x in typeC_campaigns else 'TypeB')

    typeB_mask = df['campaign_type'] == 'TypeB'
    typeC_mask = df['campaign_type'] == 'TypeC'

    results = logrank_test(
        df[typeB_mask]['duration'],
        df[typeC_mask]['duration'],
        df[typeB_mask]['event'],
        df[typeC_mask]['event']
    )

    print("Log-rank Test Results:")
    print(f"Test statistic: {results.test_statistic:.4f}")
    print(f"P-value: {results.p_value:.4f}")

    print("\nBasic Statistics by Campaign Type:")
    stats_by_type = df.groupby('campaign_type').agg({
        'coupon_uid': 'count',
        'event': ['sum', 'mean'],
        'duration': ['mean', 'std']
    }).round(4)
    stats_by_type.columns = ['Total Coupons', 'Redemptions', 'Redemption Rate', 'Mean Duration', 'Std Duration']
    print(stats_by_type)

    plt.figure(figsize=(12, 6))
    kmf_B = KaplanMeierFitter()
    kmf_B.fit(df[typeB_mask]['duration'], df[typeB_mask]['event'], label=f'TypeB (n={sum(typeB_mask)})')
    kmf_B.plot()

    kmf_C = KaplanMeierFitter()
    kmf_C.fit(df[typeC_mask]['duration'], df[typeC_mask]['event'], label=f'TypeC (n={sum(typeC_mask)})')
    kmf_C.plot()

    plt.title('Survival Curves by Campaign Type\n' +
              f'Log-rank test p-value: {results.p_value:.4f}')
    plt.xlabel('Days since coupon issued')
    plt.ylabel('Survival probability')
    plt.grid(True)
    plt.savefig('result/campaign_type_survival.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results, stats_by_type


def plot_hazard_ratios(cph, filename="result/hazard_ratios.png", title="Significant Hazard Ratios from Cox Model"):
    hazard_ratios = pd.DataFrame({
        'HR': np.exp(cph.params_),
        'lower': np.exp(cph.confidence_intervals_.iloc[:, 0]),
        'upper': np.exp(cph.confidence_intervals_.iloc[:, 1])
    })
    hazard_ratios.index = cph.params_.index

    main_effects = ['is_typeC', 'day_issued_std', 'validity_period_std']
    significant_vars = cph.summary.index[cph.summary['p'] < 0.05].tolist()
    vars_to_plot = list(set(main_effects + significant_vars))

    hazard_ratios_filtered = hazard_ratios.loc[vars_to_plot]

    plt.figure(figsize=(12, len(vars_to_plot) * 0.5))
    plt.clf()
    y_positions = range(len(vars_to_plot))

    plt.errorbar(
        x=hazard_ratios_filtered['HR'],
        y=y_positions,
        xerr=[
            hazard_ratios_filtered['HR'] - hazard_ratios_filtered['lower'],
            hazard_ratios_filtered['upper'] - hazard_ratios_filtered['HR']
        ],
        fmt='o',
        capsize=5,
        color='blue',
        markersize=8
    )

    plt.axvline(x=1, color='k', linestyle='--', alpha=0.5)
    plt.yticks(y_positions, vars_to_plot)
    plt.xlabel('Hazard Ratio (95% CI)')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    for i, (var, row) in enumerate(hazard_ratios_filtered.iterrows()):
        plt.text(
            row['upper'] + 0.01,
            i,
            f"HR={row['HR']:.2f}\n({row['lower']:.2f}-{row['upper']:.2f})",
            verticalalignment='center'
        )

    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

############################
#   RSF 分析 + 可视化      #
############################

def run_cox_analysis(df):
    cox_data = prepare_cox_data(df)

    print("\nData Summary:")
    print(f"Total observations: {len(cox_data)}")
    print(f"Events (redemptions): {cox_data['event'].sum()}")
    print(f"Censored: {len(cox_data) - cox_data['event'].sum()}")

    # dropna
    cox_data_no_na = cox_data.dropna(axis=0)
    print(f"After dropna: {len(cox_data_no_na)} rows remain.")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        cox_data_no_na,
        duration_col='duration',
        event_col='event',
        show_progress=True
    )

    print("\nCox Model Summary:")
    print(cph.print_summary())

    hr_fig = plot_hazard_ratios(cph)  # 默认保存到 result/hazard_ratios.png
    plt.close(hr_fig)

    print("\nTesting Proportional Hazards Assumption:")
    assumptions_fig = cph.check_assumptions(cox_data_no_na, show_plots=True)
    plt.tight_layout()
    plt.savefig('result/proportional_hazards_test.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    print("\nFeature Correlations:")
    correlations = cox_data_no_na.corr().round(4)
    print(correlations)

    return cph, cox_data_no_na


def plot_feature_importance(importance_df, filename="result/rsf_feature_importance.png"):
    plt.figure(figsize=(10, 6))
    plt.clf()
    bars = plt.bar(
        range(len(importance_df)),
        importance_df['importance_mean'],
        yerr=importance_df['importance_std'],
        capsize=5,
        color='skyblue'
    )

    plt.xticks(
        range(len(importance_df)),
        importance_df['feature'],
        rotation=45,
        ha='right'
    )
    plt.title('Feature Importance in Random Survival Forest')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.grid(True, alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_results(df, cox_data, importance_df, file_path='result/detailed_results.xlsx'):
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df.describe().round(3).to_excel(writer, sheet_name='Data_Summary')
        cox_data.describe().round(3).to_excel(writer, sheet_name='Cox_Data')
        importance_df.round(3).to_excel(writer, sheet_name='Feature_Importance')

        workbook = writer.book
        worksheet = workbook.add_worksheet('Charts')
        importance_data = importance_df.values.tolist()
        for i, row in enumerate(importance_data):
            worksheet.write_row(i, 0, row)

    print(f"Detailed results saved to {file_path}")


def plot_survival_curves(rsf, y_train, X, filename="result/rsf_survival_curves.png"):
    plt.figure(figsize=(10, 6))
    plt.clf()

    times = np.percentile(y_train['time'], np.linspace(0, 100, 100))

    scenario_base = pd.DataFrame(np.zeros((2, X.shape[1])), columns=X.columns)
    scenario_base['day_issued'] = X['day_issued'].mean()
    scenario_base['validity_period'] = X['validity_period'].mean()

    scenario_base.iloc[0, scenario_base.columns.get_loc('is_typeC')] = 0  # TypeB
    scenario_base.iloc[1, scenario_base.columns.get_loc('is_typeC')] = 1  # TypeC

    for col in X.columns:
        if col not in ['is_typeC', 'day_issued', 'validity_period']:
            most_common = X[col].mode()[0]
            scenario_base[col] = most_common

    for i, row in scenario_base.iterrows():
        surv_fn = rsf.predict_survival_function(row.values.reshape(1, -1))[0]
        surv_probs = [surv_fn(t) for t in times]
        label_str = "Type C" if row['is_typeC'] else "Type B"
        plt.plot(
            times,
            surv_probs,
            label=label_str,
            linewidth=2
        )

    plt.title('Predicted Survival Curves by Campaign Type\n(with average demographic features)')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival probability')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


#########################
#         MAIN         #
#########################

def main():
    # 生成数据表
    df = create_complete_dataset()

    # 2) 生存分析（可选）
    # df = prepare_basic_survival_data(df)
    ## 检验数据是否满足 COX 模型的统计假设
    # cph = check_cox_assumptions(df)

    ## 用分层和加入时间交互项的方式跑 cox 分析
    # 运行分析

    # 分层 COX 模型
    cph, df_with_interactions = run_stratified_cox_analysis(df)

    print("\nAll tasks completed successfully.")

"""

    # 2.5) ---------------------- 保存 df 到本地 CSV ----------------------
    df.to_csv("result/master_dataset.csv", index=False, encoding="utf_8_sig")
    print("DataFrame saved to result/final_dataset.csv (UTF-8 with BOM).")

    df_km = basic_survival_analysis(df)
    df_lr, stats_by_type = analyze_campaign_types(df)

    # 3) Cox
    cph, cox_data_no_na = run_cox_analysis(df)

    # 4) RSF
    rsf_model, importance_df, split_data = run_rsf_analysis(df)

    # 5) 导出结果 => /result/ 下
    save_model_results(cph, rsf_model, importance_df, 'result/survival_analysis_results.pdf')
    save_detailed_results(df, cox_data_no_na, importance_df, 'result/survival_analysis_detailed.xlsx')
"""

# 若直接跑脚本，则调用 main()
if __name__ == "__main__":
    main()