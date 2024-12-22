import os
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
from lifelines import CoxPHFitter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import seaborn as sns
sns.set_style('whitegrid')
import warnings
import reportlab
warnings.filterwarnings('ignore')

# 建议：若没有 result 文件夹，先创建
os.makedirs('result', exist_ok=True)

#####################
#  数据清洗 & 整合  #
#####################

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

############################
#  有序 vs 无序的处理函数  #
############################

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
#   prepare_cox_data()    #
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


############################
#   prepare_rsf_data()    #
############################

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

#####################
#  KM / Log Rank   #
#####################

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

############################
#   Cox 分析 + 可视化     #
############################

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

############################
#   RSF 分析 + 可视化      #
############################

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


#########################
#         MAIN         #
#########################

def main():
    # 1) 生成最终数据
    df = create_complete_dataset()

    # 2) 生存分析（可选）
    df = prepare_basic_survival_data(df)

    # 2.5) ---------------------- 保存 df 到本地 CSV ----------------------
    df.to_csv("result/master_dataset.csv", index=False, encoding="utf_8_sig")
    print("DataFrame saved to result/final_dataset.csv (UTF-8 with BOM).")

    # df_km = basic_survival_analysis(df)
    # df_lr, stats_by_type = analyze_campaign_types(df)

    # 3) Cox
    cph, cox_data_no_na = run_cox_analysis(df)

    # 4) RSF
    rsf_model, importance_df, split_data = run_rsf_analysis(df)

    # 5) 导出结果 => /result/ 下
    save_model_results(cph, rsf_model, importance_df, 'result/survival_analysis_results.pdf')
    save_detailed_results(df, cox_data_no_na, importance_df, 'result/survival_analysis_detailed.xlsx')

    print("\nAll tasks completed successfully.")

# 若直接跑脚本，则调用 main()
if __name__ == "__main__":
    main()