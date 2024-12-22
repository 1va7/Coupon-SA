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


def create_coupon_base_table(coupon_path: str, campaign_desc_path: str, campaign_table_path: str):
    """
    Creates the base coupon issuance table with unique IDs for each coupon issuance.

    Args:
        coupon_path: Path to the coupon CSV file
        campaign_desc_path: Path to the campaign description CSV file
        campaign_table_path: Path to the campaign table CSV file
    """
    # Read the input files
    df_coupon = pd.read_csv(coupon_path)
    df_campaign_desc = pd.read_csv(campaign_desc_path)
    df_campaign_table = pd.read_csv(campaign_table_path)

    # Get list of TypeB and TypeC campaigns
    campaign_list = df_campaign_desc[df_campaign_desc['DESCRIPTION'].isin(['TypeB', 'TypeC'])]['CAMPAIGN'].tolist()

    # Create campaign_bundle: mapping of campaigns to their coupons
    campaign_bundle = df_coupon.groupby('CAMPAIGN')['COUPON_UPC'].apply(set).apply(list).to_dict()
    campaign_bundle = {k: v for k, v in campaign_bundle.items() if k in campaign_list}

    # Create household_bundle: mapping of households to their campaigns
    household_bundle = df_campaign_table.groupby('household_key')['CAMPAIGN'].apply(list).to_dict()

    # Create UID_bundle: final mapping of households to their campaigns and coupons
    UID_bundle = {}
    for key, value in household_bundle.items():
        temp = {}
        for campaign in value:
            if campaign not in campaign_bundle:
                continue
            temp[campaign] = campaign_bundle[campaign]
        UID_bundle[key] = temp

    # Convert to DataFrame
    df_UID = pd.DataFrame(
        {'household_key': household_key, 'CAMPAIGN': campaign, 'COUPON_UPC': coupon}
        for household_key, campaign_bundle in UID_bundle.items()
        for campaign, coupons in campaign_bundle.items()
        for coupon in coupons
    )

    # Generate unique IDs
    df_UID['coupon_uid'] = df_UID.index + 1

    # Reorder columns
    df_UID = df_UID[['coupon_uid', 'household_key', 'CAMPAIGN', 'COUPON_UPC']]

    return df_UID


def add_dates_and_redemptions(base_df, campaign_desc_path: str, coupon_redempt_path: str):
    """
    Adds issuance, expiration, and redemption dates to the base coupon table.
    Excludes TypeA campaigns from redemption data.
    """
    # Define TypeA campaigns
    typeA_campaigns = [8, 13, 18, 26, 30]

    # Add campaign dates
    campaign_desc = pd.read_csv(campaign_desc_path)
    with_dates = base_df.merge(
        campaign_desc[['CAMPAIGN', 'START_DAY', 'END_DAY']],
        on='CAMPAIGN',
        how='left'
    ).rename(columns={'START_DAY': 'day_issued', 'END_DAY': 'day_expired'})

    # Add redemption data, excluding TypeA campaigns
    coupon_redempt = pd.read_csv(coupon_redempt_path)
    coupon_redempt_filtered = coupon_redempt[~coupon_redempt['CAMPAIGN'].isin(typeA_campaigns)]

    # Since duplicates were found in redemption data, keep only the first redemption
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
    Creates and saves the complete coupon dataset, excluding TypeA campaigns.
    """
    # Define TypeA campaigns
    typeA_campaigns = [8, 13, 18, 26, 30]

    # Create base table
    base_df = create_coupon_base_table(
        'data/coupon.csv',
        'data/campaign_desc.csv',
        'data/campaign_table.csv'
    )

    # Filter out TypeA campaigns from base table
    base_df_filtered = base_df[~base_df['CAMPAIGN'].isin(typeA_campaigns)]

    # Add dates and redemption data
    final_df = add_dates_and_redemptions(
        base_df_filtered,
        'data/campaign_desc.csv',
        'data/coupon_redempt.csv'
    )

    return final_df

    # Save the final dataset
    # final_df.to_csv('coupon_complete_data.csv', index=False)


def prepare_basic_survival_data(df, hh_demographic_path='data/hh_demographic.csv'):
    """
    Prepare the basic survival analysis dataset with demographic features
    """
    # 读取人口统计数据
    hh_demo = pd.read_csv(hh_demographic_path)

    # 计算基础的生存时间和事件指示符
    df['duration'] = df.apply(
        lambda x: x['day_redeemed'] - x['day_issued'] if pd.notnull(x['day_redeemed'])
        else x['day_expired'] - x['day_issued'],
        axis=1
    )
    df['event'] = df['day_redeemed'].notna().astype(int)

    # 合并人口统计数据
    df = df.merge(
        hh_demo,
        on='household_key',
        how='left'
    )

    return df


def prepare_cox_data(df):
    """
    Prepare data for Cox proportional hazards model with demographic features
    """
    # Campaign type indicator
    typeC_campaigns = [3, 6, 14, 15, 20, 27]
    df['is_typeC'] = df['CAMPAIGN'].isin(typeC_campaigns).astype(int)

    # Calculate validity period
    df['validity_period'] = df['day_expired'] - df['day_issued']

    # 对连续变量进行标准化
    df['day_issued_std'] = (df['day_issued'] - df['day_issued'].mean()) / df['day_issued'].std()
    df['validity_period_std'] = (df['validity_period'] - df['validity_period'].mean()) / df['validity_period'].std()

    # 处理分类变量
    categorical_columns = {
        'classification_1': 'class1',
        'classification_2': 'class2',
        'classification_3': 'class3',
        'HOMEOWNER_DESC': 'homeowner',
        'classification_4': 'class4',
        'classification_5': 'class5',
        'KID_CATEGORY_DESC': 'kids'
    }

    # 为每个分类变量创建虚拟变量
    dummy_dfs = []
    for col, prefix in categorical_columns.items():
        if col in df.columns:
            dummy_df = pd.get_dummies(df[col], prefix=prefix, drop_first=True)
            dummy_dfs.append(dummy_df)

    # 准备最终的数据集
    cox_data = pd.concat(
        [df[['duration', 'event', 'is_typeC', 'day_issued_std', 'validity_period_std']]] + dummy_dfs,
        axis=1
    )

    return cox_data


def prepare_rsf_data(df):
    """
    Prepare data for Random Survival Forest analysis with demographic features
    """
    # Create campaign type indicator
    typeC_campaigns = [3, 6, 14, 15, 20, 27]
    df['is_typeC'] = df['CAMPAIGN'].isin(typeC_campaigns).astype(int)

    # Calculate validity period
    df['validity_period'] = df['day_expired'] - df['day_issued']

    # Standardize continuous variables
    scaler = StandardScaler()
    features_to_scale = ['day_issued', 'validity_period']
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # 处理分类变量
    categorical_columns = {
        'classification_1': 'class1',
        'classification_2': 'class2',
        'classification_3': 'class3',
        'HOMEOWNER_DESC': 'homeowner',
        'classification_4': 'class4',
        'classification_5': 'class5',
        'KID_CATEGORY_DESC': 'kids'
    }

    # 为每个分类变量创建虚拟变量
    dummy_dfs = []
    for col, prefix in categorical_columns.items():
        if col in df.columns:
            dummy_df = pd.get_dummies(df[col], prefix=prefix, drop_first=True)
            dummy_dfs.append(dummy_df)

    # Prepare feature matrix
    X = pd.concat(
        [df_scaled[['is_typeC', 'day_issued', 'validity_period']]] + dummy_dfs,
        axis=1
    )

    # Prepare survival data structure
    y = np.zeros(len(df), dtype=[('event', bool), ('time', float)])
    y['event'] = df['event'].astype(bool)
    y['time'] = df['duration']

    return X, y, scaler

def plot_km_curve(df, group_col=None):
    """
    Plot Kaplan-Meier survival curves
    If group_col is provided, create separate curves for each group
    """
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    if group_col is None:
        # Plot single curve for all data
        kmf.fit(
            df['duration'],
            df['event'],
            label='Overall'
        )
        kmf.plot()
    else:
        # Plot curves for each group
        for group in df[group_col].unique():
            mask = df[group_col] == group
            kmf.fit(
                df[mask]['duration'],
                df[mask]['event'],
                label=f'{group_col}={group}'
            )
            kmf.plot()

    plt.title('Coupon Survival Curve')
    plt.xlabel('Days since coupon issued')
    plt.ylabel('Survival probability')
    plt.grid(True)

    return plt


def basic_survival_analysis(df):
   """
   Execute basic survival analysis with Kaplan-Meier curves
   """
   import matplotlib
   matplotlib.use('TkAgg')

   # Prepare survival data
   df = prepare_basic_survival_data(df)

   # Print basic statistics
   print("Basic Survival Statistics:")
   print(f"Total coupons: {len(df)}")
   print(f"Redemption events: {df['event'].sum()}")
   print(f"Redemption rate: {df['event'].mean():.2%}")
   print("\nDuration Statistics (days):")
   print(df['duration'].describe())

   # Plot overall survival curve
   fig1, ax1 = plt.subplots(figsize=(10, 6))
   kmf = KaplanMeierFitter()
   kmf.fit(df['duration'], df['event'], label='Overall')
   kmf.plot(ax=ax1)
   ax1.set_title('Overall Coupon Survival Curve')
   ax1.set_xlabel('Days since coupon issued')
   ax1.set_ylabel('Survival probability')
   ax1.grid(True)
   fig1.savefig('overall_survival.png', dpi=300, bbox_inches='tight')
   plt.show()  # Show the first plot

   # Plot campaign-specific survival curves
   fig2, ax2 = plt.subplots(figsize=(12, 6))
   for campaign in sorted(df['CAMPAIGN'].unique()):
       mask = df['CAMPAIGN'] == campaign
       if mask.sum() > 0:
           kmf = KaplanMeierFitter()
           kmf.fit(
               df[mask]['duration'],
               df[mask]['event'],
               label=f'Campaign {campaign}'
           )
           kmf.plot(ax=ax2)

   ax2.set_title('Coupon Survival Curves by Campaign')
   ax2.set_xlabel('Days since coupon issued')
   ax2.set_ylabel('Survival probability')
   ax2.grid(True)
   ax2.legend()
   fig2.savefig('campaign_survival.png', dpi=300, bbox_inches='tight')
   plt.show()  # Show the second plot

   # Calculate campaign-specific statistics
   campaign_stats = df.groupby('CAMPAIGN').agg({
       'event': ['count', 'sum', 'mean'],
       'duration': ['mean', 'std', 'min', 'max']
   }).round(4)

   print("\nCampaign-level Statistics:")
   print(campaign_stats)

   return df


def analyze_campaign_types(df):
    """
    Perform log-rank test and plot survival curves comparing TypeB and TypeC campaigns
    """
    # Set pandas display option to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)  # 也添加这个以确保输出不会被截断

    # Define campaign types
    typeC_campaigns = [3, 6, 14, 15, 20, 27]

    # Create campaign type indicator
    df['campaign_type'] = df['CAMPAIGN'].apply(
        lambda x: 'TypeC' if x in typeC_campaigns else 'TypeB'
    )

    # Perform log-rank test
    typeB_mask = df['campaign_type'] == 'TypeB'
    typeC_mask = df['campaign_type'] == 'TypeC'

    results = logrank_test(
        df[typeB_mask]['duration'],
        df[typeC_mask]['duration'],
        df[typeB_mask]['event'],
        df[typeC_mask]['event']
    )

    # Print test results
    print("Log-rank Test Results:")
    print(f"Test statistic: {results.test_statistic:.4f}")
    print(f"P-value: {results.p_value:.4f}")

    # Print basic statistics for each type
    print("\nBasic Statistics by Campaign Type:")
    stats_by_type = df.groupby('campaign_type').agg({
        'coupon_uid': 'count',
        'event': ['sum', 'mean'],
        'duration': ['mean', 'std']
    }).round(4)
    stats_by_type.columns = ['Total Coupons', 'Redemptions', 'Redemption Rate', 'Mean Duration', 'Std Duration']
    print(stats_by_type)

    # Plot survival curves
    plt.figure(figsize=(12, 6))

    # Plot TypeB
    kmf_B = KaplanMeierFitter()
    kmf_B.fit(
        df[typeB_mask]['duration'],
        df[typeB_mask]['event'],
        label=f'TypeB (n={sum(typeB_mask)})'
    )
    kmf_B.plot()

    # Plot TypeC
    kmf_C = KaplanMeierFitter()
    kmf_C.fit(
        df[typeC_mask]['duration'],
        df[typeC_mask]['event'],
        label=f'TypeC (n={sum(typeC_mask)})'
    )
    kmf_C.plot()

    plt.title('Survival Curves by Campaign Type\n' +
              f'Log-rank test p-value: {results.p_value:.4f}')
    plt.xlabel('Days since coupon issued')
    plt.ylabel('Survival probability')
    plt.grid(True)

    # Save and show plot
    plt.savefig('campaign_type_survival.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results, stats_by_type


def plot_hazard_ratios(cph, filename="hazard_ratios.png", title="Significant Hazard Ratios from Cox Model"):
    """
    Plot hazard ratios with confidence intervals for all variables
    """
    # Extract hazard ratios and confidence intervals for all variables
    hazard_ratios = pd.DataFrame({
        'HR': np.exp(cph.params_),
        'lower': np.exp(cph.confidence_intervals_.iloc[:, 0]),
        'upper': np.exp(cph.confidence_intervals_.iloc[:, 1])
    })

    # Add variable names as index
    hazard_ratios.index = cph.params_.index

    # Only plot significant variables (p < 0.05) or main effects
    main_effects = ['is_typeC', 'day_issued_std', 'validity_period_std']
    significant_vars = cph.summary.index[cph.summary['p'] < 0.05].tolist()
    vars_to_plot = list(set(main_effects + significant_vars))

    hazard_ratios_filtered = hazard_ratios.loc[vars_to_plot]

    # Create figure
    plt.figure(figsize=(12, len(vars_to_plot) * 0.5))
    plt.clf()

    # Plot points and lines
    y_positions = range(len(vars_to_plot))

    # Plot confidence intervals
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

    # Add reference line at HR=1
    plt.axvline(x=1, color='k', linestyle='--', alpha=0.5)

    # Customize plot
    plt.yticks(y_positions, vars_to_plot)
    plt.xlabel('Hazard Ratio (95% CI)')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add HR values as text
    for i, (var, row) in enumerate(hazard_ratios_filtered.iterrows()):
        plt.text(
            row['upper'] + 0.01,
            i,
            f"HR={row['HR']:.2f}\n({row['lower']:.2f}-{row['upper']:.2f})",
            verticalalignment='center'
        )

    plt.tight_layout()
    fig = plt.gcf()
    # 保存
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return plt.gcf()


def run_cox_analysis(df):
    """
    Run complete Cox analysis with visualization
    """
    # Prepare data
    cox_data = prepare_cox_data(df)

    # Print basic statistics
    print("\nData Summary:")
    print(f"Total observations: {len(cox_data)}")
    print(f"Events (redemptions): {cox_data['event'].sum()}")
    print(f"Censored: {len(cox_data) - cox_data['event'].sum()}")

    # Fit Cox model
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        cox_data,
        duration_col='duration',
        event_col='event',
        show_progress=True
    )

    # Print model summary
    print("\nCox Model Summary:")
    print(cph.print_summary())

    # Plot hazard ratios
    hr_fig = plot_hazard_ratios(cph)
    hr_fig.savefig('hazard_ratios.png', bbox_inches='tight', dpi=300)
    plt.close(hr_fig)

    # Test proportional hazards assumption
    print("\nTesting Proportional Hazards Assumption:")
    assumptions_fig = cph.check_assumptions(cox_data, show_plots=True)
    plt.tight_layout()
    plt.savefig('proportional_hazards_test.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    # Print feature correlations
    print("\nFeature Correlations:")
    correlations = cox_data.corr().round(4)
    print(correlations)

    return cph, cox_data


def plot_feature_importance(importance_df):
    """
    Plot feature importance using matplotlib with Agg backend
    """
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

    # Add value labels
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
    plt.savefig('rsf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_survival_curves(rsf, y_train, X):
    """
    Plot survival curves using matplotlib with Agg backend
    Modified to handle all features
    """
    plt.figure(figsize=(10, 6))
    plt.clf()

    # Get time points
    times = np.percentile(y_train['time'], np.linspace(0, 100, 100))

    # Create two scenarios: TypeB and TypeC
    # Start with average values for all features
    scenario_base = pd.DataFrame(np.zeros((2, X.shape[1])), columns=X.columns)

    # Set basic feature values (using means for continuous variables)
    scenario_base['day_issued'] = X['day_issued'].mean()
    scenario_base['validity_period'] = X['validity_period'].mean()

    # Set TypeB and TypeC
    scenario_base.iloc[0, scenario_base.columns.get_loc('is_typeC')] = 0  # TypeB
    scenario_base.iloc[1, scenario_base.columns.get_loc('is_typeC')] = 1  # TypeC

    # Set most common values for categorical variables
    for col in X.columns:
        if col not in ['is_typeC', 'day_issued', 'validity_period']:
            most_common = X[col].mode()[0]
            scenario_base[col] = most_common

    # Plot survival curves
    for i, row in scenario_base.iterrows():
        surv_fn = rsf.predict_survival_function(row.values.reshape(1, -1))[0]
        surv_probs = [surv_fn(t) for t in times]
        plt.plot(
            times,
            surv_probs,
            label=f"{'Type C' if row['is_typeC'] else 'Type B'} Campaign",
            linewidth=2
        )

    plt.title('Predicted Survival Curves by Campaign Type\n(with average demographic features)')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival probability')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rsf_survival_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_rsf_analysis(df):
    """
    Run Random Survival Forest analysis with fixed visualization
    """
    # 准备数据
    X, y, scaler = prepare_rsf_data(df)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练模型
    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_leaf=15,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )

    print("\nFitting Random Survival Forest...")
    rsf.fit(X_train, y_train)

    # 计算性能指标
    train_score = rsf.score(X_train, y_train)
    test_score = rsf.score(X_test, y_test)

    print(f"\nModel Performance:")
    print(f"Training C-index: {train_score:.3f}")
    print(f"Testing C-index: {test_score:.3f}")

    # 计算特征重要性
    r = permutation_importance(
        rsf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': r.importances_mean,
        'importance_std': r.importances_std
    }).sort_values('importance_mean', ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

    # 绘制特征重要性图
    plot_feature_importance(importance_df)

    print("\nGenerating survival curves for different scenarios...")
    # 使用完整的特征集来生成生存曲线
    plot_survival_curves(rsf, y_train, X)

    print("\nFeature Summary Statistics:")
    print(X.describe())

    return rsf, importance_df, (X_train, X_test, y_train, y_test)


def save_model_results(cph, rsf, importance_df, file_path='model_results.pdf'):
    """
    将模型结果保存为PDF
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from io import StringIO
    import sys

    # 创建PDF文档
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # 添加标题和日期
    elements.append(Paragraph("Survival Analysis Results", styles['Heading1']))
    elements.append(Spacer(1, 12))

    # Cox模型结果
    elements.append(Paragraph("Cox Proportional Hazards Model Results", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # 创建Cox模型结果表格
    cox_stats = [
        ['Number of observations', str(cph._n_examples)],
        ['Number of events', str(cph.event_observed.sum())],
        ['Concordance', f"{cph.concordance_index_:.3f}"],
        ['Partial AIC', f"{cph.AIC_partial_:.2f}"],
        ['Log-likelihood ratio test', f"{cph.log_likelihood_ratio_test().test_statistic:.2f}"],
        ['Number of parameters', str(len(cph.params_))]
    ]

    # Cox模型系数表格
    coef_df = pd.DataFrame({
        'coef': cph.params_,
        'exp(coef)': np.exp(cph.params_),
        'se(coef)': cph.standard_errors_,
        'z': cph.summary['z'],
        'p': cph.summary['p'],
    }).round(3)

    coef_data = [['Variable', 'Coef', 'Exp(coef)', 'SE', 'z', 'p-value']] + \
                [[idx] + list(row) for idx, row in coef_df.iterrows()]

    # RSF结果
    elements.append(Paragraph("Random Survival Forest Results", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # 特征重要性表格
    importance_data = [['Feature', 'Importance', 'Std']] + \
                      importance_df.values.tolist()

    # 添加表格样式
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

    # 添加所有表格到PDF
    elements.append(create_table_with_style(cox_stats, header=False))
    elements.append(Spacer(1, 12))
    elements.append(create_table_with_style(coef_data))
    elements.append(Spacer(1, 12))
    elements.append(create_table_with_style(importance_data))

    # 生成PDF
    doc.build(elements)

    print(f"Results saved to {file_path}")


def save_detailed_results(df, cox_data, importance_df, file_path='detailed_results.xlsx'):
    """
    保存详细结果到Excel文件，包括数据统计、模型结果和图表
    """
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # 基础数据统计
        df.describe().round(3).to_excel(writer, sheet_name='Data_Summary')

        # Cox模型数据
        cox_data.describe().round(3).to_excel(writer, sheet_name='Cox_Data')

        # 特征重要性
        importance_df.round(3).to_excel(writer, sheet_name='Feature_Importance')

        # 保存图表
        workbook = writer.book

        # 添加图表工作表
        worksheet = workbook.add_worksheet('Charts')

        # 保存特征重要性数据
        importance_data = importance_df.values.tolist()
        for i, row in enumerate(importance_data):
            worksheet.write_row(i, 0, row)

    print(f"Detailed results saved to {file_path}")


## Considering Ordinality in HH Demographics Data

def extract_age_group(s):
    """
    将 'Age GroupX' 形式的字符串转换成数值 X
    若 s 为空或不是字符串，返回 np.nan
    """
    if not isinstance(s, str):
        return np.nan
    # 假设是固定格式，比如 'Age Group4'
    return int(s.replace("Age Group", "").strip())


def transform_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    将具有顺序意义的分类字段转换成可数值化的顺序编码，
    针对 classification_1 ~ classification_5, KID_CATEGORY_DESC, HOMEOWNER_DESC。
    """
    df = df.copy()  # 不修改原 df

    # 1) classification_1: "Age Group4" -> 4, ...
    if 'classification_1' in df.columns:
        df['classification_1_ordinal'] = df['classification_1'].apply(extract_age_group)

    # 2) classification_2: "X", "Y", "Z" -> 1, 2, 3（示例）
    mapping_2 = {'X': 1, 'Y': 2, 'Z': 3}
    if 'classification_2' in df.columns:
        def map_class2(s):
            return mapping_2.get(s, np.nan)
        df['classification_2_ordinal'] = df['classification_2'].apply(map_class2)

    # 3) classification_3: "Level1" ~ "Level12" -> 取中间数字
    def extract_level(s):
        if not isinstance(s, str):
            return np.nan
        return int(s.replace("Level", "").strip())

    if 'classification_3' in df.columns:
        df['classification_3_ordinal'] = df['classification_3'].apply(extract_level)

    # 4) classification_4 的唯一值: ['2' '3' '4' '1' '5+']
    #    假设 "5+" 表示比 5 更高一些，所以映射为 6
    if 'classification_4' in df.columns:
        def parse_class_4(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).strip()
            if val_str.endswith('+'):
                base = val_str.replace('+', '')
                # "5+" -> "5" -> int(5) + 1 -> 6
                try:
                    return int(base) + 1
                except:
                    return np.nan
            else:
                # 常规整数字符
                try:
                    return int(val_str)
                except:
                    return np.nan
        df['classification_4_ordinal'] = df['classification_4'].apply(parse_class_4)

    # 5) classification_5 的唯一值: ['Group5' 'Group4' 'Group3' 'Group2' 'Group1' 'Group6']
    #    说明可以做一个字典映射
    if 'classification_5' in df.columns:
        classification_5_map = {
            'Group1': 1,
            'Group2': 2,
            'Group3': 3,
            'Group4': 4,
            'Group5': 5,
            'Group6': 6
        }
        def parse_class_5(val):
            if pd.isna(val):
                return np.nan
            return classification_5_map.get(val, np.nan)
        df['classification_5_ordinal'] = df['classification_5'].apply(parse_class_5)

    # 6) KID_CATEGORY_DESC 的唯一值: ['None/Unknown' '1' '2' '3+']
    #    假设 "3+" 表示比 3 多一些，就映射成 4；"None/Unknown" -> 0
    if 'KID_CATEGORY_DESC' in df.columns:
        kid_map = {
            'None/Unknown': 0,
            '1': 1,
            '2': 2,
            '3+': 3  # 或者写成 4，看你需求
        }
        df['KID_CATEGORY_DESC_ordinal'] = df['KID_CATEGORY_DESC'].map(kid_map)

    # 7) HOMEOWNER_DESC 可能没明显顺序：例如 "Homeowner"/"Unknown" 等
    #    简化成 Homeowner=1, 否则=0
    if 'HOMEOWNER_DESC' in df.columns:
        df['HOMEOWNER_ordinal'] = df['HOMEOWNER_DESC'].apply(
            lambda x: 1 if x == 'Homeowner' else 0
        )

    return df


def prepare_cox_data_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    与原函数类似，但将分类变量视为有序变量，而非哑变量。
    先行调用 transform_ordinal_features() 来生成 `_ordinal` 字段。
    """
    # ---------------------------
    # 1. 先对 df 做有序转换
    # ---------------------------
    df = transform_ordinal_features(df)

    # Campaign type indicator
    typeC_campaigns = [3, 6, 14, 15, 20, 27]
    df['is_typeC'] = df['CAMPAIGN'].isin(typeC_campaigns).astype(int)

    # Calculate validity period
    df['validity_period'] = df['day_expired'] - df['day_issued']

    # 同样做连续变量的标准化
    df['day_issued_std'] = (df['day_issued'] - df['day_issued'].mean()) / df['day_issued'].std()
    df['validity_period_std'] = (df['validity_period'] - df['validity_period'].mean()) / df['validity_period'].std()

    # ---------------------------
    # 2. 组装 Cox 模型所需的最终变量
    # ---------------------------
    # 这里我们就用 *_ordinal 的列，而不是做 dummies
    # 当然如果 HOMEOWNER_DESC 要保持离散，可以额外加进去
    cox_columns = [
        'duration', 'event', 'is_typeC',
        'day_issued_std', 'validity_period_std',
        # ordinal columns:
        'classification_1_ordinal',
        'classification_2_ordinal',
        'classification_3_ordinal',
        'classification_4_ordinal',
        'classification_5_ordinal',
        'KID_CATEGORY_DESC_ordinal',
        'HOMEOWNER_ordinal'
    ]

    # 只保留 df 已经存在的列（有些项目里字段不一定都有）
    cox_columns_final = [col for col in cox_columns if col in df.columns]

    cox_data_2 = df[cox_columns_final].copy()

    return cox_data_2


def prepare_rsf_data_2(df: pd.DataFrame):
    """
    将分类变量按顺序进行编码后，再进入随机生存森林模型。
    """
    # 1) 先转换有序字段
    df = transform_ordinal_features(df)

    # 2) Create campaign type indicator
    typeC_campaigns = [3, 6, 14, 15, 20, 27]
    df['is_typeC'] = df['CAMPAIGN'].isin(typeC_campaigns).astype(int)

    # 3) Calculate validity period
    df['validity_period'] = df['day_expired'] - df['day_issued']

    # 4) 对需要标准化的字段做标准化（这里以 day_issued, validity_period 为例）
    scaler = StandardScaler()
    features_to_scale = ['day_issued', 'validity_period']
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # 5) 将 ordinal 字段当做数值特征加入 X
    #    同时保留 is_typeC, day_issued, validity_period(已标准化)
    feature_cols = [
        'is_typeC', 'day_issued', 'validity_period',
        'classification_1_ordinal',
        'classification_2_ordinal',
        'classification_3_ordinal',
        'classification_4_ordinal',
        'classification_5_ordinal',
        'KID_CATEGORY_DESC_ordinal',
        'HOMEOWNER_ordinal'
    ]

    # 只保留 df 中有的列
    feature_cols_final = [col for col in feature_cols if col in df_scaled.columns]
    X_2 = df_scaled[feature_cols_final].copy()

    # 6) Survival data structure
    y_2 = np.zeros(len(df), dtype=[('event', bool), ('time', float)])
    y_2['event'] = df['event'].astype(bool)
    y_2['time'] = df['duration']

    return X_2, y_2, scaler

def plot_feature_importance_v2(importance_df):
    """
    Plot feature importance for RSF (v2) using matplotlib with Agg backend
    """
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
    plt.title('Feature Importance in Random Survival Forest (v2)')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.grid(True, alpha=0.3)

    # Add value labels
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
    plt.savefig('rsf_feature_importance_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature importance figure saved as rsf_feature_importance_v2.png")


def plot_survival_curves_v2(rsf, y_train, X):
    """
    Plot survival curves for RSF (v2) using matplotlib with Agg backend
    """
    plt.figure(figsize=(10, 6))
    plt.clf()

    # Get time points
    times = np.percentile(y_train['time'], np.linspace(0, 100, 100))

    # Create two scenarios: TypeB and TypeC
    # Start with average values for all features
    scenario_base = pd.DataFrame(np.zeros((2, X.shape[1])), columns=X.columns)

    # Set basic feature values (using means for continuous variables)
    scenario_base['day_issued'] = X['day_issued'].mean()
    scenario_base['validity_period'] = X['validity_period'].mean()

    # Set TypeB and TypeC
    scenario_base.iloc[0, scenario_base.columns.get_loc('is_typeC')] = 0  # TypeB
    scenario_base.iloc[1, scenario_base.columns.get_loc('is_typeC')] = 1  # TypeC

    # Set most common values for categorical variables
    for col in X.columns:
        if col not in ['is_typeC', 'day_issued', 'validity_period']:
            most_common = X[col].mode()[0]
            scenario_base[col] = most_common

    # Plot survival curves
    for i, row in scenario_base.iterrows():
        surv_fn = rsf.predict_survival_function(row.values.reshape(1, -1))[0]
        surv_probs = [surv_fn(t) for t in times]
        plt.plot(
            times,
            surv_probs,
            label=f"{'Type C' if row['is_typeC'] else 'Type B'} Campaign",
            linewidth=2
        )

    plt.title('Predicted Survival Curves by Campaign Type (v2)\n(with average demographic features)')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival probability')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rsf_survival_curves_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Survival curves figure saved as rsf_survival_curves_v2.png")


# MAIN

## clean data
df = create_complete_dataset()

## KM kaplan-mierer
# df_km = basic_survival_analysis(df)

## Log Rank
df = prepare_basic_survival_data(df)


# df_lr, stats_by_type = analyze_campaign_types(df)

## Cox
cph, cox_data = run_cox_analysis(df)

## RSF
rsf_model, importance_df, split_data = run_rsf_analysis(df)

## Save model result
# save_model_results(cph, rsf_model, importance_df)
# 运行保存功能
save_model_results(cph, rsf_model, importance_df, 'survival_analysis_results.pdf')
save_detailed_results(df, cox_data, importance_df, 'survival_analysis_detailed.xlsx')

# v2：ordinal

df_2 = transform_ordinal_features(df)

# Cox：
cox_data_2 = prepare_cox_data_2(df)
cph_2 = CoxPHFitter()
cox_data_2_no_na = cox_data_2.dropna(axis=0)

cph_2.fit(
    cox_data_2_no_na,
    duration_col='duration',
    event_col='event',
    show_progress=True
)

print(cph_2.summary)
plot_hazard_ratios(
    cph_2,
    filename="hazard_ratios_v2.png",
    title="Significant Hazard Ratios from Cox Model (v2)"
)

# RSF：
X_2, y_2, scaler_2 = prepare_rsf_data_2(df)
#   注意：X_2 是 DataFrame，可以直接 isna().any(axis=1)
na_mask = X_2.isna().any(axis=1)
print("Number of rows with NaN:", na_mask.sum())
#   取反：只保留没NaN的行
X_2_no_na = X_2[~na_mask].copy()
y_2_no_na = y_2[~na_mask]
# 然后再 train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_2_no_na,
    y_2_no_na,
    test_size=0.2,
    random_state=42
)

# 训练 RSF
rsf_2 = RandomSurvivalForest(
    n_estimators=100,
    min_samples_leaf=15,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rsf_2.fit(X2_train, y2_train)
# importance_df_v2 是计算后的特征重要性 DataFrame

# 5) 评估
print("New RSF C-index (train):", rsf_2.score(X2_train, y2_train))
print("New RSF C-index (test):", rsf_2.score(X2_test, y2_test))

