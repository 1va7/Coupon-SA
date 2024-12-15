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


def prepare_basic_survival_data(df):
    """
    Prepare the basic survival analysis dataset
    """
    # Calculate duration and event indicator
    df['duration'] = df.apply(
        lambda x: x['day_redeemed'] - x['day_issued'] if pd.notnull(x['day_redeemed'])
        else x['day_expired'] - x['day_issued'],
        axis=1
    )
    df['event'] = df['day_redeemed'].notna().astype(int)

    return df


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


def prepare_cox_data(df):
    """
    Prepare data for Cox proportional hazards model
    """
    # Create campaign type indicator (0 for TypeB, 1 for TypeC)
    typeC_campaigns = [3, 6, 14, 15, 20, 27]
    df['is_typeC'] = df['CAMPAIGN'].isin(typeC_campaigns).astype(int)

    # Calculate validity period
    df['validity_period'] = df['day_expired'] - df['day_issued']

    # Standardize continuous variables
    df['day_issued_std'] = (df['day_issued'] - df['day_issued'].mean()) / df['day_issued'].std()
    df['validity_period_std'] = (df['validity_period'] - df['validity_period'].mean()) / df['validity_period'].std()

    # Prepare final dataset for Cox model
    cox_data = df[[
        'duration',
        'event',
        'is_typeC',
        'day_issued_std',
        'validity_period_std'
    ]].copy()

    return cox_data


def plot_hazard_ratios(cph):
    """
    Plot hazard ratios with confidence intervals
    """
    # Extract hazard ratios and confidence intervals
    hazard_ratios = pd.DataFrame({
        'HR': np.exp(cph.params_),
        'lower': np.exp(cph.confidence_intervals_.iloc[:, 0]),
        'upper': np.exp(cph.confidence_intervals_.iloc[:, 1]),
        'variable': ['Campaign Type', 'Issue Day', 'Validity Period']
    })

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot points and lines
    for i, row in hazard_ratios.iterrows():
        plt.plot([row['lower'], row['upper']], [i, i], 'b-', linewidth=2)
        plt.plot(row['HR'], i, 'bo', markersize=8)

        # Add text
        plt.text(
            row['upper'] + 0.01,
            i,
            f"HR={row['HR']:.2f} ({row['lower']:.2f}-{row['upper']:.2f})",
            verticalalignment='center'
        )

    # Customize plot
    plt.axvline(x=1, color='k', linestyle='--', alpha=0.5)
    plt.yticks(range(len(hazard_ratios)), hazard_ratios['variable'])
    plt.xlabel('Hazard Ratio (95% CI)')
    plt.title('Hazard Ratios from Cox Model')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
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


def prepare_rsf_data(df):
    """
    Prepare data for Random Survival Forest analysis
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

    # Prepare feature matrix
    X = df_scaled[['is_typeC', 'day_issued', 'validity_period']]

    # Prepare survival data structure
    y = np.zeros(len(df), dtype=[('event', bool), ('time', float)])
    y['event'] = df['event'].astype(bool)
    y['time'] = df['duration']

    return X, y, scaler


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


def plot_survival_curves(rsf, y_train, scenarios):
    """
    Plot survival curves using matplotlib with Agg backend
    """
    plt.figure(figsize=(10, 6))
    plt.clf()

    # Get time points
    times = np.percentile(y_train['time'], np.linspace(0, 100, 100))

    for i, row in scenarios.iterrows():
        surv_fn = rsf.predict_survival_function(row.values.reshape(1, -1))[0]
        surv_probs = [surv_fn(t) for t in times]
        plt.plot(
            times,
            surv_probs,
            label=f"{'Type C' if row['is_typeC'] else 'Type B'} Campaign",
            linewidth=2
        )

    plt.title('Predicted Survival Curves by Campaign Type')
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

    # 创建场景预测
    scenarios = pd.DataFrame({
        'is_typeC': [0, 1],
        'day_issued': [0, 0],
        'validity_period': [0, 0]
    })

    print("\nGenerating survival curves for different scenarios...")
    plot_survival_curves(rsf, y_train, scenarios)

    print("\nFeature Summary Statistics:")
    print(X.describe())

    return rsf, importance_df, (X_train, X_test, y_train, y_test)


# MAIN

## clean data
df = create_complete_dataset()

## KM
df_km = basic_survival_analysis(df)

## Log Rank
df = prepare_basic_survival_data(df)
df_lr, stats_by_type = analyze_campaign_types(df)

## Cox
cph, cox_data = run_cox_analysis(df)

## RSF
rsf_model, importance_df, split_data = run_rsf_analysis(df)
