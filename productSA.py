import pandas as pd
from BasicSA import create_complete_dataset, prepare_basic_survival_data


def basic_process(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_basic_survival_data(df)
    # Calculate campaign-specific statistics
    return df


def attach_product_info(df: pd.DataFrame, df_coupon_path: str, df_product_path: str) -> pd.DataFrame:
    """
        为coupon_uid附上department和brand两个属性，并统计department和brand的计数。
        :param df: 包含coupon_uid的DataFrame
        :param df_coupon_path: 包含COUPON_UPC和PRODUCT_ID的CSV路径
        :param df_product_path: 包含PRODUCT_ID、DEPARTMENT和BRAND的CSV路径
        :return: 附上department和brand两个属性的DataFrame
    """
    # 读取CSV文件
    df_coupon = pd.read_csv(df_coupon_path)
    df_product = pd.read_csv(df_product_path)

    # 生成product_bundle，每个UPC对应多个PRODUCT
    product_bundle = df_coupon.groupby('COUPON_UPC')['PRODUCT_ID'].apply(list).reset_index()
    product_bundle['PRODUCT_COUNT'] = product_bundle['PRODUCT_ID'].apply(len)

    # 将df_coupon与df_product合并
    df_coupon = pd.merge(df_coupon, df_product[['PRODUCT_ID', 'DEPARTMENT', 'BRAND']],
                         on='PRODUCT_ID', how='left')

    # 统计每个COUPON_UPC的department和brand计数
    department_counts = df_coupon.groupby(['COUPON_UPC', 'DEPARTMENT']).size().unstack(fill_value=0).reset_index()
    brand_counts = df_coupon.groupby(['COUPON_UPC', 'BRAND']).size().unstack(fill_value=0).reset_index()

    # 合并计数结果到df_coupon
    product_bundle = pd.merge(product_bundle, department_counts, on='COUPON_UPC', how='left')
    product_bundle = pd.merge(product_bundle, brand_counts, on='COUPON_UPC', how='left',
                              suffixes=('_dept_count', '_brand_count'))

    # 将product_bundle与原始df合并
    df = pd.merge(df, product_bundle, on='COUPON_UPC', how='left')

    # 找到每个coupon_uid数值最大的department，将名称赋值到新列，这一列命名为product_type
    df['product_type'] = df[product_bundle.columns[3:][:-2]].idxmax(axis=1)

    return df


def analyze_product_types(df):
    """
    Perform survival analysis and plot survival curves comparing different product types
    """
    import pandas as pd
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    # Set pandas display option to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)  # Ensures output is not truncated

    # Ensure 'product_type' column exists
    if 'product_type' not in df.columns:
        raise ValueError("The dataframe must contain a 'product_type' column.")

    # Perform basic statistics grouped by product_type
    stats_by_product = df.groupby('product_type').agg({
        'coupon_uid': 'count',
        'event': ['sum', 'mean'],
        'duration': ['mean', 'std']
    }).round(4)
    stats_by_product.columns = ['Total Coupons', 'Redemptions', 'Redemption Rate', 'Mean Duration', 'Std Duration']

    print("\nBasic Statistics by Product Type:")
    print(stats_by_product)

    # Initialize plot for survival curves
    plt.figure(figsize=(14, 8))

    # Colors for differentiation
    colors = plt.cm.tab20.colors

    # Iterate through unique product types and plot their survival curves
    unique_product_types = sorted(df['product_type'].unique())
    kmf = KaplanMeierFitter()

    for i, product in enumerate(unique_product_types):
        mask = df['product_type'] == product
        if mask.sum() > 0:
            kmf.fit(df[mask]['duration'], df[mask]['event'], label=f'{product} (n={mask.sum()})')
            kmf.plot(ci_show=False, color=colors[i % len(colors)])

    plt.title('Survival Curves by Product Type')
    plt.xlabel('Days since coupon issued')
    plt.ylabel('Survival probability')
    plt.grid(True)

    # Save and show plot
    plt.savefig('product_type_survival.png', dpi=300, bbox_inches='tight')
    plt.legend()

    # Perform log-rank tests between all pairs of product types
    results_list = []
    for i in range(len(unique_product_types)):
        for j in range(i + 1, len(unique_product_types)):
            type1 = unique_product_types[i]
            type2 = unique_product_types[j]
            mask1 = df['product_type'] == type1
            mask2 = df['product_type'] == type2

            if mask1.sum() > 0 and mask2.sum() > 0:
                result = logrank_test(
                    df[mask1]['duration'],
                    df[mask2]['duration'],
                    df[mask1]['event'],
                    df[mask2]['event']
                )
                results_list.append({
                    'Product Type 1': type1,
                    'Product Type 2': type2,
                    'Test Statistic': result.test_statistic,
                    'P-value': result.p_value
                })

    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results_list)
    results_df['P-value'] = results_df['P-value'].apply(lambda x: f'{x:.6f}')
    print("\nLog-rank Test Results:")
    print(results_df)
    plt.show()

    return stats_by_product, results_df





if __name__ == '__main__':
    df = create_complete_dataset()
    # 基本处理
    df = basic_process(df)
    df = attach_product_info(df, 'data/coupon.csv', 'data/product.csv')

    # 分析不同的产品类型
    df, examine = analyze_product_types(df)
    input('Press Enter to exit...')

    print('Done!')
