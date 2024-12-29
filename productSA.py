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


import pandas as pd
import numpy as np


def attach_RFM_features(df: pd.DataFrame, df_transaction_path: str) -> pd.DataFrame:
    """
    为 coupon_uid 附上 RFM 特征。
    df_transaction 中可以由 BASKET_ID 唯一确定单次购买行为，再用 groupby sum 得到每次购买金额。
    然后将每天购买金额展开为"日历式"矩阵，再做向量化计算 R、F、M。

    计算完成后，再基于 coupon_uid 的 day_issued 和 day_expired，以及所在的 household_key，
    在 R、F、M 矩阵中取对应的时间窗，做平均，得到该 coupon_uid 的 RFM 均值。

    :param df: 包含 coupon_uid 的 DataFrame，至少需要列 ['household_key','day_issued','day_expired',...]
    :param df_transaction_path: 包含 [household_key, DAY, BASKET_ID, SALES_VALUE, ...] 的 CSV 路径
    :return: 在 df 基础上新增 RFM 均值列（或字典）的 DataFrame
    """
    # 1) 读入原始交易数据
    df_transaction = pd.read_csv(df_transaction_path)

    # 2) 先按 (household_key, DAY) 聚合得到“每天总消费额”
    #    （如果你的表里已经聚合好了，也可以直接用）
    #    注意：如果原本有 BASKET_ID，需要先 groupby sum 再按照 (household_key, DAY) 再次 sum 即可。
    df_daily = (
        df_transaction
        .groupby(['household_key', 'DAY'], as_index=False)['SALES_VALUE']
        .sum()
    )

    # 3) 将每天消费额“展开”为矩阵：行 = DAY，列 = household_key
    #    单元格 = 当天消费额（无交易则填 0）
    #    为了不丢失空白天，最好把 index 补充到 [min_day ... max_day]
    all_days = range(df_daily['DAY'].min(), df_daily['DAY'].max() + 1)
    df_daily_pivot = (
        df_daily
        .pivot(index='DAY', columns='household_key', values='SALES_VALUE')
        .reindex(index=all_days, fill_value=0)  # 确保行索引是完整的天数范围
        .fillna(0)  # 万一还有缺失值就补 0
    )

    # 现在 df_daily_pivot 形状大致是 (天数 × household 数量)，
    # 其中的值是当天的销售额。

    # 4) 计算 Frequency、Monetary、Recency

    # 4.1) Frequency: 每天是否有交易 => True/False => 转换为 1/0 => 再做 cumsum
    freq_table = (df_daily_pivot > 0).cumsum(axis=0)

    # 4.2) Monetary: 对销售额做 cumsum
    monetary_table = df_daily_pivot.cumsum(axis=0)

    # 4.3) Recency: “距离最近一次购买的天数”
    #     我们自定义一个函数 compute_recency，给定某个 household 的销售序列，逐日计算距离上次购买的间隔。
    def compute_recency(sales_array: np.ndarray) -> np.ndarray:
        """
        sales_array: shape=(N,)，表示该 household 连续 N 天的销售额
        return: shape=(N,)，第 i 天的 recency（距离上次购买多少天）
        """
        recency = np.zeros_like(sales_array, dtype=int)
        last_purchase_day = -1
        for i in range(len(sales_array)):
            if sales_array[i] > 0:
                # 当天有购买
                recency[i] = 0
                last_purchase_day = i
            else:
                # 当天没有购买
                if last_purchase_day == -1:
                    # 历史上还没买过
                    recency[i] = i + 1
                else:
                    recency[i] = i - last_purchase_day
        return recency

    # 对每个 household_key（即列）逐列计算 recency
    recency_table = df_daily_pivot.apply(lambda col: compute_recency(col.values), axis=0, result_type='expand')
    # 注意：上面用 apply 时，需要设置 result_type='expand'，才能把返回的 np.array 展开为 DataFrame

    # 现在 freq_table, monetary_table, recency_table 都有相同的索引和列，分别表示每天的 F, M, R。

    # 5) 根据 coupon_uid 的 day_issued, day_expired, household_key，
    #    去这 3 张表里把对应区间的 (R,F,M) 取出来做平均。
    #    举例：对 df 的每一行 coupon，我们去定位它的 household_key 列，切出 recency_table/freq_table/monetary_table
    #    在 [day_issued, day_expired] 之间做平均，然后存入 df。
    #    你也可以把 R、F、M 合并到一个字典再存入 df，下面给出示例做法：

    r_values = []
    f_values = []
    m_values = []
    for idx, row in df.iterrows():
        # 可能要保证这几列都存在
        hhk = row['household_key']
        d0 = row['day_issued']
        d1 = row['day_expired']

        # 如果 d0, d1 不在我们表的索引范围，需做边界保护
        d0 = max(d0, df_daily_pivot.index.min())
        d1 = min(d1, df_daily_pivot.index.max())

        # 切片 [d0, d1]，再取对应列 hhk
        # 注意: 索引是 int 类型，使用 loc 切片
        r_slice = recency_table.loc[d0:d1, hhk]
        f_slice = freq_table.loc[d0:d1, hhk]
        m_slice = monetary_table.loc[d0:d1, hhk]

        # 求平均
        r_mean = r_slice.mean()
        f_mean = f_slice.mean()
        m_mean = m_slice.mean()

        r_values.append(r_mean)
        f_values.append(f_mean)
        m_values.append(m_mean)

    # 把结果加回原 DataFrame
    df['R_mean'] = r_values
    df['F_mean'] = f_values
    df['M_mean'] = m_values

    return df


if __name__ == '__main__':
    df = create_complete_dataset()
    # 基本处理
    df = basic_process(df)
    # df = attach_product_info(df, 'data/coupon.csv', 'data/product.csv')

    # 分析不同的产品类型
    # df, examine = analyze_product_types(df)

    df = attach_RFM_features(df, 'data/transaction_data.csv')
    input('Press Enter to exit...')

    print('Done!')
