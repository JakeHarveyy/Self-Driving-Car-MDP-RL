states = {'UT_H', 'UT_L', 'DT_H', 'DT_L', 'C_H', 'C_L', 'PS_H', 'PS_L', 'PD_H', 'PD_L'}
actions = {'Buy', 'Hold', 'Sell'}

#R(s,a)
# rows represent current state, columns represent actions
# Columns :'Buy', 'Hold', 'Sell']
rewards_matrix = [
    # Buy, Hold, Sell
    [ -50,   25,   75],  # 0: UT_H (Upward Trend + High Volume)
    [ -25,   25,   50],  # 1: UT_L (Upward Trend + Low Volume)
    [  50,  -25,  -50],  # 2: DT_H (Downward Trend + High Volume)
    [  25,  -25,  -25],  # 3: DT_L (Downward Trend + Low Volume)
    [  0,   0,   0],  # 4: C_H (Consolidate + High Volume)
    [  0,   0,   0],  # 5: C_L (Consolidate + Low Volume)
    [ -75,   0,   100],  # 6: PS_H (Price Spike + High Volume)
    [ -50,   0,   75],  # 7: PS_L (Price Spike + Low Volume)
    [  75,  -50,  -75],  # 8: PD_H (Price Drop + High Volume)
    [  50,  -25,  -50]   # 9: PD_L (Price Drop + Low Volume)
]


#R(s'|s a)
# Rows represent current state, columns represent actions, and the inner list represents the next states probabilities
transition_matrix = [
    # S1: UT_H - Upward Trend + High Volume
    [
        [0.40, 0.10, 0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.00, 0.05],  # Buy
        [0.50, 0.10, 0.05, 0.00, 0.15, 0.10, 0.05, 0.00, 0.00, 0.05],  # Hold
        [0.30, 0.10, 0.10, 0.05, 0.20, 0.10, 0.05, 0.00, 0.05, 0.05]   # Sell
    ],
    # S2: UT_L - Upward Trend + Low Volume
    [
        [0.20, 0.30, 0.10, 0.10, 0.15, 0.10, 0.00, 0.00, 0.00, 0.05],  # Buy
        [0.25, 0.40, 0.05, 0.05, 0.15, 0.10, 0.00, 0.00, 0.00, 0.00],  # Hold
        [0.10, 0.20, 0.15, 0.10, 0.20, 0.15, 0.00, 0.00, 0.05, 0.05]   # Sell
    ],
    # S3: DT_H - Downward Trend + High Volume
    [
        [0.10, 0.05, 0.30, 0.05, 0.10, 0.10, 0.05, 0.00, 0.20, 0.05],  # Buy
        [0.05, 0.00, 0.50, 0.10, 0.15, 0.10, 0.00, 0.00, 0.05, 0.05],  # Hold
        [0.00, 0.00, 0.40, 0.20, 0.15, 0.10, 0.00, 0.00, 0.10, 0.05]   # Sell
    ],
    # S4: DT_L - Downward Trend + Low Volume
    [
        [0.15, 0.10, 0.10, 0.30, 0.10, 0.10, 0.00, 0.00, 0.10, 0.05],  # Buy
        [0.05, 0.05, 0.10, 0.40, 0.15, 0.15, 0.00, 0.00, 0.05, 0.05],  # Hold
        [0.00, 0.00, 0.05, 0.50, 0.20, 0.15, 0.00, 0.00, 0.05, 0.05]   # Sell
    ],
    # S5: C_H - Consolidate + High Volume
    [
        [0.15, 0.05, 0.05, 0.05, 0.30, 0.10, 0.10, 0.05, 0.10, 0.05],  # Buy
        [0.10, 0.05, 0.05, 0.05, 0.40, 0.15, 0.10, 0.05, 0.00, 0.05],  # Hold
        [0.05, 0.05, 0.10, 0.05, 0.35, 0.10, 0.05, 0.05, 0.10, 0.10]   # Sell
    ],
    # S6: C_L - Consolidate + Low Volume
    [
        [0.10, 0.05, 0.05, 0.05, 0.15, 0.35, 0.10, 0.10, 0.00, 0.05],  # Buy
        [0.05, 0.05, 0.05, 0.05, 0.15, 0.50, 0.05, 0.05, 0.00, 0.05],  # Hold
        [0.05, 0.05, 0.05, 0.05, 0.15, 0.45, 0.05, 0.05, 0.05, 0.05]   # Sell
    ],
    # S7: PS_H - Price Spike + High Volume
    [
        [0.10, 0.05, 0.20, 0.10, 0.10, 0.10, 0.10, 0.05, 0.10, 0.10],  # Buy
        [0.05, 0.05, 0.25, 0.10, 0.10, 0.10, 0.05, 0.05, 0.15, 0.10],  # Hold
        [0.05, 0.05, 0.20, 0.15, 0.15, 0.10, 0.00, 0.00, 0.20, 0.10]   # Sell (next few days might see a decrease)
    ],
    # S8: PS_L - Price Spike + Low Volume
    [
        [0.05, 0.10, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.10],  # Buy
        [0.05, 0.10, 0.10, 0.20, 0.15, 0.15, 0.00, 0.05, 0.10, 0.10],  # Hold
        [0.00, 0.05, 0.20, 0.20, 0.20, 0.15, 0.00, 0.00, 0.10, 0.10]   # Sell
    ],
    # S9: PD_H - Price Drop + High Volume
    [
        [0.20, 0.10, 0.05, 0.00, 0.10, 0.10, 0.10, 0.05, 0.20, 0.10],  # Buy (anticipating price bounce back)
        [0.10, 0.05, 0.10, 0.05, 0.15, 0.10, 0.05, 0.00, 0.30, 0.20],  # Hold
        [0.05, 0.00, 0.15, 0.05, 0.15, 0.10, 0.00, 0.00, 0.40, 0.10]   # Sell
    ],
    # S10: PD_L - Price Drop + Low Volume
    [
        [0.15, 0.10, 0.05, 0.05, 0.10, 0.10, 0.05, 0.05, 0.15, 0.20],  # Buy
        [0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.00, 0.00, 0.10, 0.30],  # Hold
        [0.00, 0.00, 0.10, 0.15, 0.20, 0.15, 0.00, 0.00, 0.15, 0.25]   # Sell
    ]
]