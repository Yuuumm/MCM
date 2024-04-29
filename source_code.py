//part of the core source code
def calculate_metrics(df):
    # process useful data for the logic model
    df.sort_values(by=['match_id', 'point_no'], inplace=True)

    df['p1_serve_won_total'] = df[(df['point_victor'] == 1) & (df['server'] == 1)].groupby('match_id')[
        'server'].cumcount()
    df['p1_serve_total'] = df[df['server'] == 1].groupby('match_id')['server'].cumcount() + 1
    df['p1_unserve_won_total'] = df[(df['point_victor'] == 1) & (df['server'] == 2)].groupby('match_id')[
        'server'].cumcount()
    df['p1_unserve_total'] = df[df['server'] == 2].groupby('match_id')['server'].cumcount() + 1
    df['p1_break_pt_won_total'] = df.groupby('match_id')['p1_break_pt_won'].cumsum()
    df['p1_break_pt_total'] = df.groupby('match_id')['p1_break_pt'].cumsum()

    # similar data processing for the player2:
    # df['p2_serve_won_total'] = df[(df['point_victor'] == 2) & (df['server'] == 2)].groupby('match_id')[......

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def plot_match_momentum(df, selected_match_id):
    df['diff_momentum'] = df['p1_momentum'] - df['p2_momentum']
    
    # draw change of momentum in Point
    plt.figure(figsize=(40, 4))
    plt.plot(df['point_no'], df['diff_momentum'], label='Momentum Difference',
             color='blue', linestyle='-', alpha=0.5, linewidth=1)
    break_point_won_points_p1 = df[df['p1_break_pt_won'] == 1]['point_no']
    plt.scatter(break_point_won_points_p1, df.loc[df['point_no'].isin(break_point_won_points_p1), 'diff_momentum'],
                color='red', marker='o', label='Player 1 Break Point Won')
    break_point_won_points_p2 = df[df['p2_break_pt_won'] == 1]['point_no']
    plt.scatter(break_point_won_points_p2, df.loc[df['point_no'].isin(break_point_won_points_p2), 'diff_momentum'],
                color='yellow', marker='o', label='Player 2 Break Point Won')

    plt.xlabel('Point Number')
    plt.ylabel('Momentum Difference')
    plt.title(f'{selected_match_id} - Momentum Visualization')
    plt.legend()
    plt.show()

# encoding Non-Numeric Variables with Unique Hot Coding
def encode_categorical_features(df):
    encoder = OneHotEncoder(drop='first')
    categorical_columns = ['serve_width', 'serve_depth', 'return_depth', 'winner_shot_type']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df_encoded

# Calculate the standard deviation of momentum swings over a rolling window
def quantify_momentum_swings(df):
    df['momentum_std'] = df['p1_momentum'].rolling(window=5, min_periods=1).std()

# Visualize a decision tree using matplotlib
def visualize_decision_tree(tree, feature_names, class_names):
    plt.figure(figsize=(12, 8))
    plot_tree(tree, feature_names=feature_names, filled=True, rounded=True, fontsize=8,
              class_names=class_names, max_depth=3)
    plt.title("Random Forest Training - Decision Process Visualization")
    plt.show()

# Correlation between Momentum Difference and Game Victor
def diff_momentum_and_game(df):
    # data processing...
    # grouped = df.groupby(['match_id', 'set_no', 'game_no'], as_index=False).agg({...})
    grouped['diff_momentum'] = grouped['p1_momentum'] - grouped['p2_momentum']
    grouped['game_p1'] = grouped.groupby(['match_id'], as_index=False)['game_victor'].transform(
        lambda x: (x == 1).cumsum())
    correlation1 = grouped['diff_momentum'].corr(grouped['game_victor'])  # 0.7472
    print(f'Correlation between Momentum Difference and Game Victor : {correlation1}')

    temp = grouped.loc[:, ['diff_momentum', 'game_victor']]
    X = grouped['diff_momentum'].to_numpy().reshape((-1, 1))
    y = grouped['game_victor'].to_numpy()
    # train a classifier...
    # draw decision regions...
    # draw violin plot...
    return grouped

# prediction of potential fluctuations using a random forest model
def predict_momentum_fluctuations(df):
    """   Random Forest Model  """
    # data processing...
    # transform dataset to get 'p1_winner_unf_err_total' and so on...
    # df['p1_winner_unf_err_total'] = df[(df['p1_winner'] == 1) | (df['p1_unf_err'] == '1')].groupby......

    feature_columns = ['p1_serve_won_percentage', 'p1_break_pt_won_percentage', 'p1_winner_unf_err_percentage',
                       'p1_rally_count_won_percentage']

    X = df[feature_columns]
    y = df['momentum_std']
    imputer = SimpleImputer(strategy='mean')
    y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
    train_size = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # train Random Forest Regression Models
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    tree = model.estimators_[0]
    visualize_decision_tree(tree, feature_names=feature_columns, class_names=['No Fluctuation', 'Fluctuation'])
    mse = mean_squared_error(y_test, predictions)
    print(f"MSE: {mse}")
    plot_residuals(y_test, predictions)
    
    # get feature importance and print
    feature_importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_dict = dict(zip(feature_names, feature_importances))
    sorted_feature_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("Feature Importance Scores:")

# prediction of momentum using a logistic regression model
def train_and_evaluate_model(df, selected_match_id):
    """   Logistic Regression Model  """
    df = calculate_serve_total(df)
    df = calculate_metrics(df)
    selected_match_df = df[df['match_id'] == selected_match_id].copy()
    
    # For Player 1 (p1)
    p1_feature_columns = ['serve_win_percentage', 'return_win_percentage', 'break_point_win_percentage', 'server']
    X_p1 = selected_match_df[p1_feature_columns]
    y_p1 = selected_match_df['point_victor'] == 1

    X_train_p1, X_test_p1, y_train_p1, y_test_p1 = train_test_split(X_p1, y_p1, test_size=0.2, random_state=42)

    scaler_p1 = StandardScaler()
    X_train_scaled_p1 = scaler_p1.fit_transform(X_train_p1)
    X_test_scaled_p1 = scaler_p1.transform(X_test_p1)

    model_p1 = LogisticRegression()
    model_p1.fit(X_train_scaled_p1, y_train_p1)

    predictions_p1 = model_p1.predict(X_test_scaled_p1)
    accuracy_p1 = accuracy_score(y_test_p1, predictions_p1)
    print(f"Player1's accuracy: {accuracy_p1}")

    selected_match_df['predicted_probabilities'] = model_p1.predict_proba(selected_match_df[p1_feature_columns])[:, 1]

    # For Player 2 (p2)... similar to Player 1's process
    selected_match_df['p1_momentum'] = selected_match_df['predicted_probabilities'].diff().fillna(0)
    selected_match_df['p2_momentum'] = selected_match_df['p2_predicted_probabilities'].diff().fillna(0)

    n_features = len(model_p1.coef_[0])
    formula1 = f"Probability(Y=1) = 1 / (1 + exp(-({model_p1.intercept_[0]} + {' + '.join([f'{model_p1.coef_[0][i]}*X{i + 1}' for i in range(n_features)])})))"
    print(f"model mathematical formula：{formula1}")
    n_features = len(model_p2.coef_[0])
    formula2 = f"Probability(Y=1) = 1 / (1 + exp(-({model_p2.intercept_[0]} + {' + '.join([f'{model_p2.coef_[0][i]}*X{i + 1}' for i in range(n_features)])})))"
    print(f"model mathematical formula：{formula2}")

    diff_momentum_and_game(selected_match_df)
    plot_match_momentum(selected_match_df, selected_match_id)

    # quantify and forecast momentum's volatility standard deviation
    selected_match_df['momentum_std'] = selected_match_df['p1_momentum'].rolling(window=5, min_periods=1).std()
    predict_momentum_fluctuations(selected_match_df)
