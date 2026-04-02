# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

def Get_Games_TG_gt_2(file_path,season=0):
    """
    Processes a specific season's worth of games (380 rows) from an Excel file,
    and returns a DataFrame showing the number of games each team played 
    where the 'TG' (Total Goals) value is greater than 2.

    Parameters:
    - file_path (str): Path to the Excel file.
    - season (int): Zero-based index indicating the season (chunk of 380 rows) to analyze.

    Returns:
    - pd.DataFrame: DataFrame with columns ['Team', 'Games_TG_gt_2'], sorted by count descending.
    """
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Calculate start and end index for the specified season
    start_idx = season * 380
    end_idx = start_idx + 380

    # Slice the DataFrame to get only the rows for the selected season
    df_season = df.iloc[start_idx:end_idx]

    # Filter games where TG > 2
    df_filtered = df_season[df_season['TG'] > 2]

    # Count appearances of each team
    team_game_counts = {}

    for _, row in df_filtered.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        team_game_counts[home_team] = team_game_counts.get(home_team, 0) + 1
        team_game_counts[away_team] = team_game_counts.get(away_team, 0) + 1

    # Create a result DataFrame
    result_df = pd.DataFrame(list(team_game_counts.items()), columns=['Team', 'Games_TG_gt_2'])

    # Sort by number of games
    result_df = result_df.sort_values(by='Games_TG_gt_2', ascending=False)

    return result_df

##Weights
def compute_weights():
    """
    Calculates and returns normalized feature weights for use in the model.

    Returns:
    - dict: A dictionary of normalized weights keyed by 'w1' to 'w5'.
    """
    # Raw proportions based on domain-specific scaling
    F_proportions = {
        'w1': 0.05 / 19.73,
        'w2': 0.13 / 0.496,
        'w3': 0.17 / 2.733,
        'w4': 0.43 / 2.785,
        'w5': 0.22 / 0.563
    }

    # Normalize the weights so they sum to 1
    total = sum(F_proportions.values())
    weights = {key: value / total for key, value in F_proportions.items()}

    return weights
weightings=compute_weights()

##

def calculate_x1x2x3(factor_df, seasons, teams, weightings):
    '''

    Parameters
    ----------
    factor_df : Pandas Data Frame
        A data frame consisting of the following columns:
        'Team','TGI', 'TG>2', 'G/TGI', 'TGI per Game' and 'Season'.
    seasons : List
        Defined within the function. A list of seasons iterated
        through to calculate X1->X3 appropriately.
    teams : List
        Defined within the function. A list of teams iterated through
        so the function can assign each factor calculated to each team
        present in the data frame correctly.
    weightings : Dict
        Dictionary containing weights for each component of the factor
        and its corresponding value.

    Returns
    -------
    results : Pandas Data Frame
        Data frame containing each team with their factor for the
        respective season.

    '''
    results=[]
    teams=factor_df['Team'].unique()
    seasons=factor_df['Season'].unique()

    for season in seasons:
        for selection in teams:
            condition = (factor_df['Team'] == selection) & (factor_df['Season'] == season)
            if not factor_df.loc[condition].empty:
                X_1 = weightings['w1'] * factor_df.loc[condition, 'TG>2'].values[0]
                X_2 = weightings['w2'] * factor_df.loc[condition, 'G/TGI'].values[0]
                X_3 = weightings['w3'] * factor_df.loc[condition, 'TGI per Game'].values[0]
                f = X_1 + X_2 + X_3
                results.append({'Team': selection, 'Season': season, 'Factor': f})
                results_df=pd.DataFrame(results)
                
    return results_df
#results_df.to_excel('/Users/joshcohen/Downloads/factor.xlsx',sheet_name='x1x2x3',index=False)
#Exporting to Excel

def generate_x4x5_factors(data, teams, weightings, output_path):
    """
    Computes X_4 and X_5 factors for each team at each match date using the last 5 games played,
    and saves the results as a sheet 'x4x5' in the given Excel file.

    Parameters:
    - data (pd.DataFrame): Match data with columns like 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'TG'.
    - teams (list): List of team names to compute factors for.
    - weightings (dict): Dictionary with 'w4' and 'w5' keys for weight values.
    - output_path (str): Path to the Excel file where the 'x4x5' sheet will be saved.
    """
    abc = []
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

    def get_past_5(df, team, date):
        date = pd.to_datetime(date, dayfirst=True)
        team_fixtures = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
        recent_fixtures = team_fixtures[team_fixtures['Date'] < date].sort_values(by='Date', ascending=False)
        x = recent_fixtures.head(5)
        if len(x) < 5 or x['TG'].sum() == 0:
            return None, None  # Not enough data or avoid divide-by-zero
        X_4 = weightings['w4'] * ((x['TG'].sum()) / 5)
        goals_scored = (
            x.loc[x['HomeTeam'] == team, 'FTHG'].sum() +
            x.loc[x['AwayTeam'] == team, 'FTAG'].sum()
        )
        X_5 = weightings['w5'] * (goals_scored / x['TG'].sum())
        return X_4, X_5

    # Sort the fixture data just once
    data = data.sort_values(by='Date')

    for team in teams:
        team_fixtures = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)].sort_values(by='Date')
    
        for i in range(1, len(team_fixtures)):
            date = team_fixtures.iloc[i]['Date']
            X_4, X_5 = get_past_5(data, team, date)
            if X_4 is not None and X_5 is not None:
                total_factor = X_4 + X_5
                abc.append({'Date': date, 'Team': team, 'Total Factor': total_factor})


    abc_df = pd.DataFrame(abc)

    # Save to Excel
    with pd.ExcelWriter(output_path, mode='a', engine='openpyxl') as writer:
        abc_df.to_excel(writer, sheet_name='x4x5', index=False)

def merge_factors_with_fixtures(fixtures_path, x1x2x3_path, x4x5_path, output_path=None):
    """
    Merges x1x2x3 and x4x5 factor data per season and combines the result with the fixtures dataset.

    Parameters:
    - fixtures_path (str): Path to the Excel or CSV file containing fixtures with Date, HomeTeam, AwayTeam.
    - x1x2x3_path (str): Path to Excel file containing 'x1x2x3' sheet.
    - x4x5_path (str): Path to Excel file containing 'x4x5' sheet.
    - output_path (str, optional): If provided, saves the final merged DataFrame to this path.

    Returns:
    - pd.DataFrame: The full merged DataFrame with Home Factor and Away Factor only.
    """

    import pandas as pd

    # Load data
    fixtures = pd.read_excel(fixtures_path) if fixtures_path.endswith('.xlsx') else pd.read_csv(fixtures_path)
    x1x2x3 = pd.read_excel(x1x2x3_path, sheet_name='x1x2x3')
    x4x5 = pd.read_excel(x4x5_path, sheet_name='x4x5')

    # Ensure date format is datetime
    fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True)
    x4x5['Date'] = pd.to_datetime(x4x5['Date'], dayfirst=True)

    all_results = []

    for season in sorted(x1x2x3['Season'].unique()):
        try:
            # Convert 'YY/YY' season format to full year
            start_year = int("20" + season[:2]) if int(season[:2]) < 50 else int("19" + season[:2])
            end_year = start_year + 1

            start_date = pd.to_datetime(f"01-08-{start_year}", dayfirst=True)
            end_date = pd.to_datetime(f"31-07-{end_year}", dayfirst=True)

            # Filter data by season
            f1 = x4x5[(x4x5['Date'] >= start_date) & (x4x5['Date'] <= end_date)]
            f2 = x1x2x3[x1x2x3['Season'] == season]

            # Merge and compute factor sum
            merged = pd.merge(f1, f2, on='Team', how='inner')
            merged['Factor Sum'] = merged['Total Factor'] + merged['Factor']
            merged = merged[['Date', 'Team', 'Factor Sum']]  # Keep only necessary columns

            all_results.append(merged)
        except Exception as e:
            print(f"Error processing season {season}: {e}")

    # Combine seasonal factor data
    factor_merged = pd.concat(all_results, ignore_index=True)

    # Merge with fixtures to get Home and Away Factors
    full_df = fixtures.merge(
        factor_merged.rename(columns={'Team': 'HomeTeam', 'Factor Sum': 'Home Factor'}),
        on=['Date', 'HomeTeam'],
        how='left'
    )

    full_df = full_df.merge(
        factor_merged.rename(columns={'Team': 'AwayTeam', 'Factor Sum': 'Away Factor'}),
        on=['Date', 'AwayTeam'],
        how='left'
    )

    # Save result if path is provided
    if output_path:
        full_df.to_excel(output_path, index=False)

    return full_df

def enrich_team_stats(df,output_path=None):
    # Create team-level data from home perspective
    home_stats = df[['Date', 'HomeTeam', 'HS', 'HST', 'FTHG']].copy()
    home_stats.columns = ['Date', 'Team', 'Shots', 'ShotsOnTarget', 'Goals']
    home_stats['HomeAway'] = 'Home'

    # Create team-level data from away perspective
    away_stats = df[['Date', 'AwayTeam', 'AS', 'AST', 'FTAG']].copy()
    away_stats.columns = ['Date', 'Team', 'Shots', 'ShotsOnTarget', 'Goals']
    away_stats['HomeAway'] = 'Away'

    # Combine into one team-centric dataset
    team_stats = pd.concat([home_stats, away_stats])
    team_stats = team_stats.sort_values(by=['Team', 'Date'])

    # Conversion Rate = Goals / Shots
    team_stats['ConversionRate'] = team_stats['Goals'] / team_stats['Shots'].replace(0, np.nan)

    # Rolling stats (exclude current game using shift)
    team_stats['AvgShots_5'] = team_stats.groupby('Team')['Shots'].shift(1).rolling(5).mean()
    team_stats['AvgShotsOnTarget_5'] = team_stats.groupby('Team')['ShotsOnTarget'].shift(1).rolling(5).mean()
    team_stats['ConversionRate_5'] = team_stats.groupby('Team')['ConversionRate'].shift(1).rolling(5).mean()
    team_stats['ConversionRate_20'] = team_stats.groupby('Team')['ConversionRate'].shift(1).rolling(20).mean()
    
    # Select only necessary columns for merging
    stats_cols = ['Date', 'Team', 'AvgShots_5', 'AvgShotsOnTarget_5', 'ConversionRate_5', 'ConversionRate_20']
    team_stats = team_stats[stats_cols]
    
    # Merge for Home Team
    df = df.merge(
        team_stats[['Date', 'Team', 'AvgShots_5', 'AvgShotsOnTarget_5', 'ConversionRate_5', 'ConversionRate_20']],
        left_on=['Date', 'HomeTeam'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={
        'AvgShots_5': 'Home_AvgShots_5',
        'AvgShotsOnTarget_5': 'Home_AvgShotsOnTarget_5',
        'ConversionRate_5': 'Home_ConversionRate_5',
        'ConversionRate_20': 'Home_ConversionRate_20'
    }).drop('Team', axis=1)

    # Merge for Away Team
    df = df.merge(
        team_stats[['Date', 'Team', 'AvgShots_5', 'AvgShotsOnTarget_5', 'ConversionRate_5', 'ConversionRate_20']],
        left_on=['Date', 'AwayTeam'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={
        'AvgShots_5': 'Away_AvgShots_5',
        'AvgShotsOnTarget_5': 'Away_AvgShotsOnTarget_5',
        'ConversionRate_5': 'Away_ConversionRate_5',
        'ConversionRate_20': 'Away_ConversionRate_20'
    }).drop('Team', axis=1)
        
    if output_path:
        df.to_excel(output_path, index=False)

    return df

def add_conversion_rates(df, output_path=None):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    home_cr_5 = []
    home_cr_20 = []
    away_cr_5 = []
    away_cr_20 = []

    home_sot_cr_5 = []  # NEW
    home_sot_cr_20 = []  # NEW
    away_sot_cr_5 = []  # NEW
    away_sot_cr_20 = []  # NEW

    for idx, row in df.iterrows():
        date = row['Date']
        home = row['HomeTeam']
        away = row['AwayTeam']

        # Previous matches for each team (Home)
        past_home = df[
            ((df['HomeTeam'] == home) | (df['AwayTeam'] == home)) &
            (df['Date'] < date)
        ].sort_values('Date')

        # Previous matches for each team (Away)
        past_away = df[
            ((df['HomeTeam'] == away) | (df['AwayTeam'] == away)) &
            (df['Date'] < date)
        ].sort_values('Date')

        # Home team: goals, shots, shots on target
        last5_home = past_home.tail(5)
        last20_home = past_home.tail(20)

        goals_5_home = 0
        shots_5_home = 0
        shots_on_target_5_home = 0

        goals_20_home = 0
        shots_20_home = 0
        shots_on_target_20_home = 0

        for _, past in last5_home.iterrows():
            if past['HomeTeam'] == home:
                goals_5_home += past['FTHG']
                shots_5_home += past['HS']
                shots_on_target_5_home += past['HST']
            else:
                goals_5_home += past['FTAG']
                shots_5_home += past['AS']
                shots_on_target_5_home += past['AST']

        for _, past in last20_home.iterrows():
            if past['HomeTeam'] == home:
                goals_20_home += past['FTHG']
                shots_20_home += past['HS']
                shots_on_target_20_home += past['HST']
            else:
                goals_20_home += past['FTAG']
                shots_20_home += past['AS']
                shots_on_target_20_home += past['AST']

        # Away team: goals, shots, shots on target
        last5_away = past_away.tail(5)
        last20_away = past_away.tail(20)

        goals_5_away = 0
        shots_5_away = 0
        shots_on_target_5_away = 0

        goals_20_away = 0
        shots_20_away = 0
        shots_on_target_20_away = 0

        for _, past in last5_away.iterrows():
            if past['HomeTeam'] == away:
                goals_5_away += past['FTHG']
                shots_5_away += past['HS']
                shots_on_target_5_away += past['HST']
            else:
                goals_5_away += past['FTAG']
                shots_5_away += past['AS']
                shots_on_target_5_away += past['AST']

        for _, past in last20_away.iterrows():
            if past['HomeTeam'] == away:
                goals_20_away += past['FTHG']
                shots_20_away += past['HS']
                shots_on_target_20_away += past['HST']
            else:
                goals_20_away += past['FTAG']
                shots_20_away += past['AS']
                shots_on_target_20_away += past['AST']

        # Original conversion rates
        home_cr_5.append(goals_5_home / shots_5_home if shots_5_home > 0 else None)
        home_cr_20.append(goals_20_home / shots_20_home if shots_20_home > 0 else None)
        away_cr_5.append(goals_5_away / shots_5_away if shots_5_away > 0 else None)
        away_cr_20.append(goals_20_away / shots_20_away if shots_20_away > 0 else None)

        # NEW: SOT conversion rates
        home_sot_cr_5.append(goals_5_home / shots_on_target_5_home if shots_on_target_5_home > 0 else None)
        home_sot_cr_20.append(goals_20_home / shots_on_target_20_home if shots_on_target_20_home > 0 else None)
        away_sot_cr_5.append(goals_5_away / shots_on_target_5_away if shots_on_target_5_away > 0 else None)
        away_sot_cr_20.append(goals_20_away / shots_on_target_20_away if shots_on_target_20_away > 0 else None)

    # Add to DataFrame
    df['Home_CR_5'] = home_cr_5
    df['Home_CR_20'] = home_cr_20
    df['Away_CR_5'] = away_cr_5
    df['Away_CR_20'] = away_cr_20

    df['Home_SOT_CR_5'] = home_sot_cr_5  # NEW
    df['Home_SOT_CR_20'] = home_sot_cr_20  # NEW
    df['Away_SOT_CR_5'] = away_sot_cr_5  # NEW
    df['Away_SOT_CR_20'] = away_sot_cr_20  # NEW

    if output_path:
        df.to_excel(output_path, index=False)

    return df


def add_shot_ratio_features(df,output_path=None):
    """
    Adds rolling 5-game shot ratio features to the original dataframe for both Home and Away teams.

    Parameters:
        df (pd.DataFrame): Input match dataframe with columns ['Date', 'HomeTeam', 'AwayTeam', 'HS', 'HST', 'AS', 'AST']

    Returns:
        pd.DataFrame: Original dataframe with added columns 'Home_ShotRatio_5' and 'Away_ShotRatio_5'
    """
    # Ensure 'Date' is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Create long format data
    home_df = df[['Date', 'HomeTeam', 'HS', 'HST']].copy()
    home_df.columns = ['Date', 'Team', 'Shots', 'ShotsOnTarget']
    home_df['Side'] = 'Home'

    away_df = df[['Date', 'AwayTeam', 'AS', 'AST']].copy()
    away_df.columns = ['Date', 'Team', 'Shots', 'ShotsOnTarget']
    away_df['Side'] = 'Away'

    long_df = pd.concat([home_df, away_df])
    long_df.sort_values(by=['Team', 'Date'], inplace=True)

    # Rolling 5-game shot ratio (Shots / Shots on Target)
    long_df['ShotRatio_5'] = (
        long_df.groupby('Team')
        .apply(lambda g: g['Shots'].shift(1).rolling(5).sum() / g['ShotsOnTarget'].shift(1).rolling(5).sum())
        .reset_index(level=0, drop=True)
    )

    # Split back and merge with original df
    home_conv = long_df[long_df['Side'] == 'Home'][['Date', 'Team', 'ShotRatio_5']]
    away_conv = long_df[long_df['Side'] == 'Away'][['Date', 'Team', 'ShotRatio_5']]

    df = df.merge(home_conv, left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    df.rename(columns={'ShotRatio_5': 'Home_ShotRatio_5'}, inplace=True)
    df.drop(columns=['Team'], inplace=True)

    df = df.merge(away_conv, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    df.rename(columns={'ShotRatio_5': 'Away_ShotRatio_5'}, inplace=True)
    df.drop(columns=['Team'], inplace=True)
    
    if output_path:
        df.to_excel(output_path, index=False)
        
    return df

def add_shots_per_goal_ratio(df,output_path=None):
    """
    Adds rolling 5-game ShotsPerGoal ratio for both Home and Away teams to the match dataframe.

    Parameters:
        df (pd.DataFrame): Match dataframe with columns ['Date', 'HomeTeam', 'AwayTeam', 'HS', 'FTHG', 'AS', 'FTAG']

    Returns:
        pd.DataFrame: DataFrame with added columns 'Home_ShotsPerGoal_5' and 'Away_ShotsPerGoal_5'
    """
    # Prepare long format data
    home_df = df[['Date', 'HomeTeam', 'HS', 'FTHG']].copy()
    home_df.columns = ['Date', 'Team', 'Shots', 'Goals']

    away_df = df[['Date', 'AwayTeam', 'AS', 'FTAG']].copy()
    away_df.columns = ['Date', 'Team', 'Shots', 'Goals']

    long_df = pd.concat([home_df, away_df], ignore_index=True)
    long_df.sort_values(by=['Team', 'Date'], inplace=True)

    # Calculate rolling 5-game Shots per Goal (exclude current match)
    long_df['ShotsPerGoal_5'] = (
        (long_df['Shots'].shift(1) / long_df['Goals'].replace(0, pd.NA).shift(1))
        .groupby(long_df['Team'])
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Merge with original df for both Home and Away teams
    df = df.sort_values('Date').reset_index(drop=True)

    df = df.merge(
        long_df[['Date', 'Team', 'ShotsPerGoal_5']],
        left_on=['Date', 'HomeTeam'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={'ShotsPerGoal_5': 'Home_ShotsPerGoal_5'}).drop(columns='Team')

    df = df.merge(
        long_df[['Date', 'Team', 'ShotsPerGoal_5']],
        left_on=['Date', 'AwayTeam'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={'ShotsPerGoal_5': 'Away_ShotsPerGoal_5'}).drop(columns='Team')
    
    if output_path:
        df.to_excel(output_path, index=False)

    return df.sort_values('Date', ascending=False).reset_index(drop=True)

def add_league_positions(df, season_length=380, output_path=None):
    """
    Adds league position rankings for each team BEFORE each fixture in a rolling seasonal format.

    Parameters:
        df (pd.DataFrame): Fixture dataframe with ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        season_length (int): Number of matches in a season (default: 380)

    Returns:
        pd.DataFrame: Original dataframe with 'Home_LeaguePosition' and 'Away_LeaguePosition' columns.
    """
    import pandas as pd

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    all_season_rankings = []
    num_seasons = len(df) // season_length

    for i in range(num_seasons):
        start = i * season_length
        end = start + season_length
        season_df = df.iloc[start:end].copy()
        season_df['SeasonIndex'] = i

        # Create records for every match
        records = []
        for _, row in season_df.iterrows():
            date = row['Date']
            home, away = row['HomeTeam'], row['AwayTeam']
            fthg, ftag = row['FTHG'], row['FTAG']

            # Assign points
            if fthg > ftag:
                home_points, away_points = 3, 0
            elif fthg < ftag:
                home_points, away_points = 0, 3
            else:
                home_points, away_points = 1, 1

            records.extend([
                {'Team': home, 'Date': date, 'Points': home_points, 'GF': fthg, 'GA': ftag},
                {'Team': away, 'Date': date, 'Points': away_points, 'GF': ftag, 'GA': fthg}
            ])

        # Build cumulative stats
        stats_df = pd.DataFrame(records)
        stats_df['GD'] = stats_df['GF'] - stats_df['GA']
        stats_df.sort_values(['Team', 'Date'], inplace=True)

        stats_df['CumulativePoints'] = stats_df.groupby('Team')['Points'].cumsum()
        stats_df['CumulativeGF'] = stats_df.groupby('Team')['GF'].cumsum()
        stats_df['CumulativeGA'] = stats_df.groupby('Team')['GA'].cumsum()
        stats_df['CumulativeGD'] = stats_df['CumulativeGF'] - stats_df['CumulativeGA']

        match_positions = []

        for _, row in season_df.iterrows():
            match_date = row['Date']
            # Use only stats BEFORE this match
            prior_stats = stats_df[stats_df['Date'] < match_date]

            latest = prior_stats.sort_values(['Team', 'Date']).groupby('Team').tail(1)

            # Fill in teams that haven't played yet
            all_teams = pd.unique(season_df[['HomeTeam', 'AwayTeam']].values.ravel())
            missing = set(all_teams) - set(latest['Team'])
            for team in missing:
                latest = pd.concat([
                    latest,
                    pd.DataFrame([{
                        'Team': team,
                        'Date': match_date,
                        'CumulativePoints': 0,
                        'CumulativeGD': 0,
                        'CumulativeGF': 0
                    }])
                ], ignore_index=True)

            # Rank
            table = latest.sort_values(
                ['CumulativePoints', 'CumulativeGD', 'CumulativeGF'],
                ascending=False
            ).reset_index(drop=True)

            table['LeaguePosition'] = range(1, len(table) + 1)
            table['Date'] = match_date
            table['SeasonIndex'] = i

            match_positions.append(table[['Team', 'Date', 'LeaguePosition', 'SeasonIndex']])

        season_rank_df = pd.concat(match_positions).drop_duplicates(subset=['Team', 'Date', 'SeasonIndex'])
        all_season_rankings.append(season_rank_df)

    # Combine all seasons
    rank_df = pd.concat(all_season_rankings, ignore_index=True)

    df['SeasonIndex'] = df.index // season_length

    # Merge rankings
    df = df.merge(
        rank_df.rename(columns={'Team': 'HomeTeam', 'LeaguePosition': 'Home_LeaguePosition'}),
        on=['HomeTeam', 'Date', 'SeasonIndex'],
        how='left'
    )
    df = df.merge(
        rank_df.rename(columns={'Team': 'AwayTeam', 'LeaguePosition': 'Away_LeaguePosition'}),
        on=['AwayTeam', 'Date', 'SeasonIndex'],
        how='left'
    )

    if output_path:
        df.to_excel(output_path, index=False)

    return df

def add_defensive_strength(df, window=5,output_path=None):
    """
    Calculates and adds defensive strength metrics based on rolling shots conceded.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Date', 'HomeTeam', 'AwayTeam', 'HS', 'AS' columns.
        window (int): Rolling window size (default: 5)

    Returns:
        pd.DataFrame: Original DataFrame with 'Home_DefensiveStrength' and 'Away_DefensiveStrength' columns.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Step 2: Create long-form DataFrame for shots conceded
    home = df[['Date', 'HomeTeam', 'AS']].rename(columns={'HomeTeam': 'Team', 'AS': 'ShotsConceded'})
    away = df[['Date', 'AwayTeam', 'HS']].rename(columns={'AwayTeam': 'Team', 'HS': 'ShotsConceded'})

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(['Team', 'Date'])

    # Step 3: Compute rolling sum of shots conceded (exclude current match)
    long_df['RollingShotsConceded'] = (
        long_df.groupby('Team')['ShotsConceded']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    )

    # Step 4: Defensive strength = 1 / shots conceded
    long_df['DefensiveStrength'] = 1 / long_df['RollingShotsConceded']
    long_df['DefensiveStrength'] = long_df['DefensiveStrength'].replace([float('inf')], pd.NA)
    # Step 5: Merge back to main df
    home_strength = long_df.rename(columns={
        'Team': 'HomeTeam',
        'DefensiveStrength': 'Home_DefensiveStrength'
    })[['HomeTeam', 'Date', 'Home_DefensiveStrength']]

    away_strength = long_df.rename(columns={
        'Team': 'AwayTeam',
        'DefensiveStrength': 'Away_DefensiveStrength'
    })[['AwayTeam', 'Date', 'Away_DefensiveStrength']]

    df = df.merge(home_strength, on=['HomeTeam', 'Date'], how='left')
    df = df.merge(away_strength, on=['AwayTeam', 'Date'], how='left')

    # Fill NaNs with default median-based defensive strength
    default_strength = 1 / long_df['RollingShotsConceded'].median()
    df['Home_DefensiveStrength'] = df['Home_DefensiveStrength'].fillna(default_strength)
    df['Away_DefensiveStrength'] = df['Away_DefensiveStrength'].fillna(default_strength)
    
    if output_path:
        df.to_excel(output_path, index=False)
        
    return df

def add_promoted_column(df,output_path=None):
    """
    Adds a 'Promoted' column to the DataFrame, indicating if the team was newly promoted
    into the league that season (1 for promoted, 0 otherwise), based on a predefined dictionary.

    Parameters:
        df (pd.DataFrame): DataFrame that includes 'SeasonIndex', 'HomeTeam', 'AwayTeam'

    Returns:
        pd.DataFrame: Modified DataFrame with 'Home_Promoted' and 'Away_Promoted' columns.
    """
    # Dictionary of promoted teams per season index
    promoted_per_season = {
        0: ['Ipswich', 'Man City', 'Charlton Athletic'],
        1: ['Bolton Wanderers', 'Blackburn Rovers', 'Fulham'],
        2: ['West Bromwich Albion', 'Man City', 'Birmingham City'],
        3: ['Wolverhampton Wanderers', 'Leicester', 'Portsmouth'],
        4: ['West Bromwich Albion', 'Norwich', 'Crystal Palace'],
        5: ['West Ham', 'Sunderland', 'Wigan Athletic'],
        6: ['Watford', 'Reading', 'Sheffield United'],
        7: ['Birmingham City', 'Derby County', 'Sunderland'],
        8: ['West Bromwich Albion', 'Stoke', 'Hull'],
        9: ['Burnley', 'Birmingham City', 'Wolverhampton Wanderers'],
        10: ['Blackpool', 'West Bromwich Albion', 'Newcastle United'],
        11: ['Norwich', 'Swansea', 'Queens Park Rangers'],
        12: ['Southampton', 'Reading', 'West Ham'],
        13: ['Cardiff', 'Hull', 'Crystal Palace'],
        14: ['Leicester', 'Burnley', 'Queens Park Rangers'],
        15: ['Norwich', 'Bournemouth', 'Watford'],
        16: ['Middlesbrough', 'Burnley', 'Hull'],
        17: ['Huddersfield', 'Brighton', 'Newcastle United'],
        18: ['Cardiff', 'Fulham', 'Wolverhampton Wanderers'],
        19: ['Aston Villa', 'Norwich', 'Sheffield United'],
        20: ['Leeds', 'Fulham', 'West Bromwich Albion'],
        21: ['Brentford', 'Norwich', 'Watford'],
        22: ['Nottingham Forest', 'Fulham', 'Bournemouth'],
        23: ['Luton', 'Burnley', 'Sheffield United']
    }

    # Function to determine if a team was promoted in a given season
    def is_promoted(team, season_index):
        return int(team in promoted_per_season.get(season_index, []))

    # Apply to Home and Away teams
    df['Home_Promoted'] = df.apply(lambda row: is_promoted(row['HomeTeam'], row['SeasonIndex']), axis=1)
    df['Away_Promoted'] = df.apply(lambda row: is_promoted(row['AwayTeam'], row['SeasonIndex']), axis=1)
    
    if output_path:
        df.to_excel(output_path, index=False)
        
    return df 

def classify_derbies(df, output_path=None):
    """
    Adds 'Local Derby', 'Historical Derby', and 'Not a Derby' columns to the fixture DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'HomeTeam' and 'AwayTeam' columns.
        local_derbies (list of tuples): List of team pairs considered local derbies.
        historical_derbies (list of tuples): List of team pairs considered historical derbies.

    Returns:
        pd.DataFrame: DataFrame with three new binary columns.
    """
    local_derbies = [
    ('Arsenal', 'Tottenham'),
    ('Man United', 'Man City'),
    ('Liverpool', 'Everton'),
    ('Chelsea', 'Fulham'),
    ('Newcastle', 'Sunderland'),
    ('Cardiff', 'Swansea'),
    ('Aston Villa', 'Birmingham City'),
    ('Aston Villa', 'Wolverhampton Wanderers'),
    ('Aston Villa', 'Coventry City'),
    ('Crystal Palace', 'Brighton'),
    ('Fulham', 'Brentford'),
    ('Fulham', 'Queens Park Rangers'),
    ('Ipswich', 'Norwich'),
    ('Leeds', 'Huddersfield'),
    ('Leicester', 'Nottingham Forest'),
    ('Portsmouth', 'Southampton'),
    ('Burnley', 'Blackburn Rovers'),
    ('Bolton Wanderers', 'Wigan Athletic'),
    ('Aston Villa', 'West Bromwich Albion'),
    ('Wolverhampton Wanderers', 'West Bromwich Albion'),
    ('Tottenham', 'West Ham')
    ]
    
    historical_derbies=[
    ('Man United', 'Liverpool'),
    ('Man United', 'Arsenal'),
    ('Tottenham', 'Chelsea'),
    ('Arsenal', 'Chelsea'),
    ('Arsenal', 'Liverpool'),
    ('Liverpool', 'Man City'),
    ('Leeds', 'Man United'),
    ('Chelsea', 'Man United')    
    ]
    
    # Normalize derby lists for set comparison
    local_set = {frozenset(pair) for pair in local_derbies}
    historical_set = {frozenset(pair) for pair in historical_derbies}

    # Helper function for classification
    def get_derby_type(home, away):
        fixture = frozenset([home, away])
        if fixture in local_set:
            return (1, 0, 0)
        elif fixture in historical_set:
            return (0, 1, 0)
        else:
            return (0, 0, 1)

    # Apply to each row
    df[['Local Derby', 'Historical Derby', 'Not a Derby']] = df.apply(
        lambda row: pd.Series(get_derby_type(row['HomeTeam'], row['AwayTeam'])),
        axis=1
    )
    
    if output_path:
        df.to_excel(output_path, index=False)

    return df

def add_h2h_features(df: pd.DataFrame, output_path=None) -> pd.DataFrame:
    # Ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize lists to store features
    h2h_home_wins = []
    h2h_away_wins = []
    h2h_draws = []
    h2h_avg_goals = []
    h2h_match_count = []
    h2h_avg_goals_alltime = []  # NEW list for all-time average
    
    # Iterate over each match
    for idx, row in df.iterrows():
        date = row['Date']
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Filter past 5 matches between the two teams before the current date
        past_matches = df[
            (((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
             ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))) &
            (df['Date'] < date)
        ].sort_values('Date')
        
        # Last 5
        last_5_matches = past_matches.tail(5)
        
        # Initialize counters
        home_win = 0
        away_win = 0
        draw = 0
        total_goals_last5 = 0
        total_goals_alltime = 0
        
        # Calculate for last 5 matches
        for _, past in last_5_matches.iterrows():
            htg = past['FTHG']
            atg = past['FTAG']
            total_goals_last5 += htg + atg

            if past['HomeTeam'] == home:
                if htg > atg:
                    home_win += 1
                elif htg < atg:
                    away_win += 1
                else:
                    draw += 1
            else:
                if atg > htg:
                    home_win += 1
                elif atg < htg:
                    away_win += 1
                else:
                    draw += 1

        # Calculate total goals for ALL previous matches
        for _, past in past_matches.iterrows():
            htg = past['FTHG']
            atg = past['FTAG']
            total_goals_alltime += htg + atg

        # Append features
        h2h_home_wins.append(home_win)
        h2h_away_wins.append(away_win)
        h2h_draws.append(draw)
        h2h_avg_goals.append(total_goals_last5 / len(last_5_matches) if len(last_5_matches) > 0 else None)
        h2h_match_count.append(len(last_5_matches))
        h2h_avg_goals_alltime.append(total_goals_alltime / len(past_matches) if len(past_matches) > 0 else None)
    
    # Assign features to the DataFrame
    df['H2H_HomeWins'] = h2h_home_wins
    df['H2H_AwayWins'] = h2h_away_wins
    df['H2H_Draws'] = h2h_draws
    df['H2H_AvgGoals'] = h2h_avg_goals
    df['H2H_MatchCount'] = h2h_match_count
    df['H2HAvgGoals_AllTime'] = h2h_avg_goals_alltime  # NEW column
    
    if output_path:
        df.to_excel(output_path, index=False)

    return df


