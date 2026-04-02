# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 15:58:20 2025

@author: joshc
"""

from Functions import *
#fixtures_path="\\Users\\joshc\\OneDrive\\Documents\\Project\\Tableau_Export.xlsx"
#factor_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\Factorisation.xlsx'
#Get_Games_TG_gt_2(fixtures_path,season=22)

#compute_weights()
#weightings

#factor_df=pd.read_excel(factor_path)
#seasons=factor_df['Season'].unique()
#teams=factor_df['Team'].unique()
#calculate_x1x2x3(factor_df,seasons, teams, weightings).to_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\TestFactor.xlsx',sheet_name='x1x2x3',index=False)
#generate_x4x5_factors(pd.read_excel("\\Users\\joshc\\OneDrive\\Documents\\Project\\Tableau_Export.xlsx"), teams, weightings,'\\Users\\joshc\\OneDrive\\Documents\\Project\\TestFactor.xlsx')
#merge_factors_with_fixtures(fixtures_path, '\\Users\\joshc\\OneDrive\\Documents\\Project\\TestFactor.xlsx', '\\Users\\joshc\\OneDrive\\Documents\\Project\\TestFactor.xlsx',output_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx')
#enrich_team_stats(pd.read_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx'),output_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx')
#add_shot_ratio_features(pd.read_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx'),output_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx')
#add_shots_per_goal_ratio(pd.read_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx'),output_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx')
add_league_positions(pd.read_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\PLNew2425.xlsx'), season_length=380, output_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\PLNew2425.xlsx')
#add_defensive_strength(pd.read_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx'),window=5 ,output_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx')
#add_promoted_column(pd.read_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx'),output_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx')
#classify_derbies(pd.read_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx'), output_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx')
#add_conversion_rates(pd.read_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx'), output_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx')
#add_h2h_features(pd.read_excel('\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx'), output_path='\\Users\\joshc\\OneDrive\\Documents\\Project\\TestData.xlsx')