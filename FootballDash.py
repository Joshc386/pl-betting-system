# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 18:18:27 2025

@author: joshc
"""

# football_dashboard_spyder.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import webbrowser
import base64, io

# -------------------- Load and preprocess data --------------------
df = pd.read_csv(r"C:\Users\joshc\OneDrive\Documents\Project\CompleteDSPL_CSV.csv")  # replace with your CSV

# Add outcome column
def get_outcome(row):
    if row['Home_Goals'] > row['Away_Goals']:
        return 'Home Win'
    elif row['Home_Goals'] < row['Away_Goals']:
        return 'Away Win'
    else:
        return 'Draw'
df['Outcome'] = df.apply(get_outcome, axis=1)

# Encode Outcome
outcome_map = {'Home Win':0, 'Draw':1, 'Away Win':2}
df['OutcomeEncoded'] = df['Outcome'].map(outcome_map)

# Example features for ML model
features = ['Home_Goals','Away_Goals','Home_Shots','Home_Shots_Target',
            'Away_Shots','Away_Shots_Target','Home_Corners','Away_Corners']

# Train simple RandomForest model
X = df[features]
y = df['OutcomeEncoded']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# -------------------- Helper functions --------------------
def last5_form(team, season):
    team_games = df[((df['Home_Team']==team)|(df['Away_Team']==team)) &
                    (df['SeasonIndex']==season)].sort_values('SeasonIndex')
    last5 = team_games.tail(5)
    form_score = 0
    for idx, row in last5.iterrows():
        if row['Home_Team']==team:
            if row['Outcome']=='Home Win':
                form_score +=3
            elif row['Outcome']=='Draw':
                form_score +=1
        else:
            if row['Outcome']=='Away Win':
                form_score +=3
            elif row['Outcome']=='Draw':
                form_score +=1
    return form_score/15  # normalized

def get_league_position(season):
    # Simple points table calculation
    season_games = df[df['SeasonIndex']==season]
    teams = df['Home_Team'].unique()
    table = []
    for team in teams:
        games = season_games[(season_games['Home_Team']==team)|(season_games['Away_Team']==team)]
        points = 0
        for idx,row in games.iterrows():
            if row['Home_Team']==team:
                if row['Outcome']=='Home Win':
                    points+=3
                elif row['Outcome']=='Draw':
                    points+=1
            else:
                if row['Outcome']=='Away Win':
                    points+=3
                elif row['Outcome']=='Draw':
                    points+=1
        table.append({'Team':team,'Points':points})
    table_df = pd.DataFrame(table).sort_values('Points',ascending=False)
    table_df['Position'] = np.arange(1,len(table_df)+1)
    return table_df.set_index('Team')['Position'].to_dict()

# -------------------- Initialize Dash App --------------------
app = Dash(__name__)
port = 8051
url = f"http://127.0.0.1:{port}/"
webbrowser.open(url)

app.layout = html.Div([
    html.H1("Football Match Winner Predictor", style={"textAlign":"center"}),

    # -------------------- Filters --------------------
    html.Div([
        html.Label("Select Team(s):"),
        dcc.Dropdown(options=[{'label': t,'value':t} for t in df['Home_Team'].unique()],
                     multi=True, id='team-filter'),
        html.Label("Select Season(s):"),
        dcc.Dropdown(options=[{'label': s,'value':s} for s in df['SeasonIndex'].unique()],
                     multi=True, id='season-filter')
    ], style={'width':'50%', 'display':'inline-block'}),

    # -------------------- Statistics Table --------------------
    html.H2("Team Statistics"),
    dash_table.DataTable(id='stats-table',
                         columns=[{"name": "Team", "id":"Team"},
                                  {"name": "Avg Goals Scored", "id":"Avg_Goals_Scored"},
                                  {"name": "Avg Goals Conceded", "id":"Avg_Goals_Conceded"},
                                  {"name": "Avg Goal Involvements", "id":"Avg_Goal_Involvements"}],
                         style_table={'overflowX':'auto'},
                         page_size=20),

    # -------------------- Goal Involvement Plot --------------------
    dcc.Graph(id='goal-involvement-plot'),

    # -------------------- File Upload --------------------
    html.Div([
        html.H2("Upload Match File for Predictions"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
            style={'width':'50%','height':'60px','lineHeight':'60px',
                   'borderWidth':'1px','borderStyle':'dashed','borderRadius':'5px',
                   'textAlign':'center','margin':'10px'},
            multiple=False
        ),
        html.Div(id='uploaded-table')
    ])
])

# -------------------- Callbacks --------------------
@app.callback(
    Output('stats-table','data'),
    Output('goal-involvement-plot','figure'),
    Input('team-filter','value'),
    Input('season-filter','value')
)
def update_stats(selected_teams, selected_seasons):
    dff = df.copy()

    if selected_seasons:
        dff = dff[dff['SeasonIndex'].isin(selected_seasons)]
    if selected_teams:
        dff = dff[(dff['Home_Team'].isin(selected_teams)) | (dff['Away_Team'].isin(selected_teams))]

    teams = pd.concat([dff['Home_Team'], dff['Away_Team']]).unique()
    stats_list = []

    for t in teams:
        team_games = dff[(dff['Home_Team'] == t) | (dff['Away_Team'] == t)]

        goals_scored = []
        goals_conceded = []

        for _, row in team_games.iterrows():
            if row['Home_Team'] == t:
                goals_scored.append(row['Home_Goals'])
                goals_conceded.append(row['Away_Goals'])
            else:  # Away team
                goals_scored.append(row['Away_Goals'])
                goals_conceded.append(row['Home_Goals'])

        avg_scored = sum(goals_scored)/len(goals_scored) if goals_scored else 0
        avg_conceded = sum(goals_conceded)/len(goals_conceded) if goals_conceded else 0
        avg_involv = avg_scored + avg_conceded

        stats_list.append({
            'Team': t,
            'Avg_Goals_Scored': round(avg_scored,2),
            'Avg_Goals_Conceded': round(avg_conceded,2),
            'Avg_Goal_Involvements': round(avg_involv,2)
        })

    stats_df = pd.DataFrame(stats_list)
    fig = px.bar(stats_df, x='Team', y='Avg_Goal_Involvements', 
                 title="Average Goal Involvements per Team")

    return stats_df.to_dict('records'), fig

@app.callback(
    Output('uploaded-table','children'),
    Input('upload-data','contents')
)
def update_predictions(contents):
    if contents is None:
        return "Upload a CSV file to see predictions."
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    uploaded_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    predict_rows = []
    prob_rows = []
    for idx,row in uploaded_df.iterrows():
        season = row['SeasonIndex']
        home_form = last5_form(row['Home_Team'],season)
        away_form = last5_form(row['Away_Team'],season)
        league_pos = get_league_position(season)
        home_pos = league_pos.get(row['Home_Team'],10)
        away_pos = league_pos.get(row['Away_Team'],10)

        home_stats = df[df['Home_Team']==row['Home_Team']][features].mean()
        away_stats = df[df['Away_Team']==row['Away_Team']][features].mean()
        combined = pd.concat([home_stats, away_stats])
        combined['Home_Form'] = home_form
        combined['Away_Form'] = away_form
        combined['Home_Pos'] = home_pos
        combined['Away_Pos'] = away_pos
        predict_rows.append(combined)

    predict_df = pd.DataFrame(predict_rows).fillna(0)
    predict_scaled = scaler.transform(predict_df[features])
    predictions = model.predict(predict_scaled)
    probabilities = model.predict_proba(predict_scaled)

    uploaded_df['Prediction'] = [list(outcome_map.keys())[p] for p in predictions]
    uploaded_df['Prob_HomeWin'] = (probabilities[:,0]*100).round(1)
    uploaded_df['Prob_Draw'] = (probabilities[:,1]*100).round(1)
    uploaded_df['Prob_AwayWin'] = (probabilities[:,2]*100).round(1)

    return dash_table.DataTable(
        data=uploaded_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in uploaded_df.columns],
        style_table={'overflowX':'auto'},
        page_size=10
    )
# -------------------- Run App --------------------
if __name__ == "__main__":
    port = 8051
    url = f"http://127.0.0.1:{port}/"
    webbrowser.open(url)  # Automatically open browser
    app.run_server(debug=True, port=port, use_reloader=False)