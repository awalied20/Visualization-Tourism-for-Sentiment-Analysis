import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import base64
from io import BytesIO

# Load and preprocess data
df = pd.read_csv("preprocessed_tourism_reviewsv2.csv")
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M').astype(str)
df['day_of_week'] = df['date'].dt.day_name()
df['month'] = df['date'].dt.month_name()
df['year'] = df['date'].dt.year
unique_periods = sorted(df['year_month'].unique())

# Color map for sentiments
color_map = {'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#1f77b4'}

# Word cloud generator by sentiment
def generate_wordcloud(sentiment):
    text = ' '.join(df[df['sentiment'] == sentiment]['processed_review'].dropna())
    wc = WordCloud(width=800, height=400, background_color='black', colormap='plasma', max_words=100)
    img = wc.generate(text)
    buf = BytesIO()
    img.to_image().save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

# Compute mean word count per sentiment
def compute_mean_words(filtered_df):
    return filtered_df.groupby('sentiment')['word_count'].mean().reset_index(name='mean_word_count')

# Initialize app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "TripAdvisor Sentiment Dashboard"

# App layout
app.layout = html.Div([
    dcc.Markdown('''
    <style>
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in {
            animation: fadeInUp 0.6s ease-in-out both;
        }
        .fade-in:nth-child(1) { animation-delay: 0.2s; }
        .fade-in:nth-child(2) { animation-delay: 0.4s; }
        .fade-in:nth-child(3) { animation-delay: 0.6s; }
        .fade-in:nth-child(4) { animation-delay: 0.8s; }
        .fade-in:nth-child(5) { animation-delay: 1s; }
        .fade-in:nth-child(6) { animation-delay: 1.2s; }

        .Select-control, .Select-menu-outer, .Select-value-label {
            background-color: #161b22 !important;
            color: #c9d1d9 !important;
        }
    </style>
    ''', dangerously_allow_html=True),

    html.Div(style={'backgroundColor': '#0d1117', 'padding': '40px', 'textAlign': 'center'}, children=[
        html.H1("📛 TripAdvisor Sentiment Dashboard", style={'color': '#c9d1d9', 'fontSize': '3rem'})
    ]),

    html.Div(style={'textAlign': 'center', 'backgroundColor': '#0d1117'}, children=[
        html.Label("Select Sentiment for Word Cloud", style={'color': '#c9d1d9'}),
        dcc.Dropdown(
            id='wordcloud-sentiment-dropdown',
            options=[{'label': s, 'value': s} for s in df['sentiment'].unique()],
            value='Positive',
            style={'width': '300px', 'margin': '0 auto', 'color': '#161b22'}
        ),
        html.Img(id='wordcloud-img', style={'width': '80%', 'maxWidth': '800px', 'borderRadius': '10px', 'marginTop': '20px'})
    ]),

    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'padding': '20px', 'backgroundColor': '#161b22'}, children=[
        dcc.Dropdown(
            id='attraction-dropdown',
            options=[{'label': a, 'value': a} for a in df['attraction'].unique()],
            multi=True, placeholder='Filter Attractions',
            style={'width': '250px', 'color': '#161b22'}
        ),
        dcc.Dropdown(
            id='sentiment-dropdown',
            options=[{'label': s, 'value': s} for s in df['sentiment'].unique()],
            multi=True, placeholder='Filter Sentiments',
            style={'width': '250px', 'color': '#161b22'}
        )
    ]),

    html.Div([
        html.Div(dcc.Graph(id='sentiment-distribution', config={'displayModeBar': False}), className='fade-in'),
        html.Div(dcc.Graph(id='rating-by-sentiment', config={'displayModeBar': False}), className='fade-in'),
        html.Div(dcc.Graph(id='reviews-over-time', config={'displayModeBar': False}), className='fade-in'),
        html.Div(dcc.Graph(id='avg-rating-attraction', config={'displayModeBar': False}), className='fade-in'),
        html.Div(dcc.Graph(id='sentiment-attraction-stacked', config={'displayModeBar': False}), className='fade-in'),
        html.Div(dcc.Graph(id='mean-word-count', config={'displayModeBar': False}), className='fade-in'),
        html.Div(dcc.Graph(id='sentiment-by-day', config={'displayModeBar': False}), className='fade-in'),
        html.Div(dcc.Graph(id='top-attractions', config={'displayModeBar': False}), className='fade-in')
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(400px, 1fr))', 'gap': '30px', 'padding': '20px', 'backgroundColor': '#0d1117'})

], style={'margin': '0', 'padding': '0', 'backgroundColor': '#0d1117', 'fontFamily': 'Segoe UI'})

@app.callback(
    Output('wordcloud-img', 'src'),
    Input('wordcloud-sentiment-dropdown', 'value')
)
def update_wordcloud(sentiment):
    return generate_wordcloud(sentiment)

@app.callback(
    [Output('sentiment-distribution', 'figure'),
     Output('rating-by-sentiment', 'figure'),
     Output('reviews-over-time', 'figure'),
     Output('avg-rating-attraction', 'figure'),
     Output('sentiment-attraction-stacked', 'figure'),
     Output('mean-word-count', 'figure'),
     Output('sentiment-by-day', 'figure'),
     Output('top-attractions', 'figure')],
    [Input('attraction-dropdown', 'value'), Input('sentiment-dropdown', 'value')]
)
def update_graphs(selected_attractions, selected_sentiments):
    flt = df.copy()
    if selected_attractions:
        flt = flt[flt['attraction'].isin(selected_attractions)]
    if selected_sentiments:
        flt = flt[flt['sentiment'].isin(selected_sentiments)]

    fig1 = px.histogram(flt, x='sentiment', color='sentiment', title='Sentiment Distribution',
                        color_discrete_map=color_map, template='plotly_dark')
    fig2 = px.violin(flt, x='sentiment', y='rate', color='sentiment', box=True, points='all',
                     title='Rating Distribution by Sentiment', color_discrete_map=color_map, template='plotly_dark')
    tmp = flt.groupby(['date', 'sentiment']).size().reset_index(name='count')
    fig3 = px.line(tmp, x='date', y='count', color='sentiment', title='Reviews Over Time',
                   animation_frame='sentiment', color_discrete_map=color_map, template='plotly_dark')
    avg_rt = flt.groupby('attraction')['rate'].mean().reset_index()
    fig4 = px.bar(avg_rt, x='attraction', y='rate', title='Average Rating per Attraction',
                  template='plotly_dark')
    sc = flt.groupby(['attraction', 'sentiment']).size().reset_index(name='count')
    fig5 = px.bar(sc, x='attraction', y='count', color='sentiment', barmode='stack',
                  title='Sentiment per Attraction', color_discrete_map=color_map, template='plotly_dark')
    mw = compute_mean_words(flt)
    fig6 = px.bar(mw, x='sentiment', y='mean_word_count', color='sentiment',
                  title='Mean Word Count per Sentiment', labels={'mean_word_count':'Avg Words'},
                  color_discrete_map=color_map, template='plotly_dark')
    by_day = flt.groupby(['day_of_week', 'sentiment']).size().reset_index(name='count')
    fig7 = px.bar(by_day, x='day_of_week', y='count', color='sentiment', barmode='group',
                  title='Sentiment by Day of Week', color_discrete_map=color_map, template='plotly_dark')
    top_att = flt['attraction'].value_counts().nlargest(10).reset_index()
    top_att.columns = ['attraction', 'review_count']
    fig8 = px.pie(top_att, names='attraction', values='review_count',
                  title='Top 10 Most Reviewed Attractions', template='plotly_dark')

    for fig in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]:
        fig.update_layout(transition={'duration': 500}, paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font_color='#c9d1d9')

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8

if __name__ == '__main__':
    app.run(debug=True, port=8050)
