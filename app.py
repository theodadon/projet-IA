import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import html, dcc, callback, Input, Output
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Chargement des données
apps_data_path = 'googleplaystore.csv'
reviews_data_path = 'googleplaystore_user_reviews.csv'
apps_data = pd.read_csv(apps_data_path)
reviews_data = pd.read_csv(reviews_data_path)

# Nettoyage et préparation des données
apps_data['Installs'] = apps_data['Installs'].str.replace(r'[+,]', '', regex=True).replace('Free', '0').astype(int)
apps_data['Reviews'] = pd.to_numeric(apps_data['Reviews'], errors='coerce').fillna(0).astype(int)
apps_data['Size'] = apps_data['Size'].replace('Varies with device', None)
apps_data['Size'] = apps_data['Size'].str.extract(r'(\d+\.?\d*)')[0].astype(float)
apps_data['Size'].fillna(apps_data['Size'].mean(), inplace=True)
imputer = SimpleImputer(strategy='median')
apps_data['Rating'] = imputer.fit_transform(apps_data[['Rating']])

# Préparation des données pour le modèle de prédiction
apps_data['Type'] = LabelEncoder().fit_transform(apps_data['Type'].fillna('Free'))
apps_data['Content Rating'] = LabelEncoder().fit_transform(apps_data['Content Rating'].fillna('Everyone'))
apps_data['Genres'] = apps_data['Genres'].fillna('Unknown').apply(lambda x: x.split(';')[0])

features = ['Reviews', 'Size', 'Installs', 'Type', 'Content Rating']
target = 'Category'

X = apps_data[features]
y = apps_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Initialisation de l'application Dash
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

# Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dcc.Link('Home', href='/', className='nav-link')),
        dbc.NavItem(dcc.Link('Data Exploration', href='/data-exploration', className='nav-link')),
        dbc.NavItem(dcc.Link('Application Prediction', href='/application-prediction', className='nav-link')),
        dbc.NavItem(dcc.Link('Sentiment Analysis', href='/sentiment-analysis', className='nav-link'))
    ],
    brand='Play Store App Dashboard',
    brand_href='/',
    color='dark',
    dark=True,
    className='mb-4'
)

# Layout principal
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content', className='p-4')
])

# Index page (Home)
index_page = html.Div([
    html.H1('Play Store App Dashboard', className='text-center mb-4'),
    html.P('Welcome to the Play Store App Dashboard. Use the navigation bar to explore data, predict applications, and analyze sentiments.', className='text-center'),
    dbc.Row([
        dbc.Col(dcc.Link('Get Started with Data Exploration', href='/data-exploration', className='btn btn-primary'), className='d-grid gap-2'),
        dbc.Col(dcc.Link('Predict Applications', href='/application-prediction', className='btn btn-primary'), className='d-grid gap-2'),
        dbc.Col(dcc.Link('Analyze Sentiments', href='/sentiment-analysis', className='btn btn-primary'), className='d-grid gap-2')
    ], className='mt-4')
], className='container')

# Data Exploration Page
data_exploration_page = html.Div([
    html.H1('Data Exploration', className='mb-4'),
    dcc.Dropdown(
        id='category-dropdown',
        options=[{'label': c, 'value': c} for c in apps_data['Category'].unique()],
        value='ART_AND_DESIGN',
        className='mb-4'
    ),
    html.Div(id='category-proportion-text', className='mb-4', style={'text-align': 'center', 'font-size': '24px'}),
    dcc.Graph(id='rating-pie-chart', className='mb-4'),
    html.H2('Top 10 Apps by Popularity', className='mt-4'),
    dcc.Graph(id='top-apps-graph', className='mb-4')
])

@callback(
    [Output('category-proportion-text', 'children'),
     Output('rating-pie-chart', 'figure'),
     Output('top-apps-graph', 'figure')],
    Input('category-dropdown', 'value')
)
def update_graphs(selected_category):
    filtered_data = apps_data[apps_data['Category'] == selected_category]

    # Proportion de la catégorie dans les téléchargements
    total_installs = apps_data['Installs'].sum()
    category_installs = filtered_data['Installs'].sum()
    category_proportion = (category_installs / total_installs) * 100
    proportion_text = f"The proportion of {selected_category} in total installs is {category_proportion:.2f}%"

    # Camembert de la part des bonnes notes
    total_good_ratings = (apps_data['Rating'] >= 4.0).sum()
    good_ratings = (filtered_data['Rating'] >= 4.0).sum()
    rating_pie = pd.DataFrame({
        'Category': [selected_category, 'Other'],
        'Count': [good_ratings, total_good_ratings - good_ratings]
    })
    fig2 = px.pie(rating_pie, names='Category', values='Count', title=f"Proportion of Good Ratings (>= 4.0) in {selected_category}")

    # Top 10 applications par popularité
    filtered_data['Popularity'] = filtered_data['Rating'] * filtered_data['Installs']
    top_apps = filtered_data.nlargest(10, 'Popularity')
    fig3 = px.bar(top_apps, x='App', y='Popularity', title='Top 10 Apps by Popularity')

    return proportion_text, fig2, fig3

# Application Prediction Page
application_prediction_page = html.Div([
    html.H1('Application Prediction', className='mb-4'),
    html.Div(f"Model Accuracy: {accuracy:.2f}", className='mb-4'),
    dcc.Checklist(
        id='category-checklist',
        options=[{'label': c, 'value': c} for c in apps_data['Category'].unique()],
        value=['TOOLS', 'PRODUCTIVITY'],
        inline=True,
        className='mb-4'
    ),
    html.H2('Top 5 Free Apps', className='mb-4'),
    dcc.Graph(id='top-free-apps-graph', className='mb-4'),
    html.H2('Top 5 Paid Apps', className='mb-4'),
    dcc.Graph(id='top-paid-apps-graph', className='mb-4')
])

@callback(
    [Output('top-free-apps-graph', 'figure'),
     Output('top-paid-apps-graph', 'figure')],
    Input('category-checklist', 'value')
)
def update_prediction_graph(categories):
    filtered_data = apps_data[apps_data['Category'].isin(categories)]
    top_free = filtered_data[filtered_data['Type'] == 0].nlargest(5, 'Rating')
    top_paid = filtered_data[filtered_data['Type'] == 1].nlargest(5, 'Rating')
    
    fig1 = px.bar(top_free, x='App', y='Rating', title='Top 5 Free Apps')
    fig2 = px.bar(top_paid, x='App', y='Rating', title='Top 5 Paid Apps')
    
    return fig1, fig2

# Sentiment Analysis Page
sentiment_analysis_page = html.Div([
    html.H1('Sentiment Analysis', className='mb-4'),
    dcc.Dropdown(
        id='app-dropdown',
        options=[{'label': app, 'value': app} for app in reviews_data['App'].unique()],
        value=reviews_data['App'].unique()[0],
        className='mb-4'
    ),
    dcc.Graph(id='sentiment-graph', className='mb-4'),
    html.Div(id='sentiment-summary', className='mt-4'),
    dcc.Input(id='search-input', type='text', placeholder='Search for an app', className='mb-4'),
    html.Div(id='search-results', className='mt-4')
])

@callback(
    [Output('sentiment-graph', 'figure'),
     Output('sentiment-summary', 'children'),
     Output('search-results', 'children')],
    [Input('app-dropdown', 'value'),
     Input('search-input', 'value')]
)
def update_sentiment(selected_app, search_query):
    data = reviews_data[reviews_data['App'] == selected_app]
    fig = px.histogram(data, x='Sentiment', title=f"Sentiment Distribution for {selected_app}")
    positive = (data['Sentiment'] == 'Positive').sum()
    negative = (data['Sentiment'] == 'Negative').sum()
    neutral = (data['Sentiment'] == 'Neutral').sum()
    summary = f"Positive: {positive}, Negative: {negative}, Neutral: {neutral}"
    
    if search_query:
        search_results = reviews_data[reviews_data['App'].str.contains(search_query, case=False, na=False)]
        search_results_div = html.Div([html.P(f"{row['App']} - {row['Sentiment']}") for index, row in search_results.iterrows()])
    else:
        search_results_div = html.Div()
    
    return fig, summary, search_results_div

# Callbacks pour changer de page
@callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/data-exploration':
        return data_exploration_page
    elif pathname == '/application-prediction':
        return application_prediction_page
    elif pathname == '/sentiment-analysis':
        return sentiment_analysis_page
    else:
        return index_page

if __name__ == '__main__':
    app.run_server(debug=True)
