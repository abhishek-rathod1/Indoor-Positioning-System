import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import heapq
import numpy as np
import pickle

# Load the data
# Load the model
with open('model.pkl', 'rb') as f:
    rf_regressor = pickle.load(f)

# Load the data
df = pd.read_csv('rssi_manual.csv')

# Function to predict step number based on current Wi-Fi signal strengths
def predict_step(rss_values):
    step_prediction = rf_regressor.predict([rss_values])
    return int(step_prediction[0])

# Heuristic function for A*
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Define function for A* pathfinding algorithm
def astar(start, goal, obstacles):
    def is_in_obstacle(x, y):
        for obs in obstacles:
            if obs[0] <= x < obs[0] + obs[2] and obs[1] <= y < obs[1] + obs[3]:
                return True
        return False

    start_coords = step_numbers[start]
    goal_coords = step_numbers[goal]

    open_set = []
    heapq.heappush(open_set, (0, start_coords))

    came_from = {}
    g_score = {step: float('inf') for step in step_numbers}
    g_score[start] = 0
    f_score = {step: float('inf') for step in step_numbers}
    f_score[start] = heuristic(start_coords, goal_coords)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_coords:
            path = []
            while current in came_from:
                step = next(step for step, coords in step_numbers.items() if coords == current)
                path.append(step)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        current_step = next(step for step, coords in step_numbers.items() if coords == current)
        current_x, current_y = current

        # Possible moves: up, down, left, right
        neighbors = [
            (current_x + 2, current_y), (current_x - 2, current_y),
            (current_x, current_y + 1), (current_x, current_y - 1)
        ]

        for neighbor_x, neighbor_y in neighbors:
            if 0 <= neighbor_x < room_width and 0 <= neighbor_y < room_height and not is_in_obstacle(neighbor_x, neighbor_y):
                neighbor_step = next((step for step, coords in step_numbers.items() if coords == (neighbor_x, neighbor_y)), None)
                if neighbor_step is None:
                    continue
                tentative_g_score = g_score[current_step] + 1

                if tentative_g_score < g_score[neighbor_step]:
                    came_from[(neighbor_x, neighbor_y)] = current
                    g_score[neighbor_step] = tentative_g_score
                    f_score[neighbor_step] = g_score[neighbor_step] + heuristic((neighbor_x, neighbor_y), goal_coords)
                    if (neighbor_x, neighbor_y) not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor_step], (neighbor_x, neighbor_y)))

    return []

# Example room dimensions (in meters)
room_width = 28
room_height = 18

# Example obstacles: [x, y, width, height]
obstacles = [[4, 4, 6, 14], [14, 4, 6, 14], [24, 0, 4, 18], [0, 0, 24, 2]]  # Example obstacles (tables)

# Define step numbers and their corresponding coordinates
step_numbers = {
    1: (22,17),2: (20,17),3: (22,16),4: (20,16),5: (22,15),6: (20,15),7: (22,14),8: (20,14),9: (22,13),10: (20,13),
    11: (22,12),12: (20,12),13: (22,11),14: (20,11),15: (22,10), 16: (20,10), 17:(22,9) ,18: (20,9), 19: (22,8), 20: (20,8),
    21: (22,7),22: (20,7),23: (22,6),24: (20,6),25: (22,5), 26: (20,5), 27:(22,4) ,28: (20,4), 29: (22,3), 30: (20,3), 31:(22,2) ,32:(20,2),
    33:(18,3), 34:(18,2), 35:(16,3), 36:(16,2), 37:(14,3), 38:(14,2), 39:(12,3), 40:(12,2), 41:(10,3), 42:(10,2), 43:(8,3), 44:(8,2), 45:(6,3), 46:(6,2), 47:(4,3), 48:(4,2),
    49:(2,3), 50:(2,2), 51:(0,3), 52:(0,2), 53:(2,4), 54:(0,4), 55:(2,5), 56:(0,5), 57:(2,6), 58:(0,6), 59:(2,7), 60:(0,7), 61:(2,8), 62:(0,8),
    63:(2,9), 64:(0,9), 65:(2,10), 66:(0,10), 67:(2,11), 68:(0,11), 69:(2,12), 70:(0,12),
    71:(2,13), 72:(0,13),73:(2,14), 74:(0,14), 75:(2,15), 76:(0,15), 77:(2,16), 78:(0,16), 79:(2,17), 80:(0,17)
}

# Function to update the plot
def update_plot(current_step, destination_step, path):
    fig = go.Figure()

    # Draw walls
    fig.add_trace(go.Scatter(
        x=[0, room_width, room_width, 0, 0],
        y=[0, 0, room_height, room_height, 0],
        mode='lines',
        line=dict(color='black')
    ))

    # Draw obstacles
    for obstacle in obstacles:
        fig.add_trace(go.Scatter(
            x=[obstacle[0], obstacle[0] + obstacle[2], obstacle[0] + obstacle[2], obstacle[0], obstacle[0]],
            y=[obstacle[1], obstacle[1], obstacle[1] + obstacle[3], obstacle[1] + obstacle[3], obstacle[1]],
            fill='toself',
            fillcolor='red',
            mode='lines',
            line=dict(color='red')
        ))

    # Add step numbers
    for step, (x, y) in step_numbers.items():
        fig.add_trace(go.Scatter(
            x=[x + 1],
            y=[y + 0.5],
            text=[str(step)],
            mode='text',
            textposition='middle center',
            hoverinfo='text'
        ))

    # Add current position (predicted step)
    if current_step:
        x, y = step_numbers[current_step]
        fig.add_trace(go.Scatter(
            x=[x + 1],
            y=[y + 0.5],
            mode='markers',
            marker=dict(color='cyan', size=10),
            name='Current Position'
        ))

    # Add destination step
    if destination_step:
        x, y = step_numbers[destination_step]
        fig.add_trace(go.Scatter(
            x=[x + 1],
            y=[y + 0.5],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Destination'
        ))

    # Add path
    if path:
        path_points = [step_numbers[step] for step in path]
        path_x = [x + 1 for x, y in path_points]
        path_y = [y + 0.5 for x, y in path_points]
        fig.add_trace(go.Scatter(
            x=path_x,
            y=path_y,
            mode='lines+markers',
            line=dict(color='green', width=2),
            name='Path'
        ))

    # Update layout
    fig.update_layout(
        title='Room Map with Obstacles and Path',
        xaxis_title='X (meters)',
        yaxis_title='Y (meters)',
        xaxis=dict(tickmode='array', tickvals=list(range(0, room_width + 1, 2))),
        yaxis=dict(tickmode='array', tickvals=list(range(0, room_height + 1, 1))),
        showlegend=False,
        width=800,
        height=600
    )

    return fig

# Detect current position
current_wifi_signal_strengths = list(map(int, input("Enter the RSSI values: ").strip().split()))[:4]  # Manually input RSSI values
predicted_step = predict_step(current_wifi_signal_strengths)

# Initial plot without path
fig = update_plot(predicted_step, None, None)

# Create Dash app
app = dash.Dash(_name_)

app.layout = html.Div([
    dcc.Graph(id='room-map', figure=fig)
])

@app.callback(
    Output('room-map', 'figure'),
    Input('room-map', 'clickData')
)
def update_on_click(clickData):
    if clickData is None:
        return update_plot(predicted_step, None, None)

    point = clickData['points'][0]
    x, y = point['x'], point['y']

    # Find the closest step to the clicked coordinates
    closest_step = min(step_numbers.keys(), key=lambda step: (step_numbers[step][0] + 1 - x) ** 2 + (step_numbers[step][1] + 0.5 - y) ** 2)

    path = astar(predicted_step, closest_step, obstacles)
    return update_plot(predicted_step, closest_step, path)

if _name_ == '_main_':
    app.run_server(debug=True)