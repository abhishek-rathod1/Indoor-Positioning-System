from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
import pickle
import pywifi
import time
app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    rf_regressor = pickle.load(f)

# Load the data
df = pd.read_csv('rssi_manual.csv')

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

# Example room dimensions (in meters)
room_width = 28
room_height = 18

# Example obstacles: [x, y, width, height]
obstacles = [[4, 4, 6, 14], [14, 4, 6, 14], [24, 0, 4, 18], [0, 0, 24, 2]]  # Example obstacles (tables)

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

# Function to predict step number based on current Wi-Fi signal strengths
def predict_step(rss_values):
    step_prediction = rf_regressor.predict([rss_values])
    return int(step_prediction[0])

# Function to draw room with obstacles and path
def draw_room_with_obstacles_and_path(width, height, obstacles, step_numbers, start_point, destination_step, path):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Draw walls
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'k-')

    # Draw obstacles
    for obstacle in obstacles:
        ax.add_patch(Rectangle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], color='red'))

    # Set axis limits and labels
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Room Map with Obstacles and Path')

    # Set x-axis ticks from 0 to width with step size 2
    ax.set_xticks(range(0, width + 1, 2))

    # Set y-axis ticks from 0 to height with step size 1
    ax.set_yticks(range(height + 1))

    # Set aspect ratio to equal to ensure 1:1 aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Show grid
    ax.grid(True)

    # Add step numbers
    for step, (x, y) in step_numbers.items():
        # Calculate the center of the block
        block_center_x = x + 1  # Adjust for 2m width
        block_center_y = y + 0.5  # Adjust for 1m height
        ax.text(block_center_x, block_center_y, str(step), ha='center', va='center', fontsize=8, color='black')

    # Add start point (predicted step)
    if start_point:
        x, y = step_numbers[start_point]
        ax.plot(x + 1, y + 0.5, 'co', markersize=10)  # Mark start point in green

    # Add destination step
    if destination_step:
        x, y = step_numbers[destination_step]
        ax.plot(x + 1, y + 0.5, 'ro', markersize=10)  # Mark destination step in red

    # Add path
    if path:
        path_points = [step_numbers[step] for step in path]
        path_x = [x + 1 for x, y in path_points]
        path_y = [y + 0.5 for x, y in path_points]
        ax.plot(path_x, path_y, '-', color='green', linewidth=2)  # Draw path in green as a line

    # Save plot to file
        # Save plot to file
    plt.savefig('static/path_plot.png')
    plt.close()

def get_wifi_signals():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]
    iface.scan()
    time.sleep(2)  # Allow some time for the scan to complete
    scan_results = iface.scan_results()

    networks = []
    for network in scan_results:
        ssid = network.ssid
        signal = network.signal
        networks.append((ssid, signal))
    
    return networks


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get RSS values and destination step from the form
    networks = get_wifi_signals()
    rss_values = {}
    destination_step = int(request.form['destination'])
    for network in networks:
        rss_values[network[0]] = network[1] 
    # Predict the starting step based on RSS values
    rss_value_list = []
    for value in rss_values:
        rss_value_list.append(rss_values[value])
    predicted_step = predict_step(rss_value_list[0:4])

    # Find the path from predicted step to destination step
    path = astar(predicted_step, destination_step, obstacles)

    # Draw the room with obstacles and path
    draw_room_with_obstacles_and_path(room_width, room_height, obstacles, step_numbers, predicted_step, destination_step, path)

    # Render the result template
    return render_template('result.html', predicted_step=predicted_step, path=path, curr_loc=rss_value_list[0:4])

@app.route('/get_rssi')
def get_rssi():
    wifi_signals = get_wifi_signals()
    return jsonify(wifi_signals)

if __name__ == "__main__":
    app.run(debug=True)

