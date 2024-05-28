import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
import pickle
with open('model.pkl','rb') as f:
    rf_regressor = pickle.load(f)

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