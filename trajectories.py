'''Creating Trajectories on the Rust Map'''

import heapq
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from rust_map_grid import Rust
from copy import deepcopy

class Trajectory:
    '''Creates trajectories for Rust map'''

    def __init__(self):
        self.rust = Rust(75, 75)
        self.current_pos = ()
        self.start = ()
        self.end = ()
        self.trajectory = []
        self.trajectory_length = 0
        self.min_trajectory_length = 15
        self.build_trajectory()

    def heuristic(self, node1, node2):
        '''L2 Norm'''
        return ((node1.position[0] - node2.position[0])**2 + \
                (node1.position[1] - node2.position[1])**2)**(1/2)

    def astar(self, a_grid, start, end):
        '''Calculates shortest path'''
         # Initialize start and end node
        start_node = Node(start)
        end_node = Node(end)
        open_list = []
        closed_list = set()
        # Add the start node to the open list
        heapq.heappush(open_list, start_node)
        while open_list:
            # Get the node with the lowest f cost
            current_node = heapq.heappop(open_list)
            closed_list.add(current_node.position)
            # Check if we have reached the goal
            if current_node.position == end_node.position:
                points = []
                while current_node:
                    pos_x, pos_y = current_node.position[0], current_node.position[1]
                    points.append((pos_x, pos_y))
                    current_node = current_node.parent
                return points[::-1]
            # Generate children
            (x, y) = current_node.position
            neighbors = [
                            (x - 1, y),
                            (x + 1, y),
                            (x, y - 1),
                            (x, y + 1),
                            (x - 1, y - 1),
                            (x - 1, y + 1),
                            (x + 1, y - 1),
                            (x + 1, y + 1)
                        ]
            for next_position in neighbors:
                # Ensure within range
                if (next_position[0] < 0 or next_position[0] >= len(a_grid) or
                    next_position[1] < 0 or next_position[1] >= len(a_grid[0])):
                    continue
                # Ensure no buildings
                if a_grid[next_position[0]][next_position[1]] == 3:
                    continue
                neighbor = Node(next_position, current_node)
                if neighbor.position in closed_list:
                    continue
                # Calculate the f, g, and h values
                neighbor.g = current_node.g + 1
                neighbor.h = self.heuristic(neighbor, end_node)
                neighbor.f = neighbor.g + neighbor.h
                # Check if this path to the neighbor is better
                # if any(open_node.position == neighbor.position and \
                #        open_node.g < neighbor.g for open_node in open_list):
                #     continue
                heapq.heappush(open_list, neighbor)
        return None  # Return None if no path is found

    def build_trajectory(self):
        '''Contruction of start and end positions'''
        self.rust.show_grid()
        while self.start == () or self.end == ():
            try:
                if self.start == ():
                    self.start = random.choice(self.rust.positions)  # Randomly choose start
                    # print(f'Start: {self.start}')
                    if self.start in self.rust.buildings:  # Check if start is in a building
                        self.start = ()
                if self.end == ():   # Randomly choose end
                    self.end = random.choice(self.rust.positions)
                    # print(f'End: {self.end}')
                    # Check if end is in a building or is start or is less than 15 squares away (ensures mid to long trajectories)
                    if  self.end in self.rust.buildings or \
                        self.end == self.start or \
                        (abs(self.start[0] - self.end[0])) < self.min_trajectory_length or \
                        (abs(self.start[1] - self.end[1])) < self.min_trajectory_length:
                        self.end = ()
            except IndexError:  # if out of index, just reset the start and end positions and try again
                self.start = ()
                self.end = ()
        self.current_pos = self.start

    def move_evader(self):
        '''Moves the evader through the environment via A*'''
        if self.current_pos == self.end:
            return
          # Stop if the evader has reached the destination
        self.trajectory = self.astar(self.rust.grid, self.current_pos, self.end)
        self.trajectory_length = len(self.trajectory)
        # for p in self.trajectory:
        #     self.rust.grid[p[0]][p[1]] = 1
        return self.rust.grid

class Node:
    '''Node class for A*'''

    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start node
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f

class Dropout:
    '''Drops random points of the trajectory and classifies them from (0, 1]
       based on how close they are to the current position
    '''

    def __init__(self):
        self.trajectory = Trajectory()
        self.rust = Rust(75, 75)
        self.updated_trajectory = []
        self.gradients = []
        self.updated_gradients = []
        self.lookaheads = []
        self.all_grids = []


    def remove_positions(self, grid, trajectory):
        '''Remove positions from the current trajectory'''
        updated_trajectory = []
        updated_gradients = []
        random_difference = abs(np.random.choice(15)-np.random.choice(10))
        tail_vals = len(trajectory) - random_difference
        positions_to_remove = list(np.random.randint(2, size=tail_vals)) # randomizes tailend subsection
        len_pos_to_rem = len(positions_to_remove)
        while len_pos_to_rem < len(trajectory): # adds 1's to front of trajectory gradients
            positions_to_remove.append(1)
            len_pos_to_rem += 1
        path = trajectory
        for ind, binary in enumerate(positions_to_remove):  # filters all 0 valued indexes
            if binary == 1:
                updated_trajectory.append(path[ind])
                updated_gradients.append(self.gradients[ind])
        return updated_trajectory, updated_gradients, grid

    def snapshots(self):
        '''Assigns gradient values to the updated evader trajectory'''
        # t, k = 5, 7     # t is the amount of timesteps, k is the look ahead from t
        T = len(self.trajectory.trajectory)
        exponential = lambda x: 1.05**(-x)
        gradients = []
        # NOT BEING USED RIGHT NOW, BUT UNCOMMENT AND EDIT TO FIND LOOKAHEADS
        # =======================================================
        # for ind, traj in enumerate(self.trajectory.trajectory):
        #     # Add all lookahead positions, will filter them after
        #     try:
        #         self.lookaheads.append(self.trajectory.trajectory[ind+k])
        #     except IndexError:
        #         self.lookaheads.append(self.trajectory.trajectory[-1])
         # ======================================================
        for t, _ in enumerate(self.trajectory.trajectory):
            gradients.append(exponential(T-t))
        return self.trajectory.trajectory, gradients


    def dropout(self, current_grid, times=None):
        '''Outputs map of evader with new gradient trajectory'''
        set_grid = deepcopy(current_grid)
        all_grids = []
        all_trajes = []
        for _ in range(times):
            # self.updated_gradients = self.gradients
            trajectory, self.gradients = self.snapshots()
            self.updated_trajectory, \
                self.updated_gradients, \
                    grid = self.remove_positions(current_grid, trajectory)
                    # grid = self.remove_positions(current_grid, self.trajectory.trajectory)
            
            # self.updated_trajectory, self.updated_gradients = self.snapshots(trajectory)
            print(f'Current trajectory: {self.updated_trajectory}')
            print(f'Gradients: {self.updated_gradients}')
            for ind, ut in enumerate(self.updated_trajectory):
                grid[ut[0]][ut[1]] = self.updated_gradients[ind]
            all_grids.append(grid)
            all_trajes.append(self.updated_trajectory)
            grid = set_grid
        return all_grids, all_trajes

# =============================================
# For creating a dataset, uncomment and run this
# =============================================
grids, trajs, counter = [], [], 0
for _ in range(50):
    trajectory = Dropout()
    grid = trajectory.trajectory.move_evader()
    # trajs.append(trajectory.trajectory.trajectory)
    path, traj = trajectory.dropout(grid, 1)
    grids.append(path)
    trajs.append(traj)
    counter += 1
    print(f'trajectory {counter} complete')

traje = {"Trajectories":grids, "List":trajs}
trajes_del_bano = pd.DataFrame(traje)
trajes_del_bano.to_csv('/Users/dbosty/Desktop/gridWorld/trajectories.csv')

# =============================================
# For ease of viewing, uncomment and plot this
# =============================================
# trajectory = Dropout()
# rust_grid = trajectory.trajectory.move_evader()
# path, trajes = trajectory.dropout(rust_grid, 1)
# print(f'Length of trajes: {len(trajes)}')

# all_trajes = {}
# for ind, traj in enumerate(trajes):
#     if ind in all_trajes:
#         all_trajes[ind] += 1
#     else:
#         all_trajes[ind] = 1


# print(f'All Trajectories: {all_trajes}')

# ========================================================================================
#       Last trajectory and gradients plus number of all unique and counts of each
# ========================================================================================
# Current trajectory: [(47, 68), (46, 67), (44, 65), (44, 64), (44, 63), (42, 61), (41, 60),
#                      (39, 58), (38, 57), (35, 54), (30, 51), (28, 51), (26, 51), (25, 51),
#                      (24, 50), (23, 49), (20, 46), (18, 44), (16, 42), (14, 40), (14, 39),
#                      (14, 38), (14, 37), (14, 36), (14, 35), (14, 34), (14, 33), (14, 31),
#                      (14, 30), (14, 29), (14, 28), (14, 27), (14, 26)]
# Gradients: [0.6080388246889494, 0.6141192129358389, 0.6264630091158493, 0.6327276392070078,
#             0.6390549155990779, 0.6518999194026194, 0.6584189185966455, 0.6716531388604381,
#             0.6783696702490425, 0.6989249496272587, 0.7345771463238853, 0.7493421469649953,
#             0.7644039241189917, 0.7720479633601817, 0.7797684429937835, 0.7875661274237213,
#             0.8114301686507875, 0.8277399150406684, 0.8443774873329858, 0.8613494748283789,
#             0.8699629695766626, 0.8786625992724293, 0.8874492252651536, 0.8963237175178052,
#             0.9052869546929833, 0.914339824239913, 0.9234832224823122, 0.9420452352542067,
#             0.9514656876067488, 0.9609803444828162, 0.9705901479276444, 0.9802960494069208,
#             0.9900990099009901]
# Length of trajes: 100
# All Trajectories: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
#                   10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1,
#                   20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1,
#                   30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1,
#                   40: 1, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 1, 49: 1,
#                   50: 1, 51: 1, 52: 1, 53: 1, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1, 59: 1,
#                   60: 1, 61: 1, 62: 1, 63: 1, 64: 1, 65: 1, 66: 1, 67: 1, 68: 1, 69: 1,
#                   70: 1, 71: 1, 72: 1, 73: 1, 74: 1, 75: 1, 76: 1, 77: 1, 78: 1, 79: 1,
#                   80: 1, 81: 1, 82: 1, 83: 1, 84: 1, 85: 1, 86: 1, 87: 1, 88: 1, 89: 1,
#                   90: 1, 91: 1, 92: 1, 93: 1, 94: 1, 95: 1, 96: 1, 97: 1, 98: 1, 99: 1}

# ============================================
#       Uncomment to visualize Rust map
# ============================================
# def plot_trajes(path, trajes):
#     'Plot trajectories (still no bueno as of right now, only plots first one)'
#     cmap_1 = mpl.colors.ListedColormap(['gainsboro',      # Blank   --> Value = 0
#                                         'red',            # Evader  --> Value = (0, 1]
#                                         'green',          # Pursuer --> Value = (1, 2]
#                                         'black',          # Inacc   --> Value = 3
#                                         'darkseagreen',   # Acc_pv  --> Value = 4
#                                         'plum',           # Acc_nv  --> Value = 5
#                                         'skyblue'         # Acc_v   --> Value = 6
#                                     ])

#     fig, ax = plt.subplots()
#     for p, t in zip(path, trajes):
#         cmap = plt.get_cmap('Reds')
#         norm = plt.Normalize(min(trajectory.gradients), max(trajectory.gradients))
#         for ind, traj in enumerate(t):
#             color = cmap(norm(trajectory.gradients[ind]))
#             ax.plot(traj[1], traj[0], 's', color=color, markersize=3, mouseover=True)
#         cax = ax.imshow(p, cmap=cmap_1, mouseover=True)
#         print(p)
#         trajectory.updated_gradients = []
#         trajectory.updated_trajectory = []
#         plt.axis('off')
#         plt.show()

# plot_trajes(path, trajes)

