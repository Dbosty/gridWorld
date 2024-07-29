'''Creating grid world for Rust map'''

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class Rust:
    '''Creates the Rust map. Building values are as follows:
`       - Open Position: 0
        - Inaccessible: 3
        - Accessible / Partially Viewable: 4
        - Accessible / Not Viewable: 5
        - Accessible / Viewable: 6   
    '''

    def __init__(self, m, n):
        self.m = m              # Width of grid
        self.n = n              # Length of grid
        self.grid = []          # Grid world of size 75 x 75 with values [0, 6]
        self.positions = []     # Grid world of size 75 x 75 with values (x, y)
        self.b_inacc = []       # Inaccessible buildings
        self.b_acc_v = []       # Accessible and viewable buildings
        self.b_acc_pv = []      # Accessible and partially viewable buildings
        self.b_acc_nv = []      # Accessible and not viewable buildings
        self.buildings = []     # List of all buildings positions
        self.create_grid(m, n)

        FILEPATH = '/Users/dbosty/Desktop/gridWorld/rust_75x75.yaml'
        with open(FILEPATH, 'r') as f:
            config = yaml.safe_load(f)
        
        self.inaccessible = config['buildings']['inaccessible']
        self.access_part_view = config['buildings']['accessible_part_viewable']
        self.access_no_view = config['buildings']['accessible_no_viewable']
        self.access_view = config['buildings']['accessible_viewable']

    def create_grid(self, m, n):
        '''Create an empty grid world with 0 values in each grid square'''
        for ind_m in range(m):
            to_grid = []
            for ind_n in range(n):
                to_grid.append(0)
                self.positions.append((ind_m, ind_n))
            self.grid.append(to_grid)

    def draw_building(self, ul, ur, bl, br, value):
        '''Draws the building with the correct shape and assigns proper value'''
        params = [ul, ur, bl, br]
        min_x = min(node[0] for node in params)
        max_x = max(node[0] for node in params)
        min_y = min(node[1] for node in params)
        max_y = max(node[1] for node in params)
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                try:
                    # I don't want to overwrite values unless it's the same number then it doesn't matter
                    if (x, y) in self.buildings and self.grid[x][y] != value:
                        raise ValueError
                    if value == 3:
                        self.b_inacc.append((x, y))
                    if value == 4:
                        self.b_acc_pv.append((x, y))
                    if value == 5:
                        self.b_acc_nv.append((x, y))
                    if value == 6:
                        self.b_acc_v.append((x, y))
                except ValueError:
                    print((x, y), self.grid[x][y])
                self.grid[x][y] = value
                self.buildings.append((x, y))

    def define_inaccessible(self):
        '''Dynamically create the innaccessible buildings in the Rust map.
           No on can enter these buildings or barriers. 
           
           Value = 3
        '''
        # List of all inaccessible buildings
        buildings = self.inaccessible
        for b in buildings:
            # upper left, upper right, bottom left, bottom right
            ul, ur, bl, br = b[0], b[1], b[2], b[3]
            self.draw_building(ul, ur, bl, br, 3)


    def define_access_part_view(self):
        '''Dynamically create the accessible and partially viewable buildings in the Rust map.
           Evader can access the building or shelter, LLP can see but not enter, HLP cannot see.

           Value = 4
        '''
        # List of all accessible and partially viewable buildings
        buildings = self.access_part_view
        for b in buildings:
            # upper left, upper right, bottom left, bottom right
            ul, ur, bl, br = b[0], b[1], b[2], b[3]
            self.draw_building(ul, ur, bl, br, 4)

    def define_access_not_view(self):
        '''Dynamically create the accessible and not viewable buildings in the Rust map.
           Evader can access but neither the LLP nor the HLP can view.

           Value = 5
        '''
        # List of all accessible and not viewable buildings
        buildings = self.access_no_view
        for b in buildings:
            # upper left, upper right, bottom left, bottom right
            ul, ur, bl, br = b[0], b[1], b[2], b[3]
            self.draw_building(ul, ur, bl, br, 5)

    def define_access_view(self):
        '''Dynamically create the accessible and viewable buildings in the Rust map.
           Evader and LLP can enter, HLP can see.

           Value = 6
        '''
        # List of all accessible and viewable buildings
        buildings = self.access_view
        for b in buildings:
            # upper left, upper right, bottom left, bottom right
            ul, ur, bl, br = b[0], b[1], b[2], b[3]
            self.draw_building(ul, ur, bl, br, 6)

    def show_grid(self):
        '''Return the grid world up to this point'''
        self.define_inaccessible()
        self.define_access_part_view()
        self.define_access_not_view()
        self.define_access_view()
        return self.grid

rust_grid = Rust(75, 75)
rust_map = rust_grid.show_grid()

# ============================================
#       Uncomment to visualize Rust map
# ============================================
# cmap = mpl.colors.ListedColormap(['gainsboro',      # Blank   --> Value = 0
#                                   'red',            # Evader  --> Value = (0, 1]
#                                   'green',          # Pursuer --> Value = (1, 2]
#                                   'black',          # Inacc   --> Value = 3
#                                   'darkseagreen',   # Acc_pv  --> Value = 4
#                                   'plum',           # Acc_nv  --> Value = 5
#                                   'skyblue'         # Acc_v   --> Value = 6
#                                 ])
# plt.imshow(rust_map, cmap=cmap)
# plt.axis('off')
# plt.show()
