import numpy as np
import matplotlib.pyplot as plt
from ICMDP import *
from scipy.optimize import minimize
# end of imports

# some fuctions to be used later:

# do action function - returns the next x and speed for the given state and action:


def do_action(state,action, min_speed, max_speed, min_x, max_x, step_size):
    speed = state.speed
    my_x = state.x
    
    # move left:
    if action == 1:
        if my_x - step_size >= min_x:
            my_x = my_x - step_size
        else:
            my_x = min_x
    
    # move right:
    elif action == 2:
        if my_x + step_size <= max_x:
            my_x = my_x + step_size
        else:
            my_x = max_x
    
    # increase speed:
    elif action == 3:
        if speed < max_speed:
            speed = speed + 1
    
    # decrease speed:
    elif action == 4:
        if speed > min_speed:
            speed = speed - 1

    return [speed,my_x]

# find next state function - finds the next possible states for a given state and action:


def find_next_state(state,action,states_inv):
    [new_speed, new_x] = do_action(state,action,speeds_num[0],speeds_num[-1], left_bound,right_bound,5)

    # check if this is the first state:
    if (states_inv[','.join(str(elem) for elem in (state.as_list()))] == 0):

        # for first state - special control for speed, to choose the speed for the rest of the game:
        state_vec = []
        for x in other_car_x:
            if action == 0:
                init_speed = state.speed
            elif action == 1:
                init_speed = state.speed - 1
            elif action == 2:
                init_speed = state.speed + 1

            # insert a car in a random position:
            new_state = states_inv[ ','.join(str(elem) for elem in [init_speed,state.x,x,10])]
            state_vec.append(new_state)
        return state_vec

    # check if need to insert a new car in a random place and remove the old one:
    elif (state.other_car[1] + displace[state.speed] >= height - 10 + my_car_size[0]):
        state_vec = []
        for x in other_car_x:
            new_state = states_inv[ ','.join(str(elem) for elem in [new_speed,new_x,x,10])]
            state_vec.append(new_state)
        return state_vec

    # no new car needed - deterministic next state:
    else:
        new_state = states_inv[','.join(str(elem) for elem in [new_speed, new_x ,state.other_car[0],state.other_car[1] + displace[state.speed]])]
        return new_state


# features:
# 1. speed
# 2. changing lane


# actions:
# 0 - do nothing
# 1 -  move left
# 2 - move right

# parameters:
# right-left step siize:
step_size = 5

# boundaries of the frame
left_bound = 120
right_bound = 200
height = 180
width = 300
bottom_bound = height

# boundaries of the road:
road_left_bound = left_bound + 20
road_right_bound = right_bound - 20

# car size, width is half of the width in the format "[length,width]":
my_car_size = [40, 10]

# the y position of the player's car (stayes fixed during the game):
my_y = height - 10 - my_car_size[0]

# initiate the speed feature values, displace for each speed and numbering:
displace = [20, 40, 80]
speeds_num = [0,1,2]
speed_feature_vals = [0.5,0.75,1]

# calculate the different possible x positions of the player's car:
my_x = []
for x in range(left_bound,right_bound + step_size,step_size):
    my_x.append(x)

# the lanes locations:
lanes = [140,160,180] # the x coordinates of the lanes

# build other_car:
other_car_length = 40
other_car_width = 10
other_car_x = lanes # to lower complexity
other_car_y = [] # the legal y coordinates of the other cars
for i in range(10):
    other_car_y.append(20*i + 10)

other_car = [] # format: [x coordinate, y coordinate]
for x in other_car_x:
    for y in other_car_y:
        other_car.append([x,y])

# build actions:
# 0 - do nothing
# 1 - move left
# 2 - move right
actions = [0,1,2]

# initiate staes array and state to index (states_inv) dictionary:
states = []
states_inv = {}

# initiate features:
F = Features(dim_features=3)

# add first  state:
states.append(State(1,160,[-1,-1]))
states_inv[','.join(str(elem) for elem in (states[0].as_list()))] = 0
F.add_feature(feature=[0.75,0.5,0.5])

# build the whole state - feature mapping:
for speed in speeds_num:
    for x in my_x:
        for other_x in other_car_x:
            for other_y in other_car_y:
                states.append(State(speed,x,[other_x,other_y]))
                states_inv[','.join(str(elem) for elem in (states[len(states)-1].as_list()))] = len(states) - 1

                # add speed feature value:
                speed_val = speed_feature_vals[speed]
                
                # check collision:
                if (other_y > my_y) and (other_y - other_car_length < my_y + my_car_size[0]) and (other_x + other_car_width > x - my_car_size[1]) and (other_x - other_car_width < x + my_car_size[1]):
                    collision_val = 0
                else:
                    collision_val = 0.5

                # check off-road:
                if (x < road_left_bound) or (x  > road_right_bound):
                    off_road_val = 0
                else:
                    off_road_val = 0.5

                F.add_feature(feature=[speed_val,collision_val,off_road_val])


# setup transitions:
THETA = Transitions(num_states=len(states), num_actions=len(actions))
curr_state = 0
for state in states:
    for action in actions:
        # find next state:
        new_state = find_next_state(state,action,states_inv)

        # if there is more than 1 possible next state, calculate uniform distribution between the possibilities:
        if isinstance(new_state, list):
            num_states = len(new_state)
            trans = 1.0/num_states
            for i in range(num_states):
                THETA.set_trans(curr_state,action,new_state[i],trans)
        
        # deterministic next state:
        else:
            THETA.set_trans(curr_state,action,new_state,1)

    curr_state = curr_state + 1


# initiate inputs for constraint generation:
E_list = []
contexts = []
weights = []

# initiate an ICMDP object:
mdp = ICMDP()

# set the catculated features and transitions:
mdp.set_F(F)
mdp.set_THETA(THETA)

# set "real" W:
mdp.set_W(np.asarray([[-0.3,0.3,0.4],[0.4,-0.3,0.3],[0.3,0.4,-0.3]]))

# initiate given contexts:
Conts = [[0.3,0.3,0.4],[0.2,0.1,0.7],[0.5,0.1,0.4],[0.1,0.5,0.4],[0.25,0.25,0.5],[0.2,0.5,0.3],[0.05,0.3,0.65],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.8,0.1,0.1]]

# for each context, caculate the expert feature expectations and append to E_list, build contexts list of arrays:
for cont in Conts:
    mdp.set_C(np.asarray(cont))
    a = mdp.solve_CMDP(gamma=0.9,tol=0.0001,flag='init')
    E_list.append(a.M)
    contexts.append(np.asarray(cont))
    weights.append(1)


# ICMDP solving methods:

#################################################################
# solve using constraint generation:
# linear:
# W_sol = mdp.solve_ICMDP(gamma=0.9,contexts=contexts, expert_feature_exp= E_list, weights=weights, algo='linear')
# norm2:
W_sol = mdp.solve_ICMDP(gamma=0.9,contexts=contexts, expert_feature_exp= E_list, weights=weights, algo='norm2')
print(W_sol)
exit()
#################################################################
# solve using optimization tool on minimizing the summed value difference over W between the expert and the optimal policy under W:
func = lambda W:  mdp.feature_expectations_opt(W= W,gamma = 0.9,contexts=contexts,expert_mus=E_list,mode='value')
W_0 = np.ones(9)
W_0 = W_0/3
res = minimize(func,W_0, method='Nelder-Mead',options={'adaptive': False})
W_mid = res.x
W_sol = np.zeros([3,3])
for i in range(3):
    W_sol[i,:] = W_mid[i*3 : i*3+3]
print(W_sol)
exit()
#################################################################
# solve using optimization tool on minimizing the summed difference norm between the expert and the optimal policy under W:
func = lambda W:  mdp.feature_expectations_opt(W= W,gamma = 0.9,contexts=contexts,expert_mus=E_list,mode='mu')
W_0 = np.ones(9)
W_0 = W_0/3
res = minimize(func,W_mid,method='Nelder-Mead',options={'adaptive': False})
W_mid = res.x
W_sol = np.zeros([3,3])
for i in range(3):
    W_sol[i,:] = W_mid[i*3 : i*3+3]
print(W_sol)
exit()
#################################################################
