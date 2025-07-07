import pygame 

import numpy as np
import numpy.linalg as lg


import matplotlib.pyplot as plt






# curr = pygame.mouse.get_pos()

# if prev == None:
#     prev = curr


# if prev_press_state == None:
#     prev_press_state = 0

# curr_press_state = pygame.get_pressed(3)[0]


# if curr_press_state == 1 and prev_press_state == 0:

    ##grab cube to set hrotation handle
    # draw line to center of rotation


# def get_rel_omega(p1_rel,p2_rel):

i = np.array([1,0,0])
j = np.array([0,1,0])
k = np.array([0,0,1])



def simple_mouse_move(p1,p2,j_rel,k_rel,c,center_of_rotation = None):


    
    # c is center of screen

    #center_of_screen = profjection of c onto viewing plane.

    p1 = np.array(p1)
    p2 = np.array(p2)


    p_change = (p2 - p1)



    p1_space = c + p1[0]*j_rel + p1[1]*k_rel
    p2_space = c + p2[0]*j_rel + p2[1]*k_rel

    # p_space_change = p2_space - p1_space

    # delta_theta = np.arcsin(p_change[0] /lg.norm( p1_space + p2[0]*j_rel) )

    # delta_phi =  np.arcsin(p_change[1] /lg.norm( p2_space) )


    # delta_theta = p_change[0]/1000

    # delta_phi =  p_change[1]/1000


    delta_theta = np.arcsin(p_change[0] /lg.norm( p1_space + p2[0]*j_rel) )/2

    delta_phi =  np.arcsin(p_change[1] /lg.norm( p2_space) )/2

    # delta_theta = np.arcsin(p_change[0] /lg.norm( p1_space + p2[0]*j_rel) )

    # delta_phi =  np.arcsin(p_change[1] /lg.norm( p1_space + p2[0]*j_rel) )




    # print(delta_theta,delta_phi)

    return delta_theta, delta_phi
















    # p2_space_j_rel = p2[0]* j_rel +  c
    # p2_space_j_rel_len = lg.norm(p2_space_j_rel)

    # p2_j_rel_len = lg.norm(p2[0]* j_rel)

    # p2_space_k_rel =  p2[1]* k_rel + c
    # p2_space_k_rel_len = lg.norm(p2_space_k_rel)

    # p2_k_rel_len = lg.norm(p2[1]* k_rel)




    # delta_theta = np.arcsin(p2_j_rel_len/p2_space_j_rel_len)

    # delta_phi = np.arcsin(p2_k_rel_len/p2_space_k_rel_len)

    # return delta_theta, delta_phi









    # # v1 = p1 - center_of_rotation
    # # v2 = p2 - center_of_rotation




    


    # change_x = p2[0] - p1[0]
    # change_y = p2[1] - p1[1]

    # return np.arcsin(change_x), np.arcsin(change_y)




# curr = pygame.mouse.get_pos()

# if prev == None:
#     prev = curr


# if prev_press_state == None:
#     prev_press_state = 0

# curr_press_state = pygame.get_pressed(3)[0]


# if curr_press_state == 1 and prev_press_state == 0:

#     delta_theta, delta_phi = simple_mouse_move(curr,prev)



    

    #grab cube to set hrotation handle
    # draw line to center of rotation



    

























pygame.mouse.get_pressed

pygame.mouse.get_pos

