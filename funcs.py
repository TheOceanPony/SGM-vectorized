import numpy as np
from numba import njit


def add_frame(img, brdr_size=1, value=0):
    # Adds a border around the img
    return np.pad(img, pad_width=brdr_size, mode='constant', constant_values=value)


@njit
def initialize_H(height, width, D, img_L, img_R):
    H = np.zeros((height, width, len(D)), dtype=np.float32)
    for y in range(0, height):
        for x in range(0, width):
            for d_ind, d in enumerate(D):
                if 0<= y-d[0] < height and 0<= x-d[1] <width:
                    H[y, x, d_ind] = abs( img_L[y,x] - img_R[y - d[0], x - d[1]] )
                else:
                    H[y, x, d_ind] = np.inf # Weird results
    return H


@njit
def initialize_G(D, alpha=1):
    
    G = np.zeros((len(D), len(D)), dtype=np.float32)
    for d1_ind, d1 in enumerate(D):
        for d2_ind, d2 in enumerate(D):
            G[d1_ind, d2_ind] = (d1[0] - d2[0])**2 + (d1[1] - d2[1])**2
            
    return G


@njit
def init_left_part(height, width, D, H, G):

    Li = np.zeros((height, width, len(D)), dtype=np.float32)

    for y in range(0, height):
        for x in range(0, width):
            for d_ind, d in enumerate(D):
                Li[y, x, d_ind] = left(y, x, d_ind, Li, D, H, G)
    return Li

@njit
def left(y, x, d_ind, Li, D, H, G):

    if x == 0:
        return 0
    else:
        minl = np.inf

        for d2_ind, d2 in enumerate(D):
            temp = Li[y, x-1, d2_ind] + H[y, x-1, d2_ind] + G[d_ind, d2_ind]
            if temp < minl:
                minl = temp

        return minl


@njit
def init_right_part(height, width, D, H, G):

    Ri = np.zeros((height, width, len(D)), dtype=np.float32)

    for y in range(height-1, -1):
        for x in range(width-1, -1):
            for d_ind, d in enumerate(D):
                Ri[y, x, d_ind] = right(y, x, d_ind, Ri, D, H, G)
    return Ri

@njit
def right(y, x, d_ind, Ri, D, H, G):

    if x == width-1:
        return 0
    else:
        minl = np.inf

        for d2_ind, d2 in enumerate(D):
            temp = Ri[y, x+1, d2_ind] + H[y, x+1, d2_ind] + G[d_ind, d2_ind]
            if temp < minl:
                minl = temp

        return minl


@njit
def init_top_part(height, width, D, H, G):

    Ui = np.zeros((height, width, len(D)), dtype=np.float32)

    for y in range(0, height):
        for x in range(0, width):
            for d_ind, d in enumerate(D):
                Ui[y, x, d_ind] = up(y, x, d_ind, Ui, D, H, G)
    return Ui

@njit
def up(y, x, d_ind, Ui, D, H, G):

    if y == 0:
        return 0
    else:
        minl = np.inf

        for d2_ind, d2 in enumerate(D):
            temp = Ui[y-1, x, d2_ind] + H[y-1, x, d2_ind] + G[d_ind, d2_ind]
            if temp < minl:
                minl = temp

        return minl


@njit
def init_bottom_part(height, width, D, H, G):

    Di = np.zeros((height, width, len(D)), dtype=np.float32)

    for y in range(height-1, -1):
        for x in range(width-1, -1):
            for d_ind, d in enumerate(D):
                Di[y, x, d_ind] = right(y, x, d_ind, Di, D, H, G)
    return Di

@njit
def right(y, x, d_ind, Di, D, H, G):

    if x == height-1:
        return 0
    else:
        minl = np.inf

        for d2_ind, d2 in enumerate(D):
            temp = Di[y+1, x, d2_ind] + H[y+1, x, d2_ind] + G[d_ind, d2_ind]
            if temp < minl:
                minl = temp

        return minl


#@njit
def reconstruct(height, width, Li, Ri, Ui, Di, H):
    
    Res = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(0, height):
        for x in range(0, width):
            best_score = np.inf
            best_d = None
            for d_ind, d in enumerate(D):
                temp = Li[y, x, d_ind] + Ui[y, x, d_ind] + H[y, x, d_ind] + Ri[y, x, d_ind] + Di[y, x, d_ind]
                if temp < best_score:
                    best_score = temp
                    best_d = d_ind
                    
            Res[y,x] = best_d

    return Res


#@njit
def get_hsv(d_ind, D, max_disp):
    
    dy = D[d_ind][0]
    dx = D[d_ind][1]
    
    # Hue
    vector_1 = [dx, 0]
    vector_2 = [dx, dy]
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    Hue = (np.sin(angle) + 1)/2
    if np.isnan(Hue):
        Hue = 0

    # Value
    Val = ( dx**2 + dy**2 ) / ( 2 * max_disp**2)
    
    #Saturation
    Sat = 1
    
    return np.array( [Hue, Sat, Val] )


    #@njit

