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
            for d_ind in range(D.shape[0]):
                if 0<= y-D[d_ind][0] < height and 0<= x-D[d_ind][1] <width:
                    H[y, x, d_ind] = abs( img_L[y,x] - img_R[y - D[d_ind][0], x - D[d_ind][1]] )
                else:
                    H[y, x, d_ind] = np.inf # Weird results
    return H


@njit
def initialize_G(D, alpha=1):
    
    G = np.zeros((len(D), len(D)), dtype=np.float32)
    for d1_ind in range(D.shape[0]):
        for d2_ind in range(D.shape[0]):
            G[d1_ind, d2_ind] = (D[d1_ind][0] - D[d2_ind][0])**2 + (D[d1_ind][1] - D[d2_ind][1])**2
            
    return G


@njit
def init_left_part(height, width, D, H, G):

    Li = np.zeros((height, width, len(D)), dtype=np.float32)

    for y in range(0, height):
        for x in range(0, width):
            for d_ind in range(D.shape[0]):
                Li[y, x, d_ind] = left(y, x, d_ind, Li, D, H, G)
    return Li

@njit
def left(y, x, d_ind, Li, D, H, G):

    if x == 0:
        return 0
    else:
        minl = np.inf

        for d2_ind in range(D.shape[0]):
            temp = Li[y, x-1, d2_ind] + H[y, x-1, d2_ind] + G[d_ind, d2_ind]
            if temp < minl:
                minl = temp

        return minl


@njit
def init_right_part(height, width, D, H, G):

    Ri = np.zeros((height, width, len(D)), dtype=np.float32)

    for y in range(height-1, -1):
        for x in range(width-1, -1):
            for d_ind in range(D.shape[0]):
                Ri[y, x, d_ind] = right(y, x, d_ind, Ri, D, H, G, width)
    return Ri

@njit
def right(y, x, d_ind, Ri, D, H, G, width):

    if x == width-1:
        return 0
    else:
        minl = np.inf

        for d2_ind in range(D.shape[0]):
            temp = Ri[y, x+1, d2_ind] + H[y, x+1, d2_ind] + G[d_ind, d2_ind]
            if temp < minl:
                minl = temp

        return minl


@njit
def init_top_part(height, width, D, H, G):

    Ui = np.zeros((height, width, len(D)), dtype=np.float32)

    for y in range(0, height):
        for x in range(0, width):
            for d_ind in range(D.shape[0]):
                Ui[y, x, d_ind] = up(y, x, d_ind, Ui, D, H, G)
    return Ui

@njit
def up(y, x, d_ind, Ui, D, H, G):

    if y == 0:
        return 0
    else:
        minl = np.inf

        for d2_ind in range(D.shape[0]):
            temp = Ui[y-1, x, d2_ind] + H[y-1, x, d2_ind] + G[d_ind, d2_ind]
            if temp < minl:
                minl = temp

        return minl


@njit
def init_bottom_part(height, width, D, H, G):

    Di = np.zeros((height, width, len(D)), dtype=np.float32)

    for y in range(height-1, -1):
        for x in range(width-1, -1):
            for d_ind in range(D.shape[0]):
                Di[y, x, d_ind] = right(y, x, d_ind, Di, D, H, G, height)
    return Di

@njit
def right(y, x, d_ind, Di, D, H, G, height):

    if x == height-1:
        return 0
    else:
        minl = np.inf

        for d2_ind in range(D.shape[0]):
            temp = Di[y+1, x, d2_ind] + H[y+1, x, d2_ind] + G[d_ind, d2_ind]
            if temp < minl:
                minl = temp

        return minl


#@njit
def reconstruct(height, width, Li, Ri, Ui, Di, H, D):
    
    Res = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(0, height):
        for x in range(0, width):
            best_score = np.inf
            best_d = None
            for d_ind in range(D.shape[0]):
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

