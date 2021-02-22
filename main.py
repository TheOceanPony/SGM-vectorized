import numpy as np

from skimage.io import imsave, imshow, imread
#from skimage.color import rgba2gray
import cv2
import funcs as f



if __name__ == '__main__':

    Dy = list(range(-5,5+1))     #D[0]
    Dx = list(range(-5,5+1))     #D[1]

    D = np.transpose([np.tile(Dy, len(Dx)), np.repeat(Dx, len(Dy))])

    # Img import
    # img_L = (rgba2gray( imread("imgs/a1.png"))*255 )
    # img_R = (rgba2gray( imread("imgs/a2.png"))*255 )
    img_L = cv2.cvtColor(cv2.imread("imgs/a1.png"), cv2.COLOR_BGR2GRAY)*255
    img_R = cv2.cvtColor(cv2.imread("imgs/a2.png"), cv2.COLOR_BGR2GRAY)*255
    height, width = img_L.shape[:2] #y, x
    print(f"Img info - shape:{width, height}, max el:{np.max(img_L)}, dtype:{img_L.dtype}")

    # Init
    H = f.initialize_H(height, width, D, img_L, img_R)
    G = f.initialize_G(D)

    Li = f.init_left_part(height, width, D, H, G)
    Ri = f.init_right_part(height, width, D, H, G)
    Ui = f.init_top_part(height, width, D, H, G)
    Di = f.init_bottom_part(height, width, D, H, G)

    # reconstruct
    Res = f.reconstruct(height, width, Li, Ri, Ui, Di, H, D)
    Dm = np.zeros((height, width, 3))

    max_disp = max( np.max(np.array(Dx)), np.max(np.array(Dy)))

    for y in range(0, height):
            for x in range(0, width):
                Dm[y,x] = f.get_hsv(Res[y,x], D, max_disp)

    imshow(Dm, cmap='hsv')
    imsave('Finalresult.png',Dm)