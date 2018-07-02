from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

PATH = "C:\\Users\\john_\\Desktop\\SAR_Test\\Mount_Royal_SAR\\"
Marker_Channel = 1

def Get_Probability_Map(Image_Filename, Uncertainty=1):
    I = Image.open(PATH+Image_Filename)
    I_Blurred = I.filter(ImageFilter.GaussianBlur(Uncertainty))
    Img = (255-np.array(I_Blurred,dtype=float))/255
    PMap = Img[:,:,Marker_Channel]/np.sum(Img[:,:,Marker_Channel])
    return(PMap)


if __name__=="__main__":
    P_H1 = 0.10
    P_H2 = 0.30
    P_H3 = 0.15
    P_H4 = 0.15
    P_H5 = 0.30

    P_XY_H1 = Get_Probability_Map("Mount_Royal_H1.png")
    P_XY_H2 = Get_Probability_Map("Mount_Royal_H2.png")
    P_XY_H3 = Get_Probability_Map("Mount_Royal_H3.png")
    P_XY_H4 = Get_Probability_Map("Mount_Royal_H4.png")
    P_XY_H5 = Get_Probability_Map("Mount_Royal_H5.png")

    P_XY = P_XY_H1*P_H1 + P_XY_H2 * P_H2 + P_XY_H3 * P_H3 + P_XY_H4 * P_H4 + P_XY_H5 * P_H5

    plt.imshow(P_XY,cmap='coolwarm')
    plt.axis('off')
    plt.show()