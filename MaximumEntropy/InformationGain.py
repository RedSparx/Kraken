import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def gainDirection(pmap,pos):
    # Extract 9x9 neighborhood about the input point.
    # There is a 0 probability of being off the input map (valid PDF values are within the support region only).
    r,c=pos
    paddedPDF = np.pad(pmap,(1,1),'constant')
    localPDF = paddedPDF[r:r + 3, c:c + 3] # one has been added to each axis to account for the padding.
    localEntropy = -localPDF * np.log(localPDF)
    informationGain = localEntropy - localPDF * localEntropy

    if (informationGain.any()):
        # Find the maximum information gain within the neighborhood.
        ind = np.unravel_index(np.argmax(informationGain, axis=None), informationGain.shape)
        if ind[0] < 1:
            deltaR = - 1
        elif ind[0] > 1:
            deltaR = + 1
        else:
            deltaR = 0

        if ind[1] < 1:
            deltaC = -1
        elif ind[1] > 1:
            deltaC = +1
        else:
            deltaC = 0
    else:
        # informationGain = np.zeros_like(localPDF)
        ind=(0,0)
        # Random Walk
        deltaR = np.round(2*np.random.rand()-1)
        deltaC = np.round(2*np.random.rand()-1)

    return r+deltaR,c+deltaC


if __name__=='__main__':
    N = 200
    x = np.linspace(-3, 3, N)
    y = np.linspace(-3, 3, N)
    xv, yv = np.meshgrid(x, y)
    pdf = np.exp(-(xv** 2 + yv** 2))
    pdf = pdf/np.sum(pdf)

    Img=np.zeros_like(pdf)
    R = int((N-1)*np.random.rand())
    C = int((N-1)*np.random.rand())

    for i in range(2000):
        RNew,CNew=gainDirection(pdf,(R,C))
        Img[RNew,CNew]=1
        R=RNew
        C=CNew

    plt.imshow(1-Img)
    plt.imshow(pdf,alpha=0.8)
    plt.show()

# localEntropy = -localPDF * np.log(localPDF)
# informationGain = localEntropy - localPDF * localEntropy111