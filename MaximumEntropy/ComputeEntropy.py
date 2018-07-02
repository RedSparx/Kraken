import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Kraken.BayesianSearch import  Search as kbs
import os
import numpy as np

class Entropy:
    def LocalEntropyMap(self, ProbabilityMap=None, Resolution = 50):
        if ProbabilityMap is None:
            ProbabilityMap=np.random.randn(100,100)
        rows,cols = ProbabilityMap.shape
        # Divide the range up into the required resolution.
        rstep=np.int(np.floor(rows/Resolution))
        cstep=np.int(np.floor(cols/Resolution))
        Entropy=np.empty_like(ProbabilityMap)
        for r in range(0,rows,rstep):
            for c in range(0,cols,cstep):
                SubMatrix=ProbabilityMap[r:r+rstep, c:c+cstep]
                SubMatrix=SubMatrix/np.max(SubMatrix)
                S = -np.sum(SubMatrix*np.log2(SubMatrix))

                Entropy[r:r + rstep, c:c + cstep] = S
                #print('(%d,%d) --> (%d,%d)'%(r,c,r+rstep,c+cstep))
        print(-np.sum(ProbabilityMap*np.log2(ProbabilityMap)))
        return Entropy




#region Main executable block.
if __name__ == "__main__":
    #region Create reference to the simulation file.
    Raw_Filename = 'Mount_Royal.xml'
    Path = 'C:/Users/john_/Desktop/SAR_Test'
    Filename = os.path.join(Path,Raw_Filename)
    #endregion
    #region Read the configuration file.
    S = kbs.Search()
    S.configure(Filename)
    S.run()

    N=2000
    x = np.linspace(-5, 5, N)
    y = np.linspace(-5, 5, N)
    xv,yv=np.meshgrid(x,y)
    z=np.zeros_like(xv)
    for k in range(200):
        sigma=0.01*np.random.rand()
        xrand=5*np.random.randn()
        yrand=5*np.random.randn()
        z+=sigma*np.exp(-((xv-xrand)**2 + (yv-yrand)**2))

    z=z/np.max(z)
    # plt.imshow(z)
    # plt.show()

    E=Entropy()
    plt.subplot(131)
    plt.imshow(z,cmap=cm.hot)
    plt.title('Probability Map')
    plt.axis('off')
    plt.subplot(132)
    Ent=E.LocalEntropyMap(z,Resolution=1000)
    plt.imshow(Ent, cmap=cm.RdYlGn)
    plt.title('Local Entropy Map')
    plt.axis('off')
    plt.subplot(133)
    Grad=np.gradient(Ent)
    plt.imshow(np.sqrt(Grad[0]**2+Grad[1]**2), cmap=cm.RdYlGn)
    plt.title('dS')
    plt.axis('off')
    # plt.colorbar()
    plt.show()


    # E=Entropy()
    # plt.subplot(121)
    # plt.imshow(S.pmap)
    # plt.subplot(122)
    # plt.imshow(E.LocalEntropyMap(S.pmap,Resolution=50),cmap=cm.cool)
    # plt.colorbar()
    # plt.show()