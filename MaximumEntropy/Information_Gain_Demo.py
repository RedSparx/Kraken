import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#region Generate a gaussian mixture and normalize it into a valid PDF-like function for test.
N = 400
x = np.linspace(-15, 15, N)
y = np.linspace(-15, 15, N)
xv, yv = np.meshgrid(x, y)
pdf = np.zeros_like(xv)

G=5
for k in range(G):
    Gaussian = np.zeros_like(pdf)
    sigma = 10*np.random.rand()
    xrand = 5 * np.random.randn()
    yrand = 5 * np.random.randn()
    # xrand = 30 * np.random.rand()-15.0
    # yrand = 30 * np.random.rand()-15.0
    Gaussian +=np.exp(-((xv - xrand) ** 2 + (yv - yrand) ** 2)/(sigma))
    Gaussian = Gaussian/np.sum(Gaussian)
    pdf+=Gaussian
#endregion
pdf=pdf/np.sum(pdf)

#region INITIALIZATION: Select a random location within the PDF for a particle.  Use simple indexing for simplicity.
Drones=10
particlePos = np.zeros_like(pdf, dtype=bool)
for d in range(Drones):
    randCol=int(N*np.random.rand())
    randRow=int(N*np.random.rand())
    # randCol=int(N/2)
    # randRow=int(N/2)
    # randRow=N-1
    # randCol=N-1
    particlePos[randRow,randCol]=True
    #endregion

    #region TIMESTEP: Check the neighborhood to find the pixel of maximum information gain.
    # Compute the information gain from: I(P_New,_P) = H(P_New)-H(P_New|P_Old).  Furthermore, recall that the conditional
    # entropy is computed as: H(A|B)= -Sum[p(A|B)*H(A)].  Compute the entropy in a 9x9 grid.
    r=randRow
    c=randCol

    for i in range(10000):
        localPDF=pdf[r-1:r+2,c-1:c+2]
        localEntropy=-localPDF*np.log(localPDF)
        informationGain=localEntropy-localPDF*localEntropy
        # print(informationGain)
        ind = np.unravel_index(np.argmax(informationGain, axis=None), informationGain.shape)

        if ind[0]<1:
            r=r-1
        elif ind[0]>1:
            r=r+1

        if ind[1]<1:
            c=c-1
        elif ind[1]>1:
            c=c+1

        if r>=N:
            r=N
        if r<0:
            r=0
        if c>=N:
            c=N
        if c<0:
            c=0

        particlePos[r,c]=True
#endregion

#region Plot the PDF and the particle along the path of information gain.
# plt.subplot(121)
fig=plt.figure()

plt.imshow(pdf,alpha=1.0)
# plt.axis('equal')
# plt.axis('off')
# plt.subplot(122)
plt.imshow(particlePos,cmap=cm.Greys,alpha=0.5)
plt.axis('equal')
plt.axis('off')
fig.subplots_adjust(bottom = 0)
fig.subplots_adjust(top = 1)
fig.subplots_adjust(right = 1)
fig.subplots_adjust(left = 0)
# plt.savefig('Output.png',bbox_inches='tight')
plt.show()
#endregion