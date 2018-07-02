import os
import textwrap
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageFilter

from Kraken.MaximumEntropy.InformationGain import gainDirection

#region Parameters
#   simulation attribute must be 'BayesianSearch'
#endregion

#region Helper functions
def RawText(Str):
    Text = Str.strip('\n\r\t').lstrip()
    Text = ' '.join(Text.split())
    return(Text)

def BlockText(Str,width=80):
    return(textwrap.fill(RawText(Str),width=width))
#endregion

class Search:
    def __init__(self):
        print('Pure Bayesian Search Initialization')

        # Class variables
        self.configuration_filename=0
        self.process_root=0
        self.simulation=0
        self.input_path=0;
        self.output_filename=0;
        self.pmap = 0

    def configure(self,configuration_filename):
        print('Reading configuration file: %s'%configuration_filename)
        self.configuration_filename = configuration_filename
        self.process_root = ET.parse(configuration_filename).getroot()
        self.simulation = self.process_root.findall("simulation[@Type='BayesianSearch']")

    def run(self):
        for s in self.simulation:
            case = s.findall('case')
            for c in case:
                self.probability_map(c)

    def probability_map(self, case):
        print('Title:  %s'%case.attrib['Title'])
        print('Date:   %s\n'%case.attrib['Date'])
        print(BlockText(case.text))
        self.input_path=case.attrib['Path']
        self.output_filename=case.attrib['Output']
        self.map_filename=case.attrib['Map']
        print('\n' + self.input_path + '\n')
        hypothesis = case.findall('hypothesis')
        p_xy=0;
        for h in hypothesis:
            hypothesis_filename=h.attrib['Filename']
            print('\t- %s (%2.1f%%): %s'%(h.attrib['Title'],100*float(h.attrib['H']),BlockText(h.text)))
            # Compile probability map.
            p_xy_h = self.extract_probability(hypothesis_filename)
            p_h = float(h.attrib['H'])
            p_xy = p_xy + p_xy_h*p_h
        self.pmap = p_xy
        print('\n')

    def extract_probability(self, filename, marker_channel=1, Uncertainty=10):
        I = Image.open(self.input_path + filename)
        I_Blurred = I.filter(ImageFilter.GaussianBlur(Uncertainty))
        Img = (255-np.array(I_Blurred,dtype=float))/255
        PMap = Img[:,:,marker_channel]/np.sum(Img[:,:,marker_channel])
        return(PMap)

    def highest_likelihood(self):
        # Determine the maximum probability density value from the probability map.  This can be taken as a likelihood
        # estimate given a search is conducted (p(x,y|s).
        i,j = np.unravel_index(np.argmax(self.pmap),self.pmap.shape)
        return i,j

    def compute_entropy(self):
        S=-np.sum(self.pmap*np.log(self.pmap))
        return(S)

    def overlay_image(self):
        # Load the imput map
        # and convert it to RGB with a transparency layer.
        self.Map=Image.open(os.path.join(self.input_path,self.map_filename))
        self.Map.convert('RGBA')
        # Convert the probability map to an image. Scale the values from 0-255 and convert them to integers.  Apply a
        # suitable colormap.  Then convert to an image that has three color channels including a transparency layer.  A
        # colormap is applied to visualize the likelihood.
        PMap_Image=Image.fromarray(np.uint8(cm.jet(self.pmap/np.max(self.pmap))*255)).convert('RGBA')
        # Create a Boolean map for areas where the probability is less than 5% of the maximum.  Invert this map so that
        # it can be converted to a mask that will be used as the overlay for the final image.  To do this, it must be
        # inverted.  Boolean values are {0,1}.
        Mask = 1-np.array(self.pmap < (np.max(self.pmap)*0.05),dtype=int)
        # Convert the Boolean mask to a PIL image with a single 8-bit channel (Type 'L').  Note that Boolean values are
        # scaled to the range {0,255} by simple multiplication.  The array is then converted to the appropriate PIL
        # container.
        Mask_Image = Image.fromarray(Mask*127).convert('L')
        Mask_Image.filter(ImageFilter.GaussianBlur(2))
        # On the probability map image, insert the Boolean mask into the transparency layer, then paste it onto the
        # input map using the mask.
        PMap_Image.putalpha(Mask_Image)
        self.Map.paste(PMap_Image,(0,0),Mask_Image)
        self.Map.show()
        self.Map.save(os.path.join(self.input_path,self.output_filename))

    # def generate_html(self):

if __name__ == "__main__":
    #region Create reference to the simulation file.
    Raw_Filename = 'Mount_Royal.xml'
    Path = 'C:/Users/john_/Desktop/SAR_Test'
    Filename = os.path.join(Path,Raw_Filename)
    #endregion
    #region Read the configuration file.
    S = Search()
    S.configure(Filename)
    S.run()
    # S.overlay_image()

    #Demonstration using information gain to search from arbitrary points.  Use 2000 steps.
    Map = Image.open("C:\\Users\\john_\\Desktop\\SAR_Test\\DATA\\Mount_Royal_SAR\\Mount_Royal_BW_Result1_Overlay.png")
    Map.convert('RGBA')
    Map.load()

    # searchPathImage=np.zeros_like(S.pmap)
    # searchPathImage=np.empty_like(S.pmap)
    searchPathImage=Map.load()
    N=S.pmap.shape[0]
    for Rng in range(2):
        R = int(450+np.floor(N/10 * np.random.randn()))
        C = int(450+np.floor(N/10 * np.random.randn()))
        Steps=1500
        for i in range(Steps):
            RNew, CNew = gainDirection(S.pmap, (R, C))
            if RNew>=0 and CNew>=0 and RNew<S.pmap.shape[0] and CNew<S.pmap.shape[1]:
                searchPathImage[RNew, CNew] = (255,255,0)
            R = RNew
            C = CNew

    # Map.show()
    plt.imshow(Map)
    # plt.imshow(S.pmap, alpha=0.25)
    # plt.imshow(searchPathImage,alpha=0.6)
    plt.show()
    #endregion