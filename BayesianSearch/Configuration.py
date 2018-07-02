import xml.etree.ElementTree as ET
import textwrap
import webbrowser
import os

def RawText(Str):
    Text = Str.strip('\n\r\t').lstrip()
    Text = ' '.join(Text.split())
    return(Text)

def BlockText(Str):
    return(textwrap.fill(RawText(Str),width=80))


if __name__=="__main__":
    Filename = 'Simulation1.xml'
    Path = 'C:/Users/john_/Desktop/RedSparx/Kraken/BayesianSearch'
    Abs_Filename = os.path.join(Path,Filename)
    Configuration = ET.parse(Abs_Filename)
    #webbrowser.open(Abs_Filename)
    SimRoot = Configuration.getroot()

    # List all case descriptions.
    for Case in SimRoot.findall('case'):
        print(str.upper(Case.attrib['Title']))
        print('-'*len(Case.attrib['Title']))
        print(BlockText(Case.text)+'\n')
        for Hypothesis in Case:
            print('\t* %s (%2.1f%%): %s'%(str.upper(RawText(Hypothesis.attrib['Title'])), 100*float(Hypothesis.attrib['H']),RawText(Hypothesis.text)))
        print('\n')