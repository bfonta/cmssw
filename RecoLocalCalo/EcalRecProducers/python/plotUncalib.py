#!/usr/bin/env python
# coding: utf-8

import os, sys

#get_ipython().run_line_magic('jsroot', 'on')
import ROOT
ROOT.gROOT.SetBatch(True);
ROOT.gStyle.SetOptStat(1)
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetTextFont(42)

COLORS= 2,4 #red and blue
WIDTH=1

path = sys.argv[1]
picname = sys.argv[2]

def setAxis(histo):
  histo.GetXaxis().SetTitle("cpu");
  histo.GetYaxis().SetTitle("gpu");

def setAxisDelta(histo):
  histo.GetXaxis().SetTitle("cpu");
  histo.GetYaxis().SetTitle("#Delta gpu-cpu");

def finalizeCanvas(c):   
    """ adjusts the margins of a canvas and prints a CMS header on top"""
    c.SetLeftMargin(0.15)
    c.SetTopMargin(0.05)
    c.SetBottomMargin(0.1)
    c.SetRightMargin(0.03)
    label = ROOT.TLatex()
    label.SetTextSize(0.04)
    label.SetTextAlign(ROOT.kVAlignBottom+ROOT.kHAlignLeft)
    label.DrawLatexNDC(0.64, 0.96, "#bf{CMS} Private Work")

try:
    f = ROOT.TFile.Open(path)
    datafolder = 'DQMData/Run 323775'
    d = f.Get(datafolder)
except:
    raise

hpairs = dict()
hsingle = list()
nhistos = len(list(d.GetListOfKeys()))
print('Total number of histograms: {}'.format(nhistos))

for key in list(d.GetListOfKeys()):
    obj = key.ReadObj()
    name = obj.GetName()
    
    if 'GPU' in name:
        gpuname = name.replace('GPU', 'CPU')
        if gpuname in list(hpairs.keys()):
            hpairs[gpuname] += (obj,)
    
    elif 'vs' not in name and 'ratio' not in name:
        hpairs[name] = (obj,)
    
    if 'vs' in name:
        hsingle.append(obj)
    elif 'ratio' in name:
        hsingle.append(obj)
for k,v in hpairs.items():
    assert(len(hpairs[k])==2)

print('')
print('Number of single histograms: {}'.format(len(hsingle)))
for h in hsingle:
    print(h.GetName())
print('')
print('Number of paired histograms: {}*2'.format(len(hpairs)))
for h in hpairs:
    print(h)

assert(len(hsingle)+2*len(hpairs) == nhistos)

print('Number of pairs: {}'.format(len(hpairs)))
print('Number of singles: {}'.format(len(hsingle)))

#Filter
filt = 'Amplitudes'
lfilter = [x for x in hsingle if filt in x.GetName()]
print(len(lfilter))

cfilt = ROOT.TCanvas('cfilt','cfilt',1600,800)
cfilt.Divide(3,2)
for i,h in enumerate(lfilter):
    cfilt.cd(i+1)
    if 'ratio' in h.GetName():
        ROOT.gPad.SetLogy()
    h.SetLineColor(COLORS[1]);
    h.SetLineWidth(WIDTH);
    if 'vs' in h.GetName():
        h.Draw('colz')
    else:
        h.Draw()
cfilt.SaveAs(str(picname)+'_filt1.png')

legends = []
lfilter2 = [v for k,v in hpairs.items() if filt in k]
print(len(lfilter2))
cfilt2 = ROOT.TCanvas('cfilt2', 'cfilt2',1600,800)
cfilt2.Divide(3,2)
for i,h in enumerate(lfilter2):
    legends.append( ROOT.TLegend(0.7,0.65,0.9,0.75) );    
    cfilt2.cd(i+1)
    ROOT.gPad.SetLogy()

    h1 = h[0]
    h2 = h[1]

    h1.SetLineColor(COLORS[1]);
    h1.SetLineWidth(WIDTH);
    h1.Draw()
    h2.SetLineColor(COLORS[0]);
    h2.SetLineWidth(WIDTH);
    h2.Draw('same')
    
    legends[-1].AddEntry(h1,"CPU","f");
    legends[-1].AddEntry(h2,"GPU","f");
    legends[-1].Draw();
cfilt2.SaveAs(str(picname)+'_filt2.png')
