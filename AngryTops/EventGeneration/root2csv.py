#!/usr/bin/env python
import os, sys
import csv
import signal
# from ROOT import *
import ROOT
from ROOT import TLorentzVector, gROOT, TChain, TVector2
from tree_traversal import GetIndices
from helper_functions import *
import numpy as np
import traceback
from data_augmentation import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import AngryTops.Plotting.PlottingHelper as plot_help

gROOT.SetBatch(True)

from AngryTops.features import *

dir_plots = 'plots_Feb6/'


def make_root_2hists(y1, weight1, bin_number, y_min, y_max, title1, y2=pd.Series([]), weight2=pd.Series([]),title2=pd.Series([]), histname = 'img'):

    hist1 = ROOT.TH1F(title1, "[GeV]", int(bin_number), y_min, y_max)

    # doing this prevents memory leak apperantly 
    hist1.SetDirectory(0)

    # Add points to the hist
    for index in np.arange(len(y1)):
            hist1.Fill(y1[index])

    # FORMAT HISTOGRAMS
    xtitle = hist1.GetXaxis().GetTitle()
    ytitle = hist1.GetYaxis().GetTitle()
    # hist1.Sumw2(0)
    hist1.SetMarkerColor(ROOT.kRed)
    hist1.SetLineColor(ROOT.kRed)
    hist1.SetMarkerStyle(24)
    plot_help.Normalize(hist1)

    if y2.empty == False:
        hist2 = ROOT.TH1F(title2, "[GeV]", int(bin_number), y_min, y_max)
        for index in np.arange(len(y2)):
            hist2.Fill(y2[index],weight2[index])
            hist2.SetDirectory(0)
        
        xtitle = hist2.GetXaxis().GetTitle()
        ytitle = hist2.GetYaxis().GetTitle()
        # hist2.Sumw2()
        hist2.SetMarkerColor(ROOT.kRed)
        hist2.SetLineColor(ROOT.kRed)
        hist2.SetMarkerStyle(24)
        plot_help.Normalize(hist2)

    # set the style
    plot_help.SetTH1FStyle(hist1, color=ROOT.kGray+2, fillstyle=1001, fillcolor=ROOT.kGray, linewidth=3, markersize=0 )

    if y2.empty == False:
        plot_help.SetTH1FStyle(hist2, color=ROOT.kBlack, markersize=0, markerstyle=20, linewidth=3 )

    # DRAW HISTOGRAMS
    c, pad0, pad1 = plot_help.MakeCanvas()
    pad0.cd()
    ROOT.gStyle.SetOptTitle(0)

    hist1.Draw("h")
    data_max = hist1.GetMaximum()
    data_min = hist1.GetMinimum()
    hist1.SetMaximum(1.25 * data_max)
    hist1.SetMinimum(0.5 * data_min)

    if y2.empty == False:
        hist2.Draw("h same")
        data_max = np.maximum(hist1.GetMaximum(),hist2.GetMaximum())
        data_min = np.minimum(hist1.GetMinimum(),hist2.GetMinimum())
        hist1.SetMaximum(1.25 * data_max)
        hist2.SetMaximum(1.25 * data_max)
        hist1.SetMinimum(0.5 * data_min)
        hist2.SetMinimum(0.5 * data_min)
        print(int(hist1.GetMaximumBin()))
        print(int(hist2.GetMaximumBin()))

    # Legend
    leg = ROOT.TLegend( 0.20, 0.80, 0.50, 0.90 )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.AddEntry( hist1, title1, "f" )
    if y2.empty == False:
        leg.AddEntry( hist2, title2, "f" )
    leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
    leg.Draw()

    # SAVE AND CLOSE HISTOGRAM
    ROOT.gPad.RedrawAxis()
    pad1.cd()


    yrange = [0.4, 1.6]
    if y2.empty == False:
        frame, tot_unc, ratio = plot_help.DrawRatio(hist1, hist2, xtitle, yrange)
    else:
        frame, tot_unc, ratio = plot_help.DrawRatio(hist1, hist1, xtitle, yrange)

    c.cd()
    c.SaveAs("{}/{}.png".format(dir_plots,histname))
    pad0.Close()
    pad1.Close()
    c.Close()


def signal_handler(signal, frame):

    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)

    print('You pressed Ctrl+C!')

    momenta_3 = np.asarray(momenta_3_list)
    t_pt_diff = np.asarray(t_pt_diff_list)

    make_root_2hists(df_computed_t_had['mass'],df_computed_t_had['weight'], 50, 50., 200.,'t mass computed had',\
                   df_actual_t_had['mass'],df_actual_t_had['weight'],'t mass actual had', histname ='t_mass_had')

    make_root_2hists(df_computed_t_lep['mass'],df_computed_t_lep['weight'], 50, 50., 200.,'t mass computed lep',\
                   df_actual_t_lep['mass'],df_actual_t_lep['weight'],'t mass actual lep', histname ='t_mass_lep')
    
    make_root_2hists(df_actual_b_lep['mass'],df_actual_b_lep['weight'], 50, 4.5, 5.5 ,'b mass computed lep', histname ='b_mass_lep')
    make_root_2hists(df_actual_W_lep['mass'],df_actual_W_lep['weight'], 50, 60., 100. ,'W mass actual lep', histname ='w_mass_lep')
    make_root_2hists(df_actual_b_lep['E'],df_actual_b_lep['weight'], 50, 0., 500. ,'b E actual lep', histname ='b_E_lep')
    make_root_2hists(df_actual_t_lep['E'],df_actual_t_lep['weight'], 50, 0., 800. ,'t E actual lep', histname ='t_E_lep')
    make_root_2hists(df_actual_W_lep['E'],df_actual_W_lep['weight'], 50, 0., 500. ,'w E actual lep', histname ='w_E_lep')
    make_root_2hists(df_actual_b_lep['Pt'],df_actual_b_lep['weight'], 50, 0., 500. ,'b pt actual lep', histname ='b_pt_lep')
    make_root_2hists(df_actual_t_lep['Pt'],df_actual_t_lep['weight'], 50, 0., 500. ,'t pt actual lep', \
                    df_actual_t_had['Pt'],df_actual_t_had['weight'], 't pt actual had',histname ='t_pt')
    make_root_2hists(df_actual_W_lep['Pt'],df_actual_W_lep['weight'], 50, 0., 500. ,'w pt actual lep', histname ='w_pt_lep')

    # momenta_3 [[p_t_x,p_t_y,p_t_z],[p_w_x,p_w_y,p_w_z],[p_b_x,p_b_y,p_b_z]]
    number_of_entry = np.shape(momenta_3)[0]
    make_root_2hists(momenta_3[:,0,0]-momenta_3[:,1,0]-momenta_3[:,2,0],\
        number_of_entry, 50, -100., 100. ,'px diff(p_t-p_w-p_b)', histname ='px_diff')
    make_root_2hists(momenta_3[:,0,1]-momenta_3[:,1,1]-momenta_3[:,2,1],\
        number_of_entry, 50, -100., 100. ,'py diff(p_t-p_w-p_b)', histname ='py_diff')
    make_root_2hists(momenta_3[:,0,2]-momenta_3[:,1,2]-momenta_3[:,2,2],\
        number_of_entry, 50, -100., 100. ,'pz diff(p_t-p_w-p_b)', histname ='pz_diff')

    make_root_2hists(t_pt_diff,np.ones(len(t_pt_diff)), 50, -100., 100. ,'pt diff', histname ='pt_diff')

    # name 4-momentum 
    # Converting to np array to change dtype
    # np.savetxt(dir_plots + 'outliers_had.txt',np.array(outliers_had, dtype = object), \
    #     header = 'E(t) Px(t) Py(t) Pz(t) E(b) Px(b) Py(b) Pz(b) E(w) Px(w) Py(w) Pz(w) Computed_Mass_t Actual_Mass_t',\
    #      fmt=' '.join(['%1.3f']*np.shape(outliers_had)[1]),delimiter=',')
    
    sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)

# LOGGING
try:
    log = open("log.txt", 'w')
except Exception as e:
    os.mkdir(train_dir)
    log = open("log.txt", 'w')
sys.stdout = log

###############################
# CONSTANTS
GeV = 1e3
TeV = 1e6

# Artificially increase training data size by 5 by rotating events differently 5 different ways
n_data_aug = 1
if len(sys.argv) > 3:
    n_data_aug = int(sys.argv[3])

# Maximum number of entries
n_evt_max = -1
#if len(sys.argv) > 2: n_evt_max = int( sys.argv[2] )

###############################
# BUILDING OUTPUTFILE

# List of filenames
filelistname = sys.argv[1]

# Output filename
outfilename = sys.argv[2]
#outfilename = "csv/topreco_augmented1.csv"
#outfilename = "csv/topreco.csv"
outfile = open( outfilename, "wt" )
csvwriter = csv.writer( outfile )
print ("INFO: output file:", outfilename)

###############################
# BUILDING OUTPUTFILE

# Not entirely sure how TCHAIN works, but I am guessing the truth/nominal input determines which file it loads in?
tree = TChain("Delphes", "Delphes")
f = open( filelistname, 'r' )
for fname in f.readlines():
   fname = fname.strip()
   tree.AddFile( fname )

n_entries = tree.GetEntries()
print("INFO: entries found:", n_entries)

###############################
# LOOPING THROUGH EVENTS

# Cap on number of reconstructed events.
if n_evt_max > 0: n_entries = min( [ n_evt_max, n_entries ] )
n_jets_per_event = 5
print("INFO: looping over %i reco-level events" % n_entries)
print("INFO: using data augmentation: rotateZ %ix" % n_data_aug)

# Number of events which are actually copied over
n_good = 0


# Looping through the reconstructed entries
for ientry in range(n_entries):
    ##############################################################
    # Withdraw next event
    tree.GetEntry(ientry)
    runNumber = tree.GetTreeNumber()
    eventNumber = tree.GetLeaf("Event.Number").GetValue()
    weight = tree.GetLeaf("Event.Weight").GetValue()

    # Printing how far along in the loop we are
    if (n_entries < 10) or ((ientry+1) % int(float(n_entries)/10.) == 0):
        perc = 100. * ientry / float(n_entries)
        print("INFO: Event %-9i  (%3.0f %%)" % (ientry, perc))

    # Number of muons, leptons, jets and bjets (bjet_n set later)
    # For now, I am cutting out reactions with electrons, or more than two
    mu_n = tree.GetLeaf("Muon.PT").GetLen()
    jets_n  = tree.GetLeaf("Jet.PT").GetLen()
    bjets_n = 0

    # If more than one lepton of less than 4 jets, cut
    if mu_n != 1:
        print("Incorrect number of muons: {}. Applying cuts".format(mu_n))
        continue
    if jets_n < 4:
        print("Missing jets: {}. Applying cuts".format(jets_n))
        continue

    ##############################################################
    # Muon vector. Replaced E w/ T
    lep = TLorentzVector()
    lep.SetPtEtaPhiM( tree.GetLeaf("Muon.PT").GetValue(0),
                      tree.GetLeaf("Muon.Eta").GetValue(0),
                      tree.GetLeaf("Muon.Phi").GetValue(0),
                      0
                      )
    lep.sumPT = tree.GetLeaf("Muon.SumPt").GetValue(0)
    if lep.Pt() < 20:
        print("Lepton PT below threshold: {}. Applying cuts".format(lep.Pt()))
        continue # Fail to get a muon passing the threshold
    if np.abs(lep.Eta()) > 2.5:
        print("Lepton Eta value above threshold: {}. Apply cuts".format(lep.Eta()))
        continue

    # Missing Energy values
    met_met = tree.GetLeaf("MissingET.MET").GetValue(0)
    met_phi = tree.GetLeaf("MissingET.Phi").GetValue(0)
    met_eta = tree.GetLeaf("MissingET.Eta").GetValue(0)

    # Append jets, check prob of being a bjet, and update bjet number
    # This is what will be fed into the RNN
    # Replaced the mv2c10 value with the bjet tag value, as that is what is
    # recoreded by Delphes
    outlier_indices = []
    jets = []
    bjets = []
    for i in range(jets_n):
        if i >= n_jets_per_event: break
        if tree.GetLeaf("Jet.PT").GetValue(i) > 20 or np.abs(tree.GetLeaf("Jet.Eta").GetValue(i)) < 2.5 :
            jets += [ TLorentzVector() ]
            j = jets[-1]
            j.index = i
            j.SetPtEtaPhiM(
            tree.GetLeaf("Jet.PT").GetValue(i),
            tree.GetLeaf("Jet.Eta").GetValue(i),
            tree.GetLeaf("Jet.Phi").GetValue(i),
            tree.GetLeaf("Jet.Mass").GetValue(i))
            j.btag = tree.GetLeaf("Jet.BTag").GetValue(i)
            if j.btag > 0.0:
                bjets_n += 1
                bjets.append(j)
    # Cut based on number of passed jets
    jets_n = len(jets)
    if jets_n < 4:
        print("Missing jets: {}. Applying cuts".format(jets_n))
        continue

    ##############################################################
    # Build output data we are trying to predict with RNN
    try:
        indices = GetIndices(tree, ientry)
        ## None means something is wrong(outlier). ignore this case
        if indices == None:
            continue
    except Exception as e:
        print("Exception thrown when retrieving indices")
        print(e)
        print(traceback.format_exc())
        continue

    t_had = TLorentzVector()
    t_lep = TLorentzVector()
    W_had = TLorentzVector()
    W_lep = TLorentzVector()
    b_had = TLorentzVector()
    b_lep = TLorentzVector()

    t_had.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['t_had']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['t_had']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['t_had']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['t_had'])
                        )

    W_had.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['W_had']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['W_had']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['W_had']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['W_had'])
                        )

    t_lep.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['t_lep']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['t_lep']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['t_lep']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['t_lep']))

    W_lep.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['W_lep']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['W_lep']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['W_lep']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['W_lep']))

    b_had.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['b_had']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['b_had']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['b_had']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['b_had']))

    b_lep.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['b_lep']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['b_lep']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['b_lep']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['b_lep']))

    ##############################################################
    # CUTS USING PARTICLE LEVEL OBJECTS
    if (t_had.Pz() == 0.) or (t_had.M() != t_had.M()):
        print("Invalid t_had values, P_z = {0}, M = {1}".format(t_had.Pz(), t_had.M()))
        continue
    if (t_lep.Pz() == 0.) or (t_lep.M() != t_lep.M()):
        print("Invalid t_lep values, P_z = {0}, M = {1}".format(t_lep.Pz(), t_lep.M()))
        continue
    # if W_had.Pt() < 20:
    #     print("Invalid W_had.pt: {}".format(W_had.Pt()))
    #     continue
    # if W_lep.Pt() < 20:
    #     print("Invalid W_lep.pt: {}".format(W_lep.Pt()))
    #     continue
    # if b_had.Pt() < 20:
    #     print("Invalid b_had.pt: {}".format(b_had.Pt()))
    #     continue
    # if b_lep.Pt() < 20:
    #     print("Invalid b_lep.pt: {}".format(b_lep.Pt()))
    #     continue
    # # if np.abs(W_had.Eta()) > 2.5:
    # #     print("Invalid W_had.eta: {}".format(W_had.Eta()))
    # #     continue
    # # if np.abs(W_lep.Eta()) > 2.5:
    # #     print("Invalid W_lep.eta: {}".format(W_lep.Eta()))
    # #     continue
    # if np.abs(b_had.Eta()) > 2.5:
    #     print("Invalid b_had.eta: {}".format(b_had.Eta()))
    #     continue
    # if np.abs(b_lep.Eta()) > 2.5:
    #     print("Invalid b_lep.eta: {}".format(b_lep.Eta()))
    #     continue

    ##############################################################
    # DERIVED EVENT-WISE QUANTITES
    # Sum of all Pt's
    H_t = tree.GetLeaf("ScalarHT.HT").GetValue(0)
    # DeltaPhi( let, closest b) = min DeltaPhi( lep, biets), and mass(l,b)
    if len(bjets) > 1:
        closest_b = np.argmin([lep.Phi() - bjets[i].Phi() for i in range(len(bjets))])
        DeltaPhi = lep.Phi() - bjets[closest_b].Phi()
        InvPart = bjets[closest_b] + lep
        InvMass = InvPart.M()
    elif len(bjets) == 1:
        closest_b = 0
        DeltaPhi = lep.Phi() - bjets[0].Phi()
        InvPart = bjets[0] + lep
        InvMass = InvPart.M()
    else:
        closest_b = -1
        DeltaPhi = -1
        InvMass = -1

    vertex_n = tree.GetLeaf("Vertex_size").GetValue(0)


    ##############################################################
    # Augment Data By Rotating 5 Different Ways
    n_good += 1
    phi = 0
    flip_eta = False
    # Set the phi angle of the lepton to zero
    if len(sys.argv) > 4 and sys.argv[4] == 'aa': phi = -1 * lep.Phi()
    ## Turn off all augmentations
    #if len(sys.argv) > 5 and sys.argv[5] == 'bb': flip_eta = True
    print("Writing new row to csv file")
    for i in range(n_data_aug):
    # make event wrapper
        ## Turn off all augmentations
        #lep_aug, jets_aug, met_phi_aug = RotateEvent(lep, jets, met_phi, phi)
        #lep_flipped, jets_flipped = FlipEta(lep_aug, jets_aug)
        #sjets0, target_W_had, target_b_had, target_t_had, target_W_lep, target_b_lep, target_t_lep = MakeInput(jets_aug, W_had, b_had, t_had, W_lep, b_lep, t_lep )
        #sjets1, _, _, _, _, _, _ = MakeInput(jets_flipped, W_had, b_had, t_had, W_lep, b_lep, t_lep )

        sjets0, target_W_had, target_b_had, target_t_had, target_W_lep, target_b_lep, target_t_lep = MakeInput(jets, W_had, b_had, t_had, W_lep, b_lep, t_lep )
        sjets1, _, _, _, _, _, _ = MakeInput(jets, W_had, b_had, t_had, W_lep, b_lep, t_lep )


    # write out
        csvwriter.writerow( (
        "%i" % runNumber, "%i" % eventNumber, "%.5f" % weight, "%i" % jets_n, "%i" % bjets_n,
        #"%.5f" % lep_aug.Px(),     "%.5f" % lep_aug.Py(),     "%.5f" % lep_aug.Pz(),     "%.5f" % lep_aug.E(),      "%.5f" % met_met,      "%.5f" % met_phi_aug,
        "%.5f" % lep.Px(),     "%.5f" % lep.Py(),     "%.5f" % lep.Pz(),     "%.5f" % lep.E(),      "%.5f" % met_met,      "%.5f" % met_phi,
        "%.5f" % sjets0[0][0],  "%.5f" % sjets0[0][1],  "%.5f" % sjets0[0][2],  "%.5f" % sjets0[0][3],  "%.5f" % sjets0[0][4],  "%.5f" % sjets0[0][5],
        "%.5f" % sjets0[1][0],  "%.5f" % sjets0[1][1],  "%.5f" % sjets0[1][2],  "%.5f" % sjets0[1][3],  "%.5f" % sjets0[1][4],  "%.5f" % sjets0[1][5],
        "%.5f" % sjets0[2][0],  "%.5f" % sjets0[2][1],  "%.5f" % sjets0[2][2],  "%.5f" % sjets0[2][3],  "%.5f" % sjets0[2][4],  "%.5f" % sjets0[2][5],
        "%.5f" % sjets0[3][0],  "%.5f" % sjets0[3][1],  "%.5f" % sjets0[3][2],  "%.5f" % sjets0[3][3],  "%.5f" % sjets0[3][4],  "%.5f" % sjets0[3][5],
        "%.5f" % sjets0[4][0],  "%.5f" % sjets0[4][1],  "%.5f" % sjets0[4][2],  "%.5f" % sjets0[4][3],  "%.5f" % sjets0[4][4],  "%.5f" % sjets0[4][5],
        "%.5f" % target_W_had[0], "%.5f" % target_W_had[1], "%.5f" % target_W_had[2], "%.5f" % target_W_had[3], "%.5f" % target_W_had[4],
        "%.5f" % target_W_lep[0], "%.5f" % target_W_lep[1], "%.5f" % target_W_lep[2], "%.5f" % target_W_lep[3], "%.5f" % target_W_lep[4],
        "%.5f" % target_b_had[0], "%.5f" % target_b_had[1], "%.5f" % target_b_had[2], "%.5f" % target_b_had[3], "%.5f" % target_b_had[4],
        "%.5f" % target_b_lep[0], "%.5f" % target_b_lep[1], "%.5f" % target_b_lep[2], "%.5f" % target_b_lep[3], "%.5f" % target_b_lep[4],
        "%.5f" % target_t_had[0], "%.5f" % target_t_had[1], "%.5f" % target_t_had[2], "%.5f" % target_t_had[3], "%.5f" % target_t_had[4],
        "%.5f" % target_t_lep[0], "%.5f" % target_t_lep[1], "%.5f" % target_t_lep[2], "%.5f" % target_t_lep[3], "%.5f" % target_t_lep[4],
        "%.5f" % lep.Pt(),     "%.5f" % lep.Eta(),     "%.5f" % lep.Phi(),
        "%.5f" % sjets0[0][6],  "%.5f" % sjets0[0][7],  "%.5f" % sjets0[0][8],
        "%.5f" % sjets0[1][6],  "%.5f" % sjets0[1][7],  "%.5f" % sjets0[1][8],
        "%.5f" % sjets0[2][6],  "%.5f" % sjets0[2][7],  "%.5f" % sjets0[2][8],
        "%.5f" % sjets0[3][6],  "%.5f" % sjets0[3][7],  "%.5f" % sjets0[3][8],
        "%.5f" % sjets0[4][6],  "%.5f" % sjets0[4][7],  "%.5f" % sjets0[4][8],
        "%.5f" % target_W_had[5], "%.5f" % target_W_had[6], "%.5f" % target_W_had[7],
        "%.5f" % target_W_lep[5], "%.5f" % target_W_lep[6], "%.5f" % target_W_lep[7],
        "%.5f" % target_b_had[5], "%.5f" % target_b_had[6], "%.5f" % target_b_had[7],
        "%.5f" % target_b_lep[5], "%.5f" % target_b_lep[6], "%.5f" % target_b_lep[7],
        "%.5f" % target_t_had[5], "%.5f" % target_t_had[6], "%.5f" % target_t_had[7],
        "%.5f" % target_t_lep[5], "%.5f" % target_t_lep[6], "%.5f" % target_t_lep[7],
        "%.5f" % H_t, "%i" % closest_b, "%.5f" % DeltaPhi, "%.5f" % InvMass
            ) )

        if flip_eta:
            csvwriter.writerow( (
            "%i" % runNumber, "%i" % eventNumber, "%.5f" % weight, "%i" % jets_n, "%i" % bjets_n,
            "%.5f" % lep_flipped.Px(),     "%.5f" % lep_flipped.Py(),     "%.5f" % lep_flipped.Pz(),     "%.5f" % lep_flipped.E(),
            "%.5f" % met_met,      "%.5f" % met_phi_aug,
            "%.5f" % sjets1[0][0],  "%.5f" % sjets1[0][1],  "%.5f" % sjets1[0][2],  "%.5f" % sjets1[0][3],  "%.5f" % sjets1[0][4],  "%.5f" % sjets1[0][5],
            "%.5f" % sjets1[1][0],  "%.5f" % sjets1[1][1],  "%.5f" % sjets1[1][2],  "%.5f" % sjets1[1][3],  "%.5f" % sjets1[1][4],  "%.5f" % sjets1[1][5],
            "%.5f" % sjets1[2][0],  "%.5f" % sjets1[2][1],  "%.5f" % sjets1[2][2],  "%.5f" % sjets1[2][3],  "%.5f" % sjets1[2][4],  "%.5f" % sjets1[2][5],
            "%.5f" % sjets1[3][0],  "%.5f" % sjets1[3][1],  "%.5f" % sjets1[3][2],  "%.5f" % sjets1[3][3],  "%.5f" % sjets1[3][4],  "%.5f" % sjets1[3][5],
            "%.5f" % sjets1[4][0],  "%.5f" % sjets1[4][1],  "%.5f" % sjets1[4][2],  "%.5f" % sjets1[4][3],  "%.5f" % sjets1[4][4],  "%.5f" % sjets1[4][5],
            "%.5f" % target_W_had[0], "%.5f" % target_W_had[1], "%.5f" % target_W_had[2], "%.5f" % target_W_had[3], "%.5f" % target_W_had[4],
            "%.5f" % target_W_lep[0], "%.5f" % target_W_lep[1], "%.5f" % target_W_lep[2], "%.5f" % target_W_lep[3], "%.5f" % target_W_lep[4],
            "%.5f" % target_b_had[0], "%.5f" % target_b_had[1], "%.5f" % target_b_had[2], "%.5f" % target_b_had[3], "%.5f" % target_b_had[4],
            "%.5f" % target_b_lep[0], "%.5f" % target_b_lep[1], "%.5f" % target_b_lep[2], "%.5f" % target_b_lep[3], "%.5f" % target_b_lep[4],
            "%.5f" % target_t_had[0], "%.5f" % target_t_had[1], "%.5f" % target_t_had[2], "%.5f" % target_t_had[3], "%.5f" % target_t_had[4],
            "%.5f" % target_t_lep[0], "%.5f" % target_t_lep[1], "%.5f" % target_t_lep[2], "%.5f" % target_t_lep[3], "%.5f" % target_t_lep[4],
            "%.5f" % lep_flipped.Pt(),     "%.5f" % lep_flipped.Eta(),     "%.5f" % lep_flipped.Phi(),
            "%.5f" % sjets1[0][6],  "%.5f" % sjets1[0][7],  "%.5f" % sjets1[0][8],
            "%.5f" % sjets1[1][6],  "%.5f" % sjets1[1][7],  "%.5f" % sjets1[1][8],
            "%.5f" % sjets1[2][6],  "%.5f" % sjets1[2][7],  "%.5f" % sjets1[2][8],
            "%.5f" % sjets1[3][6],  "%.5f" % sjets1[3][7],  "%.5f" % sjets1[3][8],
            "%.5f" % sjets1[4][6],  "%.5f" % sjets1[4][7],  "%.5f" % sjets1[4][8],
            "%.5f" % target_W_had[5], "%.5f" % target_W_had[6], "%.5f" % target_W_had[7],
            "%.5f" % target_W_lep[5], "%.5f" % target_W_lep[6], "%.5f" % target_W_lep[7],
            "%.5f" % target_b_had[5], "%.5f" % target_b_had[6], "%.5f" % target_b_had[7],
            "%.5f" % target_b_lep[5], "%.5f" % target_b_lep[6], "%.5f" % target_b_lep[7],
            "%.5f" % target_t_had[5], "%.5f" % target_t_had[6], "%.5f" % target_t_had[7],
            "%.5f" % target_t_lep[5], "%.5f" % target_t_lep[6], "%.5f" % target_t_lep[7],
            "%.5f" % H_t, "%i" % closest_b, "%.5f" % DeltaPhi, "%.5f" % InvMass
                ) )
        # Change the angle to rotate by
        phi = np.random.uniform(- np.pi, np.pi)


try:
    outilers = get_outliers()
    # file_buff = []
    # for i in range(len(outliers)):
    #     file_buff.append([tree.GetLeaf("Particle.E").GetValue(outliers[i]),\
    #                                tree.GetLeaf("Particle.Px").GetValue(outliers[i]),\
    #                                tree.GetLeaf("Particle.Py").GetValue(outliers[i]),\
    #                                tree.GetLeaf("Particle.Pz").GetValue(outliers[i])]
    np.savetxt(dir_plots + 'outliers.txt',np.array(outliers, dtype = object), \
        header = 'E Px Py Pz',\
        delimiter=',')
except :
    print('pass')
    pass
##############################################################
# Close Program
outfile.close()

f_good = 100. * n_good / n_entries
print("INFO: output file:", outfilename)
print("INFO: %i entries written (%.2f %%)" % ( n_good, f_good))
sys.stdout = sys.__stdout__
log.close()
