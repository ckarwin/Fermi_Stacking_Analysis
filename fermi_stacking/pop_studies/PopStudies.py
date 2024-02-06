import os,sys #,BatLib
from math import *
#from BatLib import Delete,RunCmd,GetFileType

#from ROOT import gROOT, TCanvas, TH1F, TApplication,gStyle,TGraph,TF1,TGraphErrors,gPad,TMath,TMinuit,TGraph2D,TH2F


from scipy.integrate import romberg,quad
from scipy import sqrt as SQRT

from array import *

# ------------------------------
# *** Cosmology USED
#

g_omegaM = 0.30
g_omegaL = 0.70
g_omegaK = 0

g_H0     = 67.
g_dh     = 2.9979e5/g_H0


g_omegaM = 0.27
g_omegaL = 0.73
g_omegaK = 0

g_H0     = 71.
g_dh     = 2.9979e5/g_H0



###########################################################
class SKYCOV2D:
    """------------------------------------"""
    def __init__(self):

        self.graph2D   = TGraph2D(2680)
        self.np        = 0 # points loaded in the graph 

        self.DataArr   = []
        self.nbase     = 0
        self.ndatasets = 0
        # ---- flux limits for the different photon indices
        self.maxAreaL = []
        self.maxFluxL = []
        self.idxL     = []

        self.compl    = 1.0 # set the completeness

    """------------------------------------"""
    def AddCoverage(self,dataL):

        idx   = dataL[0]
        maxA  = dataL[1]
        maxF  = dataL[2]
        fluxL = dataL[3]
        degL  = dataL[4]

        #print idx,maxA,maxF

        # --- save the data
        self.maxAreaL.append(maxA)
        self.maxFluxL.append(maxF)
        self.idxL.append(idx)
        self.nbase = len(degL)
        self.ndatasets+=1
        
        for i in range(len(fluxL)):
            self.graph2D.SetPoint(self.np,fluxL[i],idx,degL[i])
            self.np+=1

            tmp = [fluxL[i],idx,degL[i]]
            self.DataArr.append(tmp)


        #print(" ** Skycoverage has now %d points"%(self.np))
        return

    """------------------------------------"""
    def GetIndices(self,fx,idx):

            
        i1 = -1
        i2 = -1
        i3 = -1
        i4 = -1
        d1min = 1e6
        d2min = 1e6
        d3min = +100
        d4min = +100

        fmin  = 100
        
        for i  in range (len(self.DataArr)):
            f1 = self.DataArr[i][0]
            p1 = self.DataArr[i][1]

            d1 = fx - f1
            d2 = idx - p1

            if(f1<fmin):
                fmin = f1

            if(d1>=0 and d2>=0 and
               d1<d1min and d2<=d2min):
                d1 = d1min
                d2 = d2min
                i1 = i
                i2 = i+1
                
            if(d1<=0 and d2<=0 and
               abs(d1)<d3min and abs(d2)<d4min):
                d3min = abs(d1)
                d4min = abs(d2)
                i3 = i-1
                i4 = i
                
        if((i1==-1 or i2==-1 or i3== -1 or i4==-1) and fx>1.5*fmin):
            print("ERROR finding indices for fx=%.4e idx=%.2f"%(fx,idx))
            print(1.5*fmin,i1,i2,i3,i4)
            sys.exit()
            
        elif(fx<2.5*fmin):
            """ it falls below the sky coverage """
            #print "Warning fx=%.3e < 2.5*fmin=%.3e"%(fx,2.5*fmin)
            return [-1,-1,-1,-1]

        #print "\n"
        #print self.DataArr[i1]
        #print self.DataArr[i2]
        #print self.DataArr[i3]
        #print self.DataArr[i4]


        return [i1,i2,i3,i4]
    """------------------------------------"""
    def Interpolate(self,idL,fx,idx):

  
        i1 = idL[0]
        i2 = idL[1]
        i3 = idL[2]
        i4 = idL[3]
        
        arr = self.DataArr
        f1  = arr[i1][0]
        f2  = arr[i2][0]
        p1  = arr[i1][1]
        p2  = arr[i4][1]



        if(p1>idx or p2<idx or f1>fx or f2<fx):
            print(">>ERROR in indices during Interpolation for idx=%.2f fx=%.3e"%(idx,fx))
            print(arr[i1])
            print(arr[i2])
            print(arr[i3])
            print(arr[i4])
            
            sys.exit()

        # --- these are the area values
        q11 = arr[i1][2]
        q21 = arr[i2][2]
        q12 = arr[i3][2]
        q22 = arr[i4][2]

        if(q11<=0):
            q11=1e-40
        if(q21<=0):
            q21=1e-40
        if(q12<=0):
            q12=1e-40
        if(q22<=0):
            q22=1e-40


        islog = "no" # yes=use a power-law interpolation, no=linear

        if(islog=="yes"):
            q11 = log10(q11)
            q21 = log10(q21)
            q12 = log10(q12)
            q22 = log10(q22)
            f2  = log10(f2)
            f1  = log10(f1)
            fx  = log10(fx)
            p2  = log10(p2)
            p1  = log10(p1)
            idx = log10(idx)

        a11 = (f2 - fx)/(f2-f1) 
        a12 = (fx - f1)/(f2-f1)
        a21 = (p2 -idx)/(p2-p1)
        a22 = (idx -p1)/(p2-p1)


        fr1 = a11*q11+a12*q21
        fr2 = a11*q12+a12*q22

        val = a21*fr1+a22*fr2


        # --- safety check
        if(val<min(q11,q21,q12,q22)):
            if (  abs(val/min(q11,q21,q12,q22) -1)<1e-2):
                # print val,min(q11,q21,q12,q22)
                if(islog=="yes"):
                    val = pow(10,val)
                    return val
                else:
                    return val
            print("ERROR: in interpolation")
            print("InterpVal = %.3e, q11=%.3e q12=%.3e q21=%.3e q22=%.3e"%(val,q11,q12,q21,q22))
            print(val/min(q11,q21,q12,q22))
            sys.exit()

        if(islog=="yes"):
            val = pow(10,val)
            
        return val
    """------------------------------------"""
    def GetPhIdxLim(self,fx):

        """ these are the flim corresponding to Area=610"""
        
        xl=array('d',
                 # --- corresponding to 0 area
                 #[2.7536971085004963e-10, 5.3714380847432521e-10, 1.2454239098970258e-09, 2.7713585215252018e-09, 5.4698221510216399e-09, 9.2151948942276353e-09, 1.3395894869896271e-08, 1.7256597842631556e-08, 2.0617207796173057e-08]
                 # -- corresponding to 500deg2
                 #[3.5073431086868616e-10, 6.8084832887309751e-10, 1.5611690339334824e-09, 3.4271570553439895e-09, 6.6527116310331712e-09, 1.1077073588338936e-08, 1.5923395927844532e-08, 2.0550588847112182e-08, 2.455472588661645e-08]

                 # -corresponding to 3814.70deg2
                 [4.5737557453569945e-10, 8.798872187888659e-10, 1.9955608264958949e-09, 4.3410849835151205e-09, 8.350698719910065e-09, 1.3789256781643965e-08, 1.9765148024657066e-08, 2.5363707471031826e-08, 3.02376880308492e-08]
                     )

        yl=array('d',[1.250,1.500,1.750,2.000,2.250,2.500,2.750,3.000,3.250])

        phlow = 0
        phup  = 0
        y     = 4.5
        # --- make a linear interpolation
        for i in range(len(xl)-1):
            xlow = xl[i]
            xup  = xl[i+1]
            if(fx>=xlow and fx<xup):
                phlow = yl[i]
                phup  = yl[i+1]
                break

        if(fx<xl[0]):
            xlow  = xl[0]
            xup   = xl[1]
            phlow = yl[0]
            phup  = yl[1]
            
        if(phlow>0 and phup>0):
            y = (phup-phlow)/(xup-xlow)*(fx-xlow) + phlow
        else:
            y = 3.5

        #print y,phlow,phup,fx
        
        return y
    """------------------------------------"""
    def GetPhIndexCorrection(self,fx,idx):

        #return 1. # not is use

        """derived using the GetPhIndexCorrections.py script"""
        #f  = TF1("flimit","[0]*(log10(x)-[1]) +[2] ")
        #dy = 3.0 -1.6
        #dx = log10(6e-8) - log10(2e-9)
        #f.SetParameters(dy/dx,log10(2e-9),1.6)

        #idx_lim = f.Eval(fx) # --- this is the index up to which
                             #     sources have detected


        idx_lim = self.GetPhIdxLim(fx)

        # --- the photon index distr. can be described
        #     as a Gaussian with
        mean  = 2.43
        sigma = 0.257

        mean = 2.40
        sigma = 0.30

        #mean = 2.45
        #sigma = 0.35

        # --- sources detected 
        r1 = romberg(lambda t: exp(-((t-mean)**2.)/(2.*sigma**2.)),
                     0.,idx_lim)
        # --- all sources
        r2 = romberg(lambda t: exp(-((t-mean)**2.)/(2.*sigma**2.)),
                     0.,8.)

        #print r1,r2

        #norm = sqrt(2.)*sigma
        #x1 = (idx_lim-mean)/( sqrt(2.)*sigma)
        #if(x1<0):
        #    prob = 0.5 *(1.+ TMath.Erf(-x1/sqrt(2.))) 
        #else:
        #    prob = 0.5*(1-TMath.Erf(x1/sqrt(2.)))
        #prob = 1.0- prob


        prob=r1/r2




        return prob
        
    """------------------------------------"""
    def GetAreaFast(self,fx,idx):

        #f2 = p[0]+ p[1]*idx + p[2]*idx*idx+ p[3]*pow(idx,3)
           
        parL=[35.451826181649679, -118.1826617026817, 160.95498110807281,
              -113.8422899039637, 43.737383672569834, -8.5472662226460177,
              0.66479866302097057]

        f2 =0.0
        for i in range(len(parL)):
            f2+=parL[i]*pow(idx,i)

     
        f    = fx/f2
        area  = self.GetAreaInterp(f,2.00000001)

  
        #print "F100=%.1e area=%.1f idx=%.1f Ph_corr=%.3e"%(
        #    fx,area,idx,corr)

        return area
    """------------------------------------"""
    def GetArea(self,fx,idx):

        corr = 1.0
        #corr = self.GetPhIndexCorrection(fx,idx)

        #area = self.GetAreaFast(fx,idx)
        area = self.GetAreaInterp(fx,idx)

        #if(area<3814.70):
        #    return 0
        
        #if(area< self.maxAreaL[0]*0.005):
        #    area = 0.

        #print fx,corr
        return area*self.compl*corr
        
    """------------------------------------"""
    def GetAreaInterp(self,fx,idx):

        
        # --- check MaX areas first: use just the hardest index
        #  to avoid problem: NB it can be set to 1e-6
               
        #corr = self.GetPhIndexCorrection(fx,idx)
        corr =1

        if fx>self.maxFluxL[0]:
            return self.maxAreaL[0]

        if(fx<1e-10):
            return 0
        
        # --- check min and max photon index, and change it if needed
        pmax = self.idxL[self.ndatasets-1]
        pmin = self.idxL[0]

        if(idx>=pmax):
            idx= pmax*0.9999999
        if(idx<=pmin):
            idx=pmin*1.000000000001

        # --- make a check that the requested index does
        #     not coincide exactly with one of the coverage
        for i in range(1,len(self.idxL)-1,1):
            if(idx==self.idxL[i]):
                idx*=1.000000000001
                #print "Changing index"

        
        idL = self.GetIndices(fx,idx)
        if(idL[0]==-1):
            return 0

        area = self.Interpolate(idL,fx,idx)

        #print "F100=%.1e area=%.1f idx=%.1f Ph_corr=%.3e"%(
        #    fx,area,idx,corr)

        # --- do not believe areas which are less than 1% of the total
        #if(area< self.maxAreaL[0]*0.35):
        #    area = 0.
        
        return area
    """ -----------------------------"""
    def Convert(self,emin1,emax1,emin2,emax2,alpha):

        """convert the sky coverage from band 1 to band 2"""

        g  = 1.-alpha
        f1 = pow(emax1,g) - pow(emin1,g)
        f2 = pow(emax2,g) - pow(emin2,g)

        conv = f2/f1

        g     = self.graph
        nbins = g.GetN()
        for i in range(nbins):
            x = 0.1
            y = -0.3
            g.GetPoint(i,x,y)

            x1  = (x +0.0)*conv
            y = y+0.0

            g.SetPoint(i,x1,y)

        # --- convert also the max-flux
        self.maxFlux = conv * self.maxFlux
        return
###########################################################
class SKYCOV_new:
    """------------------------------------"""
    def __init__(self,dataL):


        self.maxArea = dataL[0]
        self.maxFlux = dataL[1]
        self.fluxL   = dataL[2]
        self.areaL   = dataL[3]
        self.errL    = dataL[4]
        self.minFlux = self.fluxL[0]

        
    """------------------------------------"""
    def GetArea(self,fx):

             
        if(fx>=self.maxFlux):
            return self.maxArea
        if(fx<self.minFlux):
            return 0.

             
        # --- interpolate linearly
        fluxL = self.fluxL
        areaL = self.areaL

        for i in range(len(fluxL)-1):
            f1 = fluxL[i]
            f2 = fluxL[i+1]
            a1 = areaL[i]
            a2 = areaL[i+1]

            if(fx>=f1 and fx <f2):
                a = (a2-a1)/(f2-f1)*(fx-f1) + a1
                return a

        print(">>GetArea::ERROR interpolating fx=%.3e minF=%.3 maxF=%.3"%(
            fx,self.minFlux,self.maxFlux))
    """------------------------------------"""
    def GetErr(self,fx):
        if(fx>=self.maxFlux):
            return 0

        if(fx<self.minFlux):
            return self.maxArea # returns 100% error

        # --- interpolate linearly
        fluxL = self.fluxL
        errL  = self.errL
        
        for i in range(len(fluxL)-1):
            f1 = fluxL[i]
            f2 = fluxL[i+1]
            e1 = errL[i]
            e2 = errL[i+1]

            if(fx>=f1 and fx <f2):
                e = (e2-e1)/(f2-f1)*(fx-f1) + e1
                return a,e

        print(">>GetArea::ERROR interp. Error fx=%.3e minF=%.3 maxF=%.3"%(
            fx,self.minFlux,self.maxFlux))
        sys.exit()
    """------------------------------------"""
    def GetAreaErr(self,fx):
        if(fx>=self.maxFlux):
            return [self.maxArea,0]

        if(fx<self.minFlux):
            return [0.,0]

        # --- interpolate linearly
        fluxL = self.fluxL
        areaL = self.areaL
        errL  = self.errL
        
        for i in range(len(fluxL)-1):
            f1 = fluxL[i]
            f2 = fluxL[i+1]
            a1 = areaL[i]
            a2 = areaL[i+1]
            e1 = errL[i]
            e2 = errL[i+1]

            if(fx>=f1 and fx <f2):

                a = (a2-a1)/(f2-f1)*(fx-f1) + a1
                e = (e2-e1)/(f2-f1)*(fx-f1) + e1

                return a,e

        print(">>GetArea::ERROR interpolating fx=%.3e minF=%.3 maxF=%.3"%(
            fx,self.minFlux,self.maxFlux))
        
            
###########################################################
class SKYCOV:
    """------------------------------------"""
    def __init__(self,T_graph,T_F1,maxL):

        self.graph = T_graph
        self.f1    = T_F1
        self.maxArea = maxL[0]
        self.maxFlux = maxL[1]
    """------------------------------------"""
    def GetArea(self,fx):

        if(fx>=self.maxFlux):
            return self.maxArea

        area = self.graph.Eval(fx,0,"S") 
        if(area<0):
            return 0

        return area

    
        if(fx>7.0e-8):
            area = self.graph.Eval(fx,0,"S")
        else:
            area = self.f1.Eval(fx)

        return area
    """ -----------------------------"""
    def Convert(self,emin1,emax1,emin2,emax2,alpha):

        """convert the sky coverage from band 1 to band 2"""

        g  = 1.-alpha
        f1 = pow(emax1,g) - pow(emin1,g)
        f2 = pow(emax2,g) - pow(emin2,g)

        conv = f2/f1

        g     = self.graph
        nbins = g.GetN()
        for i in range(nbins):
            x = 0.1
            y = -0.3
            g.GetPoint(i,x,y)

            x1  = (x +0.0)*conv
            y = y+0.0

            g.SetPoint(i,x1,y)

        # --- convert also the max-flux
        self.maxFlux = conv * self.maxFlux

        return 
###########################################################
class SOURCE:

    """------------------------------------"""
    def __init__(self,name,type,ra,dec,glon,glat,
                 flux,flux_err,lx,lx_err,z,snr,area):

        self.name     = name
        self.type     = type
        self.ra       = ra
        self.dec      = dec
        self.glon     = glon
        self.glat     = glat
        self.flux     = flux
        self.flux_err = flux_err
        self.snr      = snr
        self.z        = z
        self.lx       = lx
        self.lx_err   = lx_err
        self.logLx    = log10(lx)

        self.phI      = 0
        self.phI_err  = 0

        self.area     = area # this is the skycoverage for the given source flux

        self.SKYCOV   = 0 # SKYCOV object
        
        self.Vmax     = 0.
        self.V        = 0.
        self.alpha    = 0.
        self.OmegaLZ  = 0


        self.energy   = 0.# energy in case this is an event
        self.dl       = 0 # luminosity distance
        self.dVdz     = 0.# dV/dz

        self.BHmass   = 6 # standard 10^6 solar mass BH


        self.pdf      = 0 # probability density function for the redshift
        
        return
###########################################################
class SAMPLE:
    """------------------------------------"""
    def __init__(self,objL):

        self.objL = objL
        self.nsrc = len(objL)

        self.graph = TGraph()
        self.gfake = TGraph()
        self.minFlux  = 0
        self.maxFlux  = 0

        self.GetMaxMin()
        print("\n ** Sample has %d sources "%(self.nsrc))

    """ ----------------------------------- """
    def GetMaxMin(self):

        minf = 100
        maxf = -100

        for src in self.objL:
            f = src.flux
            if(f>maxf):
                maxf = f
                
            if(f<minf):
                minf = f

        self.maxFlux = maxf
        self.minFlux = minf

        return
    """ ----------------------------------- """
    def __getitem__(self,i):

        try:
            return self.objL[i]
        except:
            print("ERROR: Sample has only %d objects. You requested obj num %d"%(self.nsrc,i))
            sys.exit()
    """ ----------------------------------- """
    def Remove(self,i):

        try:
            a         = self.objL.pop(i)
            self.nsrc = len(self.objL)
            print("Removed element source %s from Sample"%(a.name))
        except:
            print("Sample has only %d objects. You tried to remove obj num %d"%(self.nsrc,i))
            sys.exit()

    """ -----------------------------"""
    def Convert(self,emin1,emax1,emin2,emax2,alpha):

        """convert the fluxes from band 1 to band 2"""


        for src in self.objL:

            g  = 1.-abs(src.phI)
            if(alpha>=0):
                g  = 1. -alpha

            f1 = pow(emax1,g) - pow(emin1,g)
            f2 = pow(emax2,g) - pow(emin2,g)

            conv = f2/f1
            
            src.flux*=conv
            
        # --- update max/min fluxes
        self.GetMaxMin()
    """ ----------------------------------- """
    def AddSources(self,newSrcL):

        for src in newSrcL:
            self.objL.append(src)

        self.nsrc = len(self.objL)

        print("\n  ** Sample has now %d sources "%(self.nsrc))

    """ ----------------------------------- """
    def  DisplayFPh(self):
        
        nsrc = self.nsrc
        objL = self.objL
        fL  = array('d',[])
        phL  = array('d',[])

        phmin =100
        phmax =-100
        fmin = 1e80
        fmax = -1e80

        for src in objL:
            fL.append(src.flux)
            phL.append(src.phI)

            if(src.flux>fmax):
                fmax=src.flux
            if(src.flux<fmin):
                fmin=src.flux
            if(src.phI>phmax):
                phmax=src.phI
            if(src.phI<phmin):
                phmin=src.phI
            
        x     = array('d',[fmin*0.2,fmax*1.5])
        y     = array('d',[0.5,3.5])
        gfake = TGraph(2,x,y)
        gfake.SetTitle(";F_{100} [ph cm^{-2} s^{-1}]; Photon Index")
        gfake.Draw("AP")

        g = TGraph(nsrc,fL,phL)
        g.SetMarkerStyle(4)
        g.Draw("PSAME")
        
        self.graph = g
        self.gfake = gfake
        
    """------------------------------------ """
    def Display(self):

        nsrc = self.nsrc
        objL = self.objL

        lL = array('d',[])
        zL = array('d',[])

        zmin = 100
        zmax = -100
        lmin = 1e80
        lmax = -100
        
        for src in objL:
            lL.append(src.lx)
            zL.append(src.z)

            if(src.z>zmax):
                zmax=src.z
            if(src.z<zmin):
                zmin=src.z
            if(src.lx>lmax):
                lmax=src.lx
            if(src.lx<lmin):
                lmin=src.lx
            
        x     = array('d',[zmin*0.2,zmax*1.5])
        y     = array('d',[lmin*0.2,lmax*2.0])
        gfake = TGraph(2,x,y)
        gfake.SetTitle(";Redshift;L_{#gamma}")
        gfake.Draw("AP")

        
        g = TGraph(nsrc,zL,lL)
        g.SetMarkerStyle(4)
        #g.SetTitle(";z;Lx")
        g.Draw("PSAME")
        
        self.graph = g
        self.gfake = gfake
        
        #for src in objL:
        #    line = src.OmegaLZ
        #    line.SetLineWidth(1)

            #if(src.snr<5.05 and src.snr>5.0):
            #    line.Draw("same")
            #    print "src.fx=%.3e"%(src.flux)
    """ ----------------------------------- """
    def GetMaxFlux(self,indexL):

        #print indexL

        srcL    = self.objL
        maxf  = -1e6
        index = -1

        
        
        for i in range(len(srcL)):
            src  = srcL[i]
            flux = src.flux
            ok   = 0
            
            for j in indexL:
                if(i==j):
                    ok = 1

            if(flux>maxf and ok ==0):
                maxf = flux
                index = i

        return index
    """ ----------------------------------- """
    def SortFlux(self):

        newSrcL = []
        srcL    = self.objL
        indexL      = []
    
        for src in srcL:
            index = self.GetMaxFlux(indexL)
            indexL.append(index)
            
            newSrcL.append(srcL[index])

            #print "src %d flux %.4e"%(index,srcL[index].flux)

        self.objL = newSrcL
###########################################################
def dVdz_2args(z,dm):

    """ following Hogg et al. 1999 """

    
    
    H0     = g_H0
    dh     = g_dh

  
    E_z = sqrt( g_omegaM*pow(1.0+z,3.0)+g_omegaL ) # This is E(z) as defined in Hogg+00


    dVdz = dh*dm*dm/E_z 

    dl =dm*(1+z)

    return dVdz
###########################################################
def dVdz(z):
 
    dh     = g_dh

    r = romberg(lambda t: 1.0/SQRT( g_omegaM*pow(1.+t,3.)+g_omegaL  ),0.,z)

    #r = quad(lambda t: 1.0/SQRT( g_omegaM*pow(1.+t,3.)+g_omegaL  ),0.,z)[0]

    E_z = sqrt( g_omegaM*pow(1.0+z,3.0)+g_omegaL ) # This is E(z) as defined in Hogg+00

    dm   = dh*r           # in mpc
    dVdz = dh*dm*dm/E_z 


    return dVdz
###########################################################
def GetDL(z):
    """ get luminosity distance """

    #return GetAnDL(z)

    
    H0     = g_H0
    dh     = g_dh
    
    #fint=TF1("fint","1/TMath::Sqrt( 0.3*pow(1+x,3)+0.*pow(1+x,2)+0.7)")
    #r   = fint.Integral(0,z)

    r = romberg(lambda t: 1.0/SQRT( g_omegaM*pow(1.+t,3.)+g_omegaL  ),0.,z)
    
    
    dm  = dh*r         # in mpc
    dl  = dm*(1+z)     # in mpc
    dl  *=3.0831879e24 # in cm

    return dl,dm
###########################################################
def F(z,omegaM):
    """sub-function for computing the analytical DL """


    x = (1-omegaM)/(omegaM * pow(1+z,3))


    val = 1./sqrt(1+z)

    nom = 2+2.641*x + 0.8830*x**2 + 0.05313*x**3
    den = 1+1.392*x + 0.5121*x**2 + 0.03944*x**3

    val *=nom/den

    return val
    
###########################################################
def GetAnDL(z):
    """ Analytic luminosity distance from Adachi& Kasai"""


    H0     = g_H0
    dh     = g_dh

    x = (1-g_omegaM)/g_omegaM

    dl =(g_dh * (1+z)/sqrt(g_omegaM))*(F(0,g_omegaM) - F(z,g_omegaM))

    
    dm  = dl/(1+z)     # in mpc
    dl  *=3.0831879e24 # in cm

    return dl,dm
###########################################################################
def GetKLumin(ferg,z,alpha):

    """ get luminosity with k-correction """

    dh = g_dh

    #fint=TF1("fint","1/TMath::Sqrt( 0.3*pow(1+x,3)+0.*pow(1+x,2)+0.7)")
    #r   = fint.Integral(0,z)


    
    r = romberg(lambda t: 1.0/SQRT( g_omegaM*pow(1.+t,3.)+g_omegaL  ),0.,z)

    dm  = dh*r
    dl  = dm*(1.+z)
    dlc = log10(dl)+24.489

    ferg /=pow(1.+z,2.-alpha)

    try:
        o = log10(ferg)+log10(4.*pi)+2.*dlc
    except:
        print(">>GetKLumin::ERROR: ferg=%.3e "%(ferg))
        sys.exit()

    #print "r=%e dl=%s o=%e fx=%e z=%e"%(r,dl,o,fx,z)
    
    return o
##########################################################################
def GetErgFlux(flux,phI,emin,emax):

    """ returns a flux in erg/cm2/s assuming """

    k    = 1.6021776462e-6  # <-convert from MeV to erg
    g1   = 1.-phI
    g2   = 2.-phI


    if(emin<30 or emin>1e5 or phI<0):
        print("WARNING::Strange emin=%.3e or phI=%.3f"%(emin,phI))
        #sys.exit()


    ferg = flux* (g1/g2)* (pow(emax,g2) - pow(emin,g2)) 

    ferg = ferg * k / (pow(emax,g1) - pow(emin,g1))


    return ferg
##########################################################################
def GetPhFlux(ferg,phI,emin,emax):

    """ returns a flux in ph/cm2/s """
    k    = 1.6021776462e-6  # <-convert from MeV to erg
    g1   = 1.-phI
    g2   = 2.-phI


    if(emin<3 or emin>1e5 or phI<0):
        print("WARNING::Strange emin=%.3e or phI=%.3f"%(emin,phI))
        #sys.exit()



    
    fph = ferg * (g2/g1)* (pow(emax,g1) - pow(emin,g1)) 
    fph = fph/ (pow(emax,g2) - pow(emin,g2)) /k


    return fph
##########################################################################
def GetPhFlux_fromL(lx,z,alpha,emin,emax):

    """ get Photon flux from lx,z, and photon index (k-corr)"""

    if(emin<30 or emin>1e5 or alpha<0):
        print("WARNING::Strange emin=%.3e or alpha=%.3f"%(emin,alpha))
        sys.exit()

    ferg = lx* pow(1.+z,2.0-alpha) 

    dl,dm=GetDL(z)

    ferg/=4.*pi*pow(dl,2.) # this is the flux in erg/cm2/s


    fph = GetPhFlux(ferg,alpha,emin,emax)
    

    return fph,dm
#######################################################################
def GetIndex(infile):

    tok = infile.find("coverage")

    idx = infile[tok+9:tok+13]

    try:
        idx = float(idx)
    except:
        print(">>GetIndex: failed retrieving index for %s (idx=%s)"%(infile,idx))
        sys.exit()

    return idx
###################################################################
def LoadCov(infile,lat_cut):

    idx = GetIndex(infile)

    print(" ***Loading %s with index =%.2f"%(infile,idx))
    
    inputfile=open(infile, 'r')
    data = inputfile.readlines()
  
    inputfile.close()

    nofRows=len(data)
    fluxL=[]
    degL =[]

    flat       = 0
    analytical = 0 
    for i in range(nofRows):

        lineStr   = data[i]
        flux,area = GetCovValues(lineStr)

        if(area<0): # to be safe
            area = 0

#        if(area>=1.0 and flux>0): # flux <0 if the sky coverage is 
#            flat =1               # the analytical one

        if(flat==1):
            area =1.0

        # --- we are dealing with the analytical sky coverage
        if(flux<0):
            flux=pow(10,flux)
            analytical = 1
            
        fluxL.append(flux)
        degL.append(area)

    # ---revert the array if the sky coverage is analytical
    if(analytical==1):
        fluxL.reverse()
        degL.reverse()
        
    
    maxArea   = max(degL[len(degL)-1],degL[0])

    theta     = pi*(90.-lat_cut)/180.
    tot_area  = 4*pi*(1.-cos(theta))*(180/pi)**2
    if(analytical==0):
        # --- multiply by the geometric area
        theta     = pi*(90.-lat_cut)/180.
        tot_area  = 4*pi*(1.-cos(theta))*(180/pi)**2
        maxArea   *=tot_area
    
        for i in range(len(degL)):
            degL[i]*=tot_area

    print("Tot_area is %.1f -- maxArea is %.1f"%(tot_area,maxArea ))
        
    maxFlux = max(fluxL[len(fluxL)-1],fluxL[0])


    # --- renormalize the analytical sky coverage to the right val
    if(maxArea!=tot_area and analytical==1):
        print("Renormalizing the area to %.1f"%(tot_area))
        renorm  = tot_area/maxArea
        #print renorm
        maxArea = tot_area
        for i in range(len(degL)):
            degL[i]*=renorm

    if(maxArea!=tot_area):
        maxArea=tot_area
        
    return [idx,maxArea,maxFlux,fluxL,degL]
    
###################################################################
def LoadSkyCov2D(indir,lat_cut):


    skyC = SKYCOV2D()

    #fileL = GetFileType(indir,"interpolatedF200.") # remember the "."
    fileL = GetFileType(indir,"coverage_") 

    for infile in fileL:
        dataL = LoadCov(infile,lat_cut)
       

        skyC.AddCoverage(dataL)

    return skyC
##############################################################
def GetCovValues(lineStr):


    tok=lineStr.find("\t")

    if(tok<0):
        tok=lineStr.find(" ")
        if(tok<0):
            print("tok =%d in string %s"%(tok,lineStr))

    flux=float(lineStr[:tok])
    try:
        deg = float(lineStr[tok+1:])
    except:
        print(">>GetValues::ERROR reading value from line %s"%(lineStr))
        sys.exit()
        
    #print "%f %f"%(flux,deg)
    
    return [flux,deg]
##############################################################
########################################################################
def ScanHeader(header,key1,key2,key3):

    """ scan the header for a particular keywords and gives it back
    NB:key3 is an anti-match
    """

    hdr = header
    kL = hdr.items()

    keyname = "none"

    for i in range(len(kL)):

        keyname = str(kL[i][1])

        if (keyname.find(key1)>=0 and
            keyname.find(key2)>=0 and
            keyname.find(key3)<0
            ):

            

            print("Keyanme =%s -- key1 was %s and key2 was %s"%(keyname,key1,key2))
            return keyname

    print("Keyword matching %s and %s not found"%(key1,key2))
    return "none"
########################################################################
def Pause():
    # ----- wait for user input
    rep = raw_input( '\n\nenter "q" to quit: ' )
    
#########################################################################
