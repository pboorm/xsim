"""
Streamlit app to host AGN spectral model generation
Copyright: Adam Hill (2020)

TODO:
- select different models
- dynamic geometry
- worth pursuing interpolator? (probably not -- just requires user to have PyXspec)
- plotly tick marks
- plotly background colour
- streamlit location of sections on screen (i.e. margins etc)
- alternative to PyXspec?
- how to do a step plot
- how to step in log for sliders
- LaTex and y-axis label?
- auto set the width of the axes depending on screen width
"""

import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
from xspec import *
from astropy.cosmology import FlatLambdaCDM

Boorman_cosmo = FlatLambdaCDM(H0 = 67.3, Om0 = 0.315)

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

Fit.statMethod = 'cstat'
Xset.abund = "Wilm"
# Xset.parallel.error = 4
Plot.xLog = True
Plot.yLog = True
# Plot.add = True
Plot.xAxis = "keV"
# Xset.cosmo = "67.3,,0.685"
Plot.device = "/null"
# Fit.query = "yes"

def dummy(xlim):
    """
    Create dummy response
    """
    AllData.dummyrsp(lowE = xlim[0], highE = xlim[1], nBins = 10 ** 3, scaleType = "log")
    Plot("model")

def DL(z, units = "cm"):
    """
    Luminosity distance
    """
    dist = Boorman_cosmo.luminosity_distance(z).value * (10 ** 6) * (3.086 * 10 ** 18)##cm
    if units == "cm":
        dist_final = dist
    elif units == "Mpc":
        dist_final = dist / (10. ** 6) / (3.086 * 10. ** 18)
    return dist_final

def getNorm(logL210, PhoIndex, redshift):
    """
    Derives intrinsic powerlaw normalisation
    """
    DLcm = DL(redshift, "cm")
    F210 = (10 ** logL210) / (4. * np.pi * (DLcm) ** 2)
    
    ce = 1.602176565E-9
    norm = 0.
    if PhoIndex == 2.:
        norm = F210 / (ce * np.log(5.))
    else:
        B = F210 * (2. - PhoIndex) / ce
        norm = B / ((10. ** (2. - PhoIndex)) - (2. ** (2. - PhoIndex)))
    return norm

def generate_borus02_spectra(PhoIndex, Ecut, logNHtor, CFtor, thInc, A_Fe, fscatt, logL210, redshift):
    """
    This generates the function for the obscured borus02 model setup
    Args:
        PhoIndex: float, powerlaw slope of the intrinsic spectrum (1.45--2.55)
        
        Ecut: float, high-energy exponentional cut-off of the intrinsic powerlaw (100.--1000.)

        logNHtor: float, logarithm of the column density of the obscurer (22.--22.5)
        
        CFtor: float, covering factor of the obscurer (0.15--0.95)
        
        thInc: float, inclination angle of the obscurer (20.--85.), note: edge-on = 90.

        A_Fe: float, abundance of iron in the obscurer (0.1--10.)
        
        fscatt: float, percentage of scattered emission in the warm mirror (1.e-5--1.e-1)
    Returns:
        dataframe: a dataframe with columns for the energy in keV, the transmitted X-ray flux,
                   the reprocessed X-ray flux, the Thomson-scattered X-ray flux, and the total X-ray flux
    """
    mainModel = "constant*TBabs(atable{./xs_models/borus02_v170323b.fits} + zTBabs*cabs*cutoffpl + constant*cutoffpl)"
    m = Model(mainModel)
    dummy([1., 200.])

    intrinsic_NORM = getNorm(logL210, PhoIndex, redshift)

    ## set default values
    AllModels(1)(1).values = 1., -1.
    AllModels(1)(2).values = 0.01, -1.
    AllModels(1)(3).values = PhoIndex ##PhoIndex
    AllModels(1)(4).values = Ecut ##Ecut
    AllModels(1)(5).values = logNHtor ##logNHtor
    AllModels(1)(6).values = CFtor ##CFtor
    AllModels(1)(7).values = thInc ##thInc
    AllModels(1)(8).values = A_Fe ##A_Fe
    AllModels(1)(9).values = redshift ##redshift
    AllModels(1)(10).values = intrinsic_NORM ##REPR norm
    AllModels(1)(11).link = "10.^(p5 - 22.)"
    AllModels(1)(12).link = "p9"
    AllModels(1)(13).link = "p11"
    AllModels(1)(14).link = "p3"
    AllModels(1)(15).link = "p4"
    AllModels(1)(16).values = intrinsic_NORM ##TRANS norm
    AllModels(1)(17).values = fscatt ##fscatt
    AllModels(1)(18).link = "p3"
    AllModels(1)(19).link = "p4"
    AllModels(1)(20).values = intrinsic_NORM ##SCATT norm
    AllModels.show()

    wdata = pd.DataFrame(data = {"E_keV": Plot.x()})
    
    AllModels(1)(10).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Transmitted"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(10).values = intrinsic_NORM
    AllModels(1)(20).values = intrinsic_NORM
    
    AllModels(1)(16).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Reprocessed"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(16).values = intrinsic_NORM
    AllModels(1)(20).values = intrinsic_NORM
    
    AllModels(1)(10).values = 0.
    AllModels(1)(16).values = 0.
    Plot("eemodel")
    pardf_name = "Scattered"
    wdata.loc[:, pardf_name] = Plot.model()

    wdata.loc[:, "Total"] = wdata[["Transmitted", "Reprocessed", "Scattered"]].sum(axis = 1)
    AllModels(1)(10).values = intrinsic_NORM
    AllModels(1)(16).values = intrinsic_NORM
    return wdata

def generate_R17B1_spectra(logNH, PhoIndex, HighECut, rel_refl, abund, Fe_abund, inc_ang, fscatt, logL210, redshift):
    """
    This generates the function for the obscured pexrav model setup
    Args:
        logNH: float, logarithm of the column density of the obscurer (20.--26.)
        
        PhoIndex: float, powerlaw slope of the intrinsic spectrum (1.--3.)
        
        HighEcut: float, high-energy exponentional cut-off of the intrinsic powerlaw (1.--500.)

        rel_refl: float, relative reflection fraction (equivalent to just negative values in pexrav)
        
        abund: float, abundance of elements below Fe in the obscurer (0.--100.)

        Fe_abund: float, abundance of iron in the obscurer (0.--100.)
        
        inc_ang: float, inclination angle of the slab (20.--85.), note: edge-on = 90.
        fscatt: float, percentage of scattered emission in the warm mirror (1.e-5--1.e-1)
    Returns:
        dataframe: a dataframe with columns for the energy in keV, the transmitted X-ray flux,
                   the reprocessed X-ray flux, the Thomson-scattered X-ray flux, and the total X-ray flux
    """
    mainModel = "constant*TBabs(zTBabs*cabs*cutoffpl + pexrav + constant*cutoffpl)"
    m = Model(mainModel)
    dummy([0.1, 200.])

    intrinsic_NORM = getNorm(logL210, PhoIndex, redshift)

    AllModels(1)(1).values = 1., -1.
    AllModels(1)(2).values = 0.01, -1.
    AllModels(1)(3).values = 10 ** (logNH - 22.) ##nHe22
    AllModels(1)(4).values = redshift ##redshift
    AllModels(1)(5).link = "p3"
    AllModels(1)(6).values = PhoIndex ##PhoIndex
    AllModels(1)(7).values = HighECut ##HighECut
    AllModels(1)(8).values = intrinsic_NORM ##TRANS norm
    AllModels(1)(9).link = "p6"
    AllModels(1)(10).link = "p7"
    AllModels(1)(11).values = -1., 0.01, -100., -100., -0.1, -0.1
    AllModels(1)(11).values = rel_refl
    AllModels(1)(12).link = "p4" ##Redshift
    AllModels(1)(13).values = abund ##abund
    AllModels(1)(14).values = Fe_abund ##Fe_abund
    AllModels(1)(15).values = np.cos(inc_ang * np.pi / 180.) ##cosIncl
    AllModels(1)(16).values = intrinsic_NORM ##REPR norm
    AllModels(1)(17).values = fscatt ##fscatt
    AllModels(1)(18).link = "p6"
    AllModels(1)(19).link = "p7"
    AllModels(1)(20).values = intrinsic_NORM ## SCATT norm

    wdata = pd.DataFrame(data = {"E_keV": Plot.x()})
    
    AllModels(1)(16).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Transmitted"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(16).values = intrinsic_NORM
    AllModels(1)(20).values = intrinsic_NORM
    
    AllModels(1)(8).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Reprocessed"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(8).values = intrinsic_NORM
    AllModels(1)(20).values = intrinsic_NORM
    
    AllModels(1)(8).values = 0.
    AllModels(1)(16).values = 0.
    Plot("eemodel")
    pardf_name = "Scattered"
    wdata.loc[:, pardf_name] = Plot.model()

    wdata.loc[:, "Total"] = wdata[["Transmitted", "Reprocessed", "Scattered"]].sum(axis = 1)
    AllModels(1)(8).values = intrinsic_NORM
    AllModels(1)(16).values = intrinsic_NORM
    return wdata

def generate_uxclumpy_spectra(logNH, PhoIndex, Ecut, TORsigma, CTKcover, Theta_inc, fscatt, logL210, redshift):
    """
    This generates the function for the obscured uxclumpy model setup
    Args:
        logNH: float, logarithm of the column density of the obscurer (20.--26.)
        
        PhoIndex: float, powerlaw slope of the intrinsic spectrum (1.--3.)
        
        HighEcut: float, high-energy exponentional cut-off of the intrinsic powerlaw (60.--400.)

        rel_refl: float, relative reflection fraction (equivalent to just negative values in pexrav) (-100.-- -0.1)
        
        abund: float, abundance of elements below Fe in the obscurer (0.--100.)

        Fe_abund: float, abundance of iron in the obscurer (0.--100.)
        
        inc_ang: float, inclination angle of the slab (20.--85.), note: edge-on = 90.

        fscatt: float, percentage of scattered emission in the warm mirror (1.e-5--1.e-1)
    Returns:
        dataframe: a dataframe with columns for the energy in keV, the transmitted X-ray flux,
                   the reprocessed X-ray flux, the Thomson-scattered X-ray flux, and the total X-ray flux
    """
    mainModel = "constant*TBabs(atable{./xs_models/uxclumpy-cutoff-reflect.fits} + atable{./xs_models/uxclumpy-cutoff-transmit.fits} + constant*atable{./xs_models/uxclumpy-cutoff-omni.fits})"
    m = Model(mainModel)
    dummy([0.1, 200.])

    intrinsic_NORM = getNorm(logL210, PhoIndex, redshift)

    AllModels(1)(1).values = 1., -1.
    AllModels(1)(2).values = 0.01, -1.
    AllModels(1)(3).values = 10 ** (logNH - 22.) ##nHe22
    AllModels(1)(4).values = PhoIndex ##PhoIndex
    AllModels(1)(5).values = Ecut ##HighECut
    AllModels(1)(6).values = TORsigma ##TORsigma
    AllModels(1)(7).values = CTKcover ##CTKcover
    AllModels(1)(8).values = Theta_inc ##Theta_inc
    AllModels(1)(9).values = redshift ##z
    AllModels(1)(10).values = intrinsic_NORM ##REPRO norm
    AllModels(1)(11).link = "p3"
    AllModels(1)(12).link = "p4"
    AllModels(1)(13).link = "p5"
    AllModels(1)(14).link = "p6"
    AllModels(1)(15).link = "p7"
    AllModels(1)(16).link = "p8"
    AllModels(1)(17).link = "p9"
    AllModels(1)(18).values = intrinsic_NORM ##TRANS norm
    AllModels(1)(19).values = fscatt ##fscatt
    AllModels(1)(20).link = "p3"
    AllModels(1)(21).link = "p4"
    AllModels(1)(22).link = "p5"
    AllModels(1)(23).link = "p6"
    AllModels(1)(24).link = "p7"
    AllModels(1)(25).link = "p8"
    AllModels(1)(26).link = "p9"
    AllModels(1)(27).values = intrinsic_NORM ##SCATT norm

    wdata = pd.DataFrame(data = {"E_keV": Plot.x()})
    
    AllModels(1)(10).values = 0.
    AllModels(1)(27).values = 0.
    Plot("eemodel")
    pardf_name = "Transmitted"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(10).values = intrinsic_NORM
    AllModels(1)(27).values = intrinsic_NORM
    
    AllModels(1)(18).values = 0.
    AllModels(1)(27).values = 0.
    Plot("eemodel")
    pardf_name = "Reprocessed"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(18).values = intrinsic_NORM
    AllModels(1)(27).values = intrinsic_NORM
    
    AllModels(1)(10).values = 0.
    AllModels(1)(18).values = 0.
    Plot("eemodel")
    pardf_name = "Scattered"
    wdata.loc[:, pardf_name] = Plot.model()

    wdata.loc[:, "Total"] = wdata[["Transmitted", "Reprocessed", "Scattered"]].sum(axis = 1)
    AllModels(1)(10).values = intrinsic_NORM
    AllModels(1)(18).values = intrinsic_NORM
    return wdata

def generate_mytorus_spectra(logNH, IncAng, PhoIndex, fscatt, logL210, redshift):
    """
    This generates the function for the obscured uxclumpy model setup
    Args:
        logNH: float, logarithm of the column density of the obscurer (20.--26.)
        
        PhoIndex: float, powerlaw slope of the intrinsic spectrum (1.--3.)
        
        HighEcut: float, high-energy exponentional cut-off of the intrinsic powerlaw (60.--400.)

        rel_refl: float, relative reflection fraction (equivalent to just negative values in pexrav) (-100.-- -0.1)
        
        abund: float, abundance of elements below Fe in the obscurer (0.--100.)

        Fe_abund: float, abundance of iron in the obscurer (0.--100.)
        
        inc_ang: float, inclination angle of the slab (20.--85.), note: edge-on = 90.

        fscatt: float, percentage of scattered emission in the warm mirror (1.e-5--1.e-1)
    Returns:
        dataframe: a dataframe with columns for the energy in keV, the transmitted X-ray flux,
                   the reprocessed X-ray flux, the Thomson-scattered X-ray flux, and the total X-ray flux
    """
    mainModel = "constant*TBabs(etable{./xs_models/mytorus_Ezero_v00.fits}*powerlaw + atable{./xs_models/mytorus_scatteredH500_v00.fits} + atable{./xs_models/mytl_V000010nEp000H500_v00.fits} + constant*powerlaw)"
    m = Model(mainModel)
    dummy([0.5, 200.])

    intrinsic_NORM = getNorm(logL210, PhoIndex, redshift)

    AllModels(1)(1).values = 1., -1.
    AllModels(1)(2).values = 0.01, -1.
    AllModels(1)(3).values = 10 ** (logNH - 24.) ##nHe24
    AllModels(1)(4).values = IncAng ##IncAng
    AllModels(1)(5).values = redshift ##z
    AllModels(1)(6).values = PhoIndex ##PhoIndex
    AllModels(1)(7).values = intrinsic_NORM ## TRANS norm
    AllModels(1)(8).link = "p3" ##nHe24
    AllModels(1)(9).link = "p4" ##IncAng
    AllModels(1)(10).link = "p6" ##PhoIndx
    AllModels(1)(11).link = "p5" ##z
    AllModels(1)(12).values = intrinsic_NORM ##REPR_S norm
    AllModels(1)(13).link = "p3" ##nHe24
    AllModels(1)(14).link = "p4" ##IncAng
    AllModels(1)(15).link = "p6" ##PhoIndx
    AllModels(1)(16).link = "p5" ##z
    AllModels(1)(17).values = intrinsic_NORM ##REPR_L norm
    AllModels(1)(18).values = fscatt ##fscatt
    AllModels(1)(19).link = "p6"##PhoIndex
    AllModels(1)(20).values = intrinsic_NORM ##SCATT norm

    wdata = pd.DataFrame(data = {"E_keV": Plot.x()})
    
    AllModels(1)(12).values = 0.
    AllModels(1)(17).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Transmitted"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(12).values = intrinsic_NORM
    AllModels(1)(17).values = intrinsic_NORM
    AllModels(1)(20).values = intrinsic_NORM
    
    AllModels(1)(7).values = 0.
    AllModels(1)(17).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Scattered Continuum"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(7).values = intrinsic_NORM
    AllModels(1)(17).values = intrinsic_NORM
    AllModels(1)(20).values = intrinsic_NORM
    
    AllModels(1)(7).values = 0.
    AllModels(1)(12).values = 0.
    AllModels(1)(20).values = 0.
    Plot("eemodel")
    pardf_name = "Fluorescence Lines"
    wdata.loc[:, pardf_name] = Plot.model()
    AllModels(1)(7).values = intrinsic_NORM
    AllModels(1)(12).values = intrinsic_NORM
    AllModels(1)(20).values = intrinsic_NORM
    
    AllModels(1)(7).values = 0.
    AllModels(1)(12).values = 0.
    AllModels(1)(17).values = 0.
    Plot("eemodel")
    pardf_name = "Scattered"
    wdata.loc[:, pardf_name] = Plot.model()

    wdata.loc[:, "Total"] = wdata[["Transmitted", "Scattered Continuum", "Fluorescence Lines", "Scattered"]].sum(axis = 1)
    AllModels(1)(7).values = intrinsic_NORM
    AllModels(1)(12).values = intrinsic_NORM
    AllModels(1)(17).values = intrinsic_NORM
    return wdata



DEGREE_SYMBOL = "\N{DEGREE SIGN}"

model_choice = st.sidebar.selectbox('Choose a model', ('uxclumpy', 'mytorus[coupled]', 'borus02', 'pexrav (R17B1)'))

st.sidebar.title("Parameters")

# Schematic info
# st.sidebar.subheader("AGN Geometry")
# st.sidebar.image("assets/schematic.png", use_column_width=True)



# Add some context in the main window

if model_choice == "borus02":
    st.title("${\\tt borus02}$ X-ray Simulator")
    st.subheader("${\\tt const}\\times {\\tt TBabs}({\\tt borus02\\_v170323b} + {\\tt zTBabs}\\times {\\tt cabs}\\times {\\tt cutoffpl} + {\\tt fscatt}\\times {\\tt cutoffpl})$")
    # Controllers
    PhoIndex_c = st.sidebar.slider(
        "Photon Index (PhoIndex)",
        min_value=1.45,
        max_value=2.55,
        value = 1.8,
        step=0.1,
        format="%.1f",
        key="PhoIndex",
    )
    Ecut_c = st.sidebar.slider(
        "High-energy cut-off (Ecut)",
        min_value=100.,
        max_value=1000.,
        value = 270.,
        step=10.,
        format="%.0f",
        key="Ecut",
    )
    logNHtor_c = st.sidebar.slider(
        "logNH (logNHtor)",
        min_value=22.,
        max_value=25.5,
        value = 24.,
        step=0.1,
        format="%.1f",
        key="logNHtor",
    )
    CFtor_c = st.sidebar.slider(
        "Torus Covering Factor (CFtor)",
        min_value=0.15,
        max_value=0.95,
        value = 0.5,
        step=0.1,
        format="%.1f",
        key="CFtor",
    )
    thInc_c = st.sidebar.slider(
        "Inclination Angle (thInc)",
        min_value=20.,
        max_value=85.0,
        value = 60.,
        step=1.,
        format=f"%.0f{DEGREE_SYMBOL}",
        key="thInc",
    )

    A_Fe_c = st.sidebar.slider(
        "Iron Abundance (A_Fe)",
        min_value=0.1,
        max_value=10.,
        value = 1.,
        step=0.1,
        format="%.1f",
        key="A_Fe",
    )

    factor_c = st.sidebar.slider(
        "Scattered Fraction (fscatt)",
        min_value=1.e-5,
        max_value=1.e-1,
        value = 1.e-3,
        step=1.e-5,
        format="%.5f",
        key="factor",
    )

    redshift_c = st.sidebar.slider(
        "Redshift",
        min_value=0.001,
        max_value=1.,
        value = 0.01,
        step=0.001,
        format="%.3f",
        key="redshift",
    )

    logL210_c = st.sidebar.slider(
        "log(L2-10), Intrinsic",
        min_value=39.,
        max_value=45.,
        value = 42.,
        step=1.,
        format="%.0f",
        key="logL210",
    )
    # generate our data
    df = generate_borus02_spectra(PhoIndex_c, Ecut_c, logNHtor_c, CFtor_c, thInc_c, A_Fe_c, factor_c, logL210_c, redshift_c)
    
elif model_choice == "pexrav (R17B1)":
    st.title("${\\tt pexrav}$ X-ray Simulator")
    st.subheader("${\\tt const} \\times {\\tt TBabs} \\times {\\tt zTBabs}({\\tt pexrav} + {\\tt zTBabs} \\times {\\tt cabs} \\times {\\tt cutoffpl} + {\\tt fscatt} \\times {\\tt cutoffpl})$")
    # Controllers
    logNH_c = st.sidebar.slider(
        "logNH",
        min_value=20.,
        max_value=26.,
        value = 23.,
        step=0.1,
        format="%.1f",
        key="logNH",
    )
    PhoIndex_c = st.sidebar.slider(
        "Photon Index (PhoIndex)",
        min_value=1.,
        max_value=3.,
        value = 1.8,
        step=0.1,
        format="%.1f",
        key="PhoIndex",
    )
    HighECut_c = st.sidebar.slider(
        "High-energy cut-off (HighECut)",
        min_value=1.,
        max_value=500.,
        value = 300.,
        step=1.,
        format="%.0f",
        key="Ecut",
    )
    
    rel_refl_c = st.sidebar.slider(
        "Relative Reflection (|R|)",
        min_value=0.1,
        max_value=100.,
        value = 1.,
        step=0.1,
        format="%.1f",
        key="thInc",
    )

    abund_c = st.sidebar.slider(
        "Abundance (abund)",
        min_value=0.,
        max_value=100.,
        value = 1.,
        step=0.1,
        format="%.1f",
        key="A_Fe",
    )

    Fe_abund_c = st.sidebar.slider(
        "Iron Abundance (Fe_abund)",
        min_value=0.,
        max_value=100.0,
        value = 1.,
        step=0.1,
        format=f"%.0f{DEGREE_SYMBOL}",
        key="thInc",
    )

    inc_ang_c = st.sidebar.slider(
        "Inclination Angle",
        min_value=20.,
        max_value=85.0,
        value = 60.,
        step=1.,
        format=f"%.0f{DEGREE_SYMBOL}",
        key="thInc",
    )

    fscatt_c = st.sidebar.slider(
        "Scattered Fraction (fscatt)",
        min_value=1.e-5,
        max_value=1.e-1,
        value = 1.e-3,
        step=1.e-5,
        format="%.5f",
        key="fscatt",
    )

    redshift_c = st.sidebar.slider(
        "Redshift",
        min_value=0.001,
        max_value=1.,
        value = 0.01,
        step=0.001,
        format="%.3f",
        key="redshift",
    )

    logL210_c = st.sidebar.slider(
        "log(L2-10), Intrinsic",
        min_value=39.,
        max_value=45.,
        value = 42.,
        step=1.,
        format="%.0f",
        key="logL210",
    )
    # generate our data
    df = generate_R17B1_spectra(logNH_c, PhoIndex_c, HighECut_c, -rel_refl_c, abund_c, Fe_abund_c, inc_ang_c, fscatt_c, logL210_c, redshift_c)
    
elif model_choice == "mytorus[coupled]":
    st.title("${\\tt mytorus[coupled]}$ X-ray Simulator")
    st.subheader("${\\tt const}\\times {\\tt TBabs}({\\tt Ezero\\_v00} + {\\tt scatteredH500\\_v00} + {\\tt fscatt} \\times {\\tt V000010nEp000H500\\_v00})$")
    # Controllers
    logNH_c = st.sidebar.slider(
        "logNH",
        min_value=20.,
        max_value=25.,
        value = 23.,
        step=0.1,
        format="%.1f",
        key="logNH",
    )

    IncAng_c = st.sidebar.slider(
        "Inclination Angle",
        min_value=0.,
        max_value=90.0,
        value = 75.,
        step=1.,
        format=f"%.0f{DEGREE_SYMBOL}",
        key="IncAng",
    )

    PhoIndex_c = st.sidebar.slider(
        "Photon Index (PhoIndex)",
        min_value=1.,
        max_value=3.,
        value = 1.8,
        step=0.1,
        format="%.1f",
        key="PhoIndex",
    )

    fscatt_c = st.sidebar.slider(
        "Scattered Fraction (fscatt)",
        min_value=1.e-5,
        max_value=1.e-1,
        value = 1.e-3,
        step=1.e-5,
        format="%.5f",
        key="fscatt",
    )

    redshift_c = st.sidebar.slider(
        "Redshift",
        min_value=0.001,
        max_value=1.,
        value = 0.01,
        step=0.001,
        format="%.3f",
        key="redshift",
    )

    logL210_c = st.sidebar.slider(
        "log(L2-10), Intrinsic",
        min_value=39.,
        max_value=45.,
        value = 42.,
        step=1.,
        format="%.0f",
        key="logL210",
    )
    # generate our data
    df = generate_mytorus_spectra(logNH_c, IncAng_c, PhoIndex_c, fscatt_c, logL210_c, redshift_c)
    
elif model_choice == "uxclumpy":
    st.title("${\\tt uxclumpy}$ X-ray Simulator")
    st.subheader("${\\tt const}\\times {\\tt TBabs}({\\tt uxclumpy\\_reflect} + {\\tt uxclumpy\\_transmit} + {\\tt fscatt} \\times {\\tt uxclumpy\\_omni})$")
    # Controllers
    logNH_c = st.sidebar.slider(
        "logNH",
        min_value=20.,
        max_value=26.,
        value = 23.,
        step=0.1,
        format="%.1f",
        key="logNH",
    )
    PhoIndex_c = st.sidebar.slider(
        "Photon Index (PhoIndex)",
        min_value=1.,
        max_value=3.,
        value = 1.8,
        step=0.1,
        format="%.1f",
        key="PhoIndex",
    )
    Ecut_c = st.sidebar.slider(
        "High-energy cut-off (Ecut)",
        min_value=60.,
        max_value=400.,
        value = 300.,
        step=1.,
        format="%.0f",
        key="Ecut",
    )
    TORsigma_c = st.sidebar.slider(
        "Torus Opening Angle (TORsigma)",
        min_value=0.,
        max_value=84.,
        value = 60.,
        step=1.,
        format="%.0f",
        key="TORsigma",
    )
    CTKcover_c = st.sidebar.slider(
        "Compton-thick Cover fraction",
        min_value=0.,
        max_value=0.6,
        value = 0.3,
        step=0.05,
        format="%.2f",
        key="CTKcover",
    )
    Theta_inc_c = st.sidebar.slider(
        "Inclination Angle (Theta_inc)",
        min_value=0.,
        max_value=90.0,
        value = 60.,
        step=1.,
        format=f"%.0f{DEGREE_SYMBOL}",
        key="Theta_inc",
    )
    fscatt_c = st.sidebar.slider(
        "Scattered Fraction (fscatt)",
        min_value=1.e-5,
        max_value=1.e-1,
        value = 1.e-3,
        step=1.e-5,
        format="%.5f",
        key="fscatt",
    )

    redshift_c = st.sidebar.slider(
        "Redshift",
        min_value=0.001,
        max_value=1.,
        value = 0.01,
        step=0.001,
        format="%.3f",
        key="redshift",
    )

    logL210_c = st.sidebar.slider(
        "log(L2-10), Intrinsic",
        min_value=39.,
        max_value=45.,
        value = 42.,
        step=1.,
        format="%.0f",
        key="logL210",
    )
    # generate our data
    df = generate_uxclumpy_spectra(logNH_c, PhoIndex_c, Ecut_c, TORsigma_c, CTKcover_c, Theta_inc_c, fscatt_c, logL210_c, redshift_c)

AllModels(1).show()
AllModels.calcFlux("2. 10. " + str(redshift_c))
F210_obs = AllModels(1).flux[0]
logL210_obs = np.log10(F210_obs * 4. * np.pi * DL(redshift_c, "cm") ** 2)

st.sidebar.text("log(L2-10), Observed = %.2f" %(logL210_obs))
ymin = df["Total"].min()
xmin = df["E_keV"].min()

if model_choice == "mytorus[coupled]":
    mod_df = df.melt(
        id_vars=["E_keV"], value_vars=["Transmitted", "Scattered Continuum", "Fluorescence Lines", "Scattered", "Total"]
    )

else:
    mod_df = df.melt(
        id_vars=["E_keV"], value_vars=["Transmitted", "Reprocessed", "Scattered", "Total"]
    )

mod_df = mod_df.rename(columns={"variable": "Model Component", "value": "Flux"})
# Construct our plot
fig = px.line(
    mod_df,
    x="E_keV",
    y="Flux",
    color="Model Component",
    log_x=True,
    log_y=True,
    width=1000,
    height=700,
    labels=dict(Flux="EFE / keV s-1 cm-2", Energy="Energy / keV"),
)
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), yaxis=dict(range=[np.log10(ymin), 1.]), xaxis=dict(range=[np.log10(xmin), np.log10(200.)]))

st.plotly_chart(fig)

st.sidebar.markdown("### Model outputs")
if st.sidebar.checkbox("Show Table", False):
    st.subheader("Raw Data Table")
    st.write(df, index=False)

# Some advertising
st.sidebar.markdown("Designed by: Dr Peter Boorman & Dr Adam Hill")
