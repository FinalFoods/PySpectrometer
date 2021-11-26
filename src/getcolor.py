import csv
import colour 
from colour.plotting import *
from colour.colorimetry import *

fname="FF-26-11-2021-12:32:52.csv"
data_sample = {}
with open(fname, newline='') as pscfile:
    reader = csv.DictReader(pscfile)
    for row in reader:
        data_sample[float(row['Wavelength'])] = int(row['Intensity'])

#print(data_sample)

sd = colour.SpectralDistribution(data_sample, name="Sample")
#print(repr(sd))

sd_reshaped = reshape_sd(sd, shape=SpectralShape(360, 780, 1), method='Align')

# convert to tristimulus value
cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
illumant = colour.SDS_ILLUMINANTS['D65']

# calculating the sample spectral distribution *CIE XYZ* tristimulus values
XYZ = colour.sd_to_XYZ(sd_reshaped, cmfs, illumant) 
print("XYZ", XYZ)

# convert to RGB
RGB = colour.XYZ_to_sRGB(XYZ / 100)
print("RGB", RGB)

# convert to L*a*b*
LAB = colour.XYZ_to_Lab(XYZ) #, illuminant=illumant)
print("L*a*b*", LAB)

plot_single_colour_swatch(
    ColourSwatch('Spectrum', RGB),
    text_kwargs={'size': 'x-large'}
)

plot_single_sd(sd)
