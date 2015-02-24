"""
Created on: Wed Nov 14 09:58:53 2013
Author: Nick Silverman
Email: nick.silverman11@gmail.com
Description: Driver file to calculate and plot spatial statistics
"""

#==============================================================================
# Import modules
#%%============================================================================
import numpy as np                                                                                                   
import matplotlib.pyplot as plt                                                
import os
from scipy import stats

#==============================================================================
# Bring in data
#%%============================================================================
HOME = os.path.expanduser("~/")
data_path = HOME+"Copy/workspace/bayes_unc_precip/data/"


sig_aspect = np.loadtxt(data_path+"sig_aspect2.csv")
unc_aspect = np.loadtxt(data_path+"unc_aspect2.csv")
dom_aspect = np.loadtxt(data_path+"dom_aspect2.csv")
sig_elev = np.loadtxt(data_path+"sig_elev2.csv")
unc_elev = np.loadtxt(data_path+"unc_elev2.csv")
dom_elev = np.loadtxt(data_path+"dom_elev2.csv")

#==============================================================================
# Calculate angle statistics
#%%============================================================================

def mean_angle(deg):
    """
    Calculates the mean of an array of angles in degrees
    Method is from wikipedia directional statistics
    """
    rad = np.deg2rad(deg)
    s = np.sum(np.sin(rad))/len(deg)
    c = np.sum(np.cos(rad))/len(deg)
    deg_mean = np.arctan2(s, c)
    deg_mean = np.rad2deg(deg_mean)
    return deg_mean

def std_angle(deg):
    """
    Calculates the standard deviation of an array of angles in degrees
    Method is from wikipedia directional statistics    
    """
    rad = np.deg2rad(deg)
    sins = np.sum(np.sin(rad))/len(deg)
    coss = np.sum(np.cos(rad))/len(deg)
    std = np.sqrt(-np.log(sins*sins+coss*coss))
    return np.rad2deg(std)
        
sig_aspect_mean = mean_angle(sig_aspect)
dom_aspect_mean = mean_angle(dom_aspect)
unc_aspect_mean = mean_angle(unc_aspect)
aspect_mean = mean_angle([sig_aspect_mean, dom_aspect_mean])
sig_aspect_std = std_angle(sig_aspect)
unc_aspect_std = std_angle(unc_aspect)
aspect_std = std_angle(np.hstack((sig_aspect, dom_aspect)))

nosig_rad = np.log(np.shape(dom_aspect)[0])
sig_rad = np.log(np.shape(sig_aspect)[0])
unc_rad = np.log(np.shape(unc_aspect)[0])

#==============================================================================
# Plot
#%%============================================================================
# Aspect plot
ax = plt.axes(polar=True)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
theta = np.deg2rad(np.array([aspect_mean, sig_aspect_mean, unc_aspect_mean]))
radii = np.array([nosig_rad, sig_rad, unc_rad])
width = np.deg2rad(np.array([aspect_std, sig_aspect_std, unc_aspect_std]))
bars = plt.bar(theta, radii, width=width, bottom=0.0, alpha=0.5, 
               color=['black', 'DodgerBlue', 'DarkOrange'])
plt.legend(bars, ('domain', 'significant', 'certain'), 
           bbox_to_anchor = (.70,.30), loc=1 )
ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
ax.set_yticklabels([])
plt.show()

# Aspect plot 2
ax = plt.axes(polar=True)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
x = np.linspace(0,2*np.pi,100)
theta_dom = np.deg2rad(dom_aspect)
theta_sig = np.deg2rad(sig_aspect)
theta_unc = np.deg2rad(unc_aspect)
dens_dom = stats.kde.gaussian_kde(theta_dom)
dens_sig = stats.kde.gaussian_kde(theta_sig)
dens_unc = stats.kde.gaussian_kde(theta_unc)
plt.plot(x, dens_dom(x), color='black')
plt.fill_between(x, dens_dom(x), color='black', alpha=0.5)
plt.plot(x, dens_sig(x), color='DodgerBlue')
plt.fill_between(x, dens_sig(x), color='DodgerBlue', alpha=0.5)
plt.plot(x, dens_unc(x), color='DarkOrange')
plt.fill_between(x, dens_unc(x), color='DarkOrange', alpha=0.5)
ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
ax.set_yticklabels([])
plt.show()

# Aspect plot 3
ax = plt.axes(polar=True)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
x = np.linspace(0,2*np.pi,100)
theta_nosig = np.deg2rad(dom_aspect)
theta_sig = np.deg2rad(sig_aspect)
theta_unc = np.deg2rad(unc_aspect)
hist_nosig, bins_nosig = np.histogram(theta_nosig, bins=x, density=True)
hist_sig, bins_sig = np.histogram(theta_sig, bins=x, density=True)
hist_unc, bins_unc = np.histogram(theta_unc, bins=x, density=True)
width = 0.7 * (bins_nosig[1] - bins_nosig[0])
center = (bins_nosig[:-1] + bins_nosig[1:]) / 2
plt.bar(center, hist_nosig, align='center', width=width, color='black', alpha=0.5)
plt.bar(center, hist_sig, align='center', width=width, color='DodgerBlue', alpha=0.5)
plt.bar(center, hist_unc, align='center', width=width, color='DarkOrange', alpha=0.5)
plt.legend(bars, ('domain', 'significant', 'certain'), 
           bbox_to_anchor = (.70,.30), loc=1 )
ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
ax.set_yticklabels([])
plt.show()

# Aspect plot 4
#ax = plt.axes()
#ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
#ax.set_yticklabels([])
x = np.linspace(0,360,100)
dens_dom = stats.kde.gaussian_kde(dom_aspect)
dens_sig = stats.kde.gaussian_kde(sig_aspect)
dens_unc = stats.kde.gaussian_kde(unc_aspect)
m, = plt.plot(x, dens_dom(x), color='black')#, label='domain')
plt.fill_between(x, dens_dom(x), color='black', alpha=0.5)
n, = plt.plot(x, dens_sig(x), color='DodgerBlue')#, label='significant')
plt.fill_between(x, dens_sig(x), color='DodgerBlue', alpha=0.5)
o, = plt.plot(x, dens_unc(x), color='DarkOrange')#, label='certain')
plt.fill_between(x, dens_unc(x), color='DarkOrange', alpha=0.5)
plt.xlim(0,360)
plt.xlabel('Aspect (degrees)')
plt.ylabel('Frequency')
l = plt.legend((m, n, o),('domain', 'significant', 'certain'), loc=0 )
for lobj in l.legendHandles:
    lobj.set_linewidth(4.0)
plt.show()

#%% Elevation plot
plt.axes(polar=False)
p1 = plt.hist((np.hstack((sig_elev, dom_elev)), sig_elev, unc_elev), 
              normed=True, bins=30, histtype='bar', alpha=1, 
              label=('domain', 'significant', 'certain'),
              color=['grey', 'DodgerBlue', 'DarkOrange'])
plt.xlim((400, 2500))
plt.ylim((0, 0.0028))
plt.xlabel("Elevation (m)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#%% Elevation plot 2
y = np.linspace(400, 2500, 1000)
dens_dom = stats.kde.gaussian_kde(dom_elev)
dens_sig = stats.kde.gaussian_kde(sig_elev)
dens_unc = stats.kde.gaussian_kde(unc_elev)
q, = plt.plot(y, dens_dom(y), color='black')#, label='domain')
plt.fill_between(y, dens_dom(y), color='black', alpha=0.5)
r, = plt.plot(y, dens_sig(y), color='DodgerBlue')#, label='significant')
plt.fill_between(y, dens_sig(y), color='DodgerBlue', alpha=0.5)
s, = plt.plot(y, dens_unc(y), color='DarkOrange')#, label='certain')
plt.fill_between(y, dens_unc(y), color='DarkOrange', alpha=0.5)
plt.xlim((400, 2500))
plt.ylim((0, 0.002))
plt.xlabel("Elevation (m)")
plt.ylabel("Frequency")
l = plt.legend((q, r, s),('domain', 'significant', 'certain'), loc=0 )
for lobj in l.legendHandles:
    lobj.set_linewidth(4.0)
plt.show()




