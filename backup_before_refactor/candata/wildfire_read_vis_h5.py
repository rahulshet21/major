#%%
import xarray as xr
from pathlib import Path
from dask.distributed import Lock
import os, glob
import json
from easydict import EasyDict as edict
import numpy as np

np.random.seed(42)
        
# import tables
import matplotlib.pyplot as plt
# from datatree import Tree, Node

def get_vis_percentile(da, p=5):
  vmin, vmax = np.percentile(np.array(da.data), [p, 100-p])
  return vmin, vmax

def remove_label_and_ticks(ax):
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlabel(None)
  ax.set_ylabel(None)
  ax.grid(False)



url = "D:/wildfire-s1s2-dataset-all/wildfire-s1s2-alos-dataset-canada-uint16.h5"

if False:
  import datatree as dt
  tree = dt.open_datatree(url, engine='h5netcdf', chunks=-1)
  print(tree)

#%%
event = 'CA_2017_BC_1157' # CA_2019_QC_808
year = event.split("_")[1]

ds = xr.open_dataset(url, engine='h5netcdf', chunks=-1, group=f'{year}/{event}')
ds


#%%

""" Water Mask """
landCover = ds.aux.sel(aux_band=['landCover'])
landCover = landCover.astype(np.int16)
waterMask = (1 - ((landCover==80) | (landCover == 200))).astype(np.int8)
# waterMask.sel(aux_band='landCover').plot()

# convert DataArray into pure np.array
waterMask = np.array(waterMask.data)

# apply water masking on Sentinel-1 (s1) and ALOS PALSAR-2 (alos)
s1 = ds.s1 * waterMask
alos = ds.alos * waterMask

vmin_s1, vmax_s1 = get_vis_percentile(s1.sel(time='post'))
vmin_alos, vmax_alos = get_vis_percentile(alos.sel(time='post'))

_, H, W = waterMask.shape
ratio = (H / W)
#%%
""" Visualize Dataset and Labels (S1 and ALOS applied water masking) """
ncols = 3 
fig_W = 3 * ncols
fig_H = 9 * ratio + 1

fig, axs = plt.subplots(nrows=4, ncols=ncols, figsize=(fig_W, fig_H))
axs = axs.flatten()

ds.s2.sel(s2_band=['B12','B8','B4'], time='pre').plot.imshow(vmin=0, vmax=4000, ax=axs[0])
s1.sel(s1_band=['ND','VH','VV'], time='pre').plot.imshow(vmin=vmin_s1, vmax=vmax_s1, ax=axs[1])
alos.sel(alos_band=['ND','HV','HH'], time='pre').plot.imshow(vmin=vmin_alos, vmax=vmax_alos, ax=axs[2])

ds.s2.sel(s2_band=['B12','B8','B4'], time='post').plot.imshow(vmin=0, vmax=4000, ax=axs[3])
s1.sel(s1_band=['ND','VH','VV'], time='post').plot.imshow(vmin=vmin_s1, vmax=vmax_s1, ax=axs[4])
alos.sel(alos_band=['ND','HV','HH'], time='post').plot.imshow(vmin=vmin_alos, vmax=vmax_alos, ax=axs[5])

ds.mask.sel(mask_band=['poly']).plot(ax=axs[6], add_colorbar=False)
ds.mask.sel(mask_band=['modis']).plot(ax=axs[7], add_colorbar=False, vmin=150, vmax=300)
ds.mask.sel(mask_band=['firecci']).plot(ax=axs[8], add_colorbar=False, vmin=150, vmax=300)

ds.aux.sel(aux_band=['landCover']).plot(ax=axs[9], add_colorbar=False)
vmin, vmax = get_vis_percentile(ds.aux.sel(aux_band=['elevation']))
ds.aux.sel(aux_band=['elevation']).plot(ax=axs[10], add_colorbar=False, vmin=vmin, vmax=vmax)
vmin, vmax = get_vis_percentile(ds.aux.sel(aux_band=['slope']))
ds.aux.sel(aux_band=['slope']).plot(ax=axs[11], add_colorbar=False, vmin=vmin, vmax=vmax)


titles = [
          's2: pre', 's1: pre', 'alos: pre', 
          's2: post', 's1: post', 'alos: post', 
          'mask: NBAC', 'mask: modis', 'mask: firecci', 
          'aux: landCover', 'aux: elvevation', 'aux: slope', 
          ]
for idx, ax in enumerate(axs):
  ax.set_title(titles[idx])
  remove_label_and_ticks(ax)

plt.tight_layout()
# fig.savefig(f"figures/{event}_data.png")
# plt.show()


#%% diff

""" Visualize Temporal Difference """
ncols = 4 
fig_W = 3 * ncols
fig_H = 9 * ((ratio * 10) // 10 + 0.5)

fig, axs = plt.subplots(nrows=3, ncols=ncols, figsize=(fig_W, fig_H))
axs = axs.flatten()
titles = []

s2_pre = ds.s2.sel(s2_band=['B12','B8','B4'], time='pre').astype(np.int16)
s2_post = ds.s2.sel(s2_band=['B12','B8','B4'], time='post').astype(np.int16)
s2_diff = s2_pre - s2_post

vmin, vmax = get_vis_percentile(s2_diff)

s2_diff.sel(s2_band=['B8','B12','B4']).plot.imshow(vmin=vmin, vmax=vmax, ax=axs[0])
for idx, band in enumerate(['B12','B8','B4'], start=1):
  da = s2_diff.sel(s2_band=band)
  vmin, vmax = get_vis_percentile(da)
  da.plot(vmin=vmin, vmax=vmax, ax=axs[idx], add_colorbar=False, cmap='gray')
titles.append(['s2_diff', 'B12_diff', 'B8_diff', 'B4_diff'])
  
s1_pre = s1.sel(s1_band=['ND','VH','VV'], time='pre').astype(np.int16)
s1_post = s1.sel(s1_band=['ND','VH','VV'], time='post').astype(np.int16)
s1_diff = s1_pre - s1_post

vmin, vmax = get_vis_percentile(s1_diff)
s1_diff.sel(s1_band=['VH','VV','VH']).plot.imshow(vmin=vmin, vmax=vmax, ax=axs[4])
for idx, band in enumerate(['ND','VH','VV'], start=5):
  da = s1_diff.sel(s1_band=band)
  vmin, vmax = get_vis_percentile(da)
  da.plot(vmin=vmin, vmax=vmax, ax=axs[idx], add_colorbar=False, cmap='gray')
titles.append(['s1_diff', 'ND_diff', 'VH_diff', 'VV_diff'])

alos_pre = alos.sel(alos_band=['ND','HV','HH'], time='pre').astype(np.int16)
alos_post = alos.sel(alos_band=['ND','HV','HH'], time='post').astype(np.int16)
alos_diff = alos_pre - alos_post

vmin, vmax = get_vis_percentile(alos_diff)
alos_diff.sel(alos_band=['HV','HH', 'HV']).plot.imshow(vmin=vmin, vmax=vmax, ax=axs[8])
for idx, band in enumerate(['ND','HV', 'HH'], start=9):
  da = alos_diff.sel(alos_band=band)
  vmin, vmax = get_vis_percentile(da)
  da.plot(vmin=vmin, vmax=vmax, ax=axs[idx], add_colorbar=False, cmap='gray')
titles.append(['alos_diff', 'ND_diff', 'HV_diff', 'HH_diff'])

# titles = [
#           's2_diff', 'B12_diff', 'B8_diff', 'B4_diff', 
#           's1_diff', 'ND_diff', 'VH_diff', 'VV_diff', 
#           'alos_diff', 'ND_diff', 'HV_diff', 'HH_diff'
#           ]

titles = np.array(titles).flatten()
for idx, ax in enumerate(axs):
  ax.set_title(titles[idx])
  remove_label_and_ticks(ax)
plt.tight_layout()
fig.savefig(f"figures/{event}_diff.png")
# plt.show()
