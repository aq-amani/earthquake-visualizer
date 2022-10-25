import matplotlib.pyplot as plt
# Specify a Japanese font to correctly display text
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
# Remove pygame welcome message
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import matplotlib.colors as mpl_colors
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation

import numpy as np
import pandas as pd
import datetime
import pygame as pg
import re

pg.mixer.init()
pg.init()

pg.mixer.set_num_channels(50)


def dms2dd_min(degrees, minutes, direction):
    dd = float(degrees) + float(minutes)/60;
    if direction == 'W' or direction == 'S':
        dd *= -1
    return dd;

def parse_dms(dms):
    parts = re.split('[°\']+', dms)
    lat = dms2dd_min(parts[0], parts[1], parts[2])
    return (lat)

## Read and prepare quake data
#csv_file = 'test_data.csv'
csv_file = '20220316.csv'
data = pd.read_csv(f'quake_data/{csv_file}')
quake_data = data.to_records(index=False)

# Cleanup data before converting to float
quake_data['mag'][quake_data['mag'] == '-'] = '0'
quake_data['mag'] = quake_data['mag'].astype(float)

## Thresholding quake data
# Remove quakes that are less than magnitude_data_min_threshold in magnitude
magnitude_data_min_threshold = 0.0
quake_data = np.delete(quake_data, np.where(quake_data['mag'] < magnitude_data_min_threshold), axis=0)
# Remove magnitude zero quakes
quake_data = np.delete(quake_data, np.where(quake_data['mag'] == 0.0), axis=0)
###

## Plot and animation settings
dt = 30 # timesteps per animation frame in seconds
magnitude_multiplier = 1.8 # For plotting large enough circles
magnitude_power = 2.8 # power to use with magnitude values to make circles grow exponentially in size

log_view_threshold = 4.5 # Minimum Magnitude necessary to view logs
alpha_decay_rate = 0.005 # Rate at wich alpha values of scatter colors decrease
enable_alpha_decay = True
log_line_limit = 6
initial_alpha = 0.5
###

## Numpy arrays to hold data neessary for visualization
scat_colors = np.zeros(quake_data.shape, dtype=(float, 4))
positions = np.zeros(quake_data.shape, dtype=(float, 2))
datetimes = np.zeros(quake_data.shape, dtype=object)
# Scale magnitude data to make scatter points large enough to be seen on the map
mag_data_scaled = magnitude_multiplier*(quake_data['mag'] ** magnitude_power)*quake_data['mag']
###

## Figure area settings
fig = plt.figure(figsize=(16, 8), dpi = 200)

ax = plt.axes()
plt.style.use('dark_background')
fig.patch.set_facecolor('black')

basemap = Basemap(projection='lcc', resolution='i',
            lat_0=35.68, lon_0=139.77,
            #width=4E6, height=4E6)
            width=4000000, height=2500000)
basemap.bluemarble()
basemap.drawcoastlines(color='darkgray', linewidth=0.2)
basemap.drawcountries(color='darkgray', linewidth=0.2)

scat = basemap.scatter([], [], latlon=True, lw=0.35)

time_text = ax.text(0.7, 0.93, '', c = 'white', fontweight = 'bold', ha = 'left', va = 'bottom', fontsize=14/2.5, transform=ax.transAxes)
log_text = ax.text(0.58, 0.23, '', c = 'white', fontweight = 'bold', ha = 'left', va = 'top', fontsize=11/2.5, transform=ax.transAxes)
###
## Colormap settings
norm = mpl_colors.Normalize(vmin=0, vmax=9)
# Make a modified version of the gist_ncar colormap to use
orig_cmap = plt.get_cmap('gist_ncar')
colorlist = orig_cmap(np.linspace(0.04, 0.98, orig_cmap.N))
new_color_map = mpl_colors.LinearSegmentedColormap.from_list('cropped_gist_ncar', colorlist[20:-30])
mp = cm.ScalarMappable(norm=norm, cmap=new_color_map)
fig.colorbar(mp, label='Magnitude')
###


for i in range(len(quake_data['year'])):
    h, min = map(int, quake_data['time'][i].split(':'))
    datetimes[i] = datetime.datetime(quake_data['year'][i], quake_data['month'][i], quake_data['day'][i], h, min, int(quake_data['second'][i]), 10**5 * int(str(quake_data['second'][i]).split('.')[1]))

    lat=parse_dms(str(quake_data['lat'][i]))
    lon=parse_dms(str(quake_data['lon'][i]))
    positions[i] = basemap(lon, lat)

    scat_colors[i] = mp.to_rgba(quake_data['mag'][i], alpha=initial_alpha)

edge_colors = scat_colors.copy()
edge_colors[:,:3] = 0 # black edges

current_index = 0
start_index = 0
log_info = ''
log_lines = []
quake_time = datetimes[0]
simulation_time = datetimes[0]
total_time_frames = int((datetimes[-1] - datetimes[0]).total_seconds()/dt)+100

log_header = f'Earthquake log (M{log_view_threshold} and above):\n------------------------------------------------\n'
time_text.set_text(f'Time: {simulation_time}')
log_text.set_text(log_header)

def refresh_graph(frame_number):
    global simulation_time
    global quake_time
    global current_index
    global log_info
    global log_lines
    global start_index
    global edge_colors

    while simulation_time >= quake_time and current_index < len(datetimes):

        sound = pg.mixer.Sound(f"audio/{int(quake_data['mag'][current_index])}.wav")
        sound.set_volume(0.5)
        sound.play()

        if len(log_lines) >= log_line_limit:
            log_lines.pop(0)
        if quake_data['mag'][current_index] >= log_view_threshold:
            log_lines.append(f"{quake_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-5]} M{quake_data['mag'][current_index]}@ {quake_data['depth'][current_index]}Km {(quake_data['city'][current_index]).strip()}\n")
        log_info = ''
        for line in log_lines:
            log_info += line
        # Limit range of the points to plot to those that are still visible
        # for better performance
        if len(np.where(scat_colors[:,3] == 0)[0]) > 0 and enable_alpha_decay:
            start_index = max(np.where(scat_colors[:,3] == 0)[0])

        scat.set_edgecolors(edge_colors[start_index:current_index])

        scat.set_facecolors(scat_colors[start_index:current_index])
        scat.set_sizes(list(mag_data_scaled[start_index:current_index]))
        scat.set_offsets(positions[start_index:current_index])

        current_index += 1
        if current_index >= len(datetimes):
            break
        quake_time = datetimes[current_index]
    #Alpha decay
    if enable_alpha_decay:
        scat_colors[start_index:current_index,3] = np.where(scat_colors[start_index:current_index,3] >= alpha_decay_rate, scat_colors[start_index:current_index,3] - alpha_decay_rate, 0)
        edge_colors[start_index:current_index,3] = np.where(edge_colors[start_index:current_index,3] >= alpha_decay_rate, edge_colors[start_index:current_index,3] - alpha_decay_rate, 0)

    scat.set_edgecolors(edge_colors[start_index:current_index])
    scat.set_facecolors(scat_colors[start_index:current_index])
    scat.set_sizes(list(mag_data_scaled[start_index:current_index]))
    scat.set_offsets(positions[start_index:current_index])
    #
    time_text.set_text(f"  Time: {simulation_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-5]}")
    log_text.set_text(log_header + log_info)
    simulation_time += datetime.timedelta(seconds=dt)
    #plt.savefig(f'quake_frames/{frame_number}.png', dpi = 300)
    return scat,time_text,log_text



def init_plot():
    #print("initializing plot")
    #pass
    return scat,time_text,log_text


animation = FuncAnimation(fig, refresh_graph, init_func = init_plot, interval=1, frames=total_time_frames, repeat = False, blit=True)
plt.show()
