import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

curv_stats = pd.read_pickle('curv_stats.pkl')

colors = {('wL3', True) : 'black', ('2hAPF', True) : (52/255, 73/255, 147/255), ('4hAPF', True) : (100/255, 134/255, 189/255), ('6hAPF', True) : (169/255, 202/255, 221/255),
          ('4hAPF', False) : (189/255, 134/255, 100/255), ('6hAPF', False) : (221/255, 202/255, 169/255)}
linestyles = {True: '-', False: '-.'}


objects_to_plot = {('wL3', True) : "wL3", 
                ('4hAPF', True) : "4hAPF",
                ('4hAPF', False) : "4hAPF mutant",
                ('6hAPF', True) : "6hAPF",
                ('6hAPF', False) : "6hAPF mutant"}
plot_error = True
fig, axs = plt.subplots(figsize=(12,6), ncols=2)
for idx, ((stage, wildtype), label) in enumerate(objects_to_plot.items()):
    subset = curv_stats[(curv_stats['stage'] == stage) & (curv_stats['wildtype'] == wildtype)]
    axs[0].plot(subset['arclength'].values[0], subset['k_a_dvb'].values[0], label=label, color=colors[(stage, wildtype)], linestyle=linestyles[wildtype], zorder=idx)
    axs[1].plot(subset['arclength'].values[0], subset['k_dvb'].values[0], label=label, color=colors[(stage, wildtype)], linestyle=linestyles[wildtype], zorder=idx)
    if plot_error:
        axs[0].fill_between(subset['arclength'].values[0], subset['k_a_dvb_q5'].values[0], subset['k_a_dvb_q95'].values[0], color=colors[(stage, wildtype)], alpha=0.7, zorder=idx)
        axs[1].fill_between(subset['arclength'].values[0], subset['k_dvb_q5'].values[0], subset['k_dvb_q95'].values[0], color=colors[(stage, wildtype)], alpha=0.7, zorder=idx)

min_x = min(axs[0].get_xlim()[0], axs[1].get_xlim()[0])
max_x = max(axs[0].get_xlim()[1], axs[1].get_xlim()[1])
min_y = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
max_y = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
axs[0].set_xlim(min_x, max_x)
axs[0].set_ylim(min_y, max_y)
axs[1].set_xlim(min_x, max_x)
axs[1].set_ylim(min_y, max_y)

axs[0].set_xlabel('Arclength a-DVB')
axs[0].set_ylabel('Curvature a-DVB')

axs[1].set_xlabel('Arclength DVB')
axs[1].set_ylabel('Curvature DVB')


axs[0].set_title("a-DVB curvature across development")
axs[1].set_title("DVB curvature across development")

axs[0].legend()
axs[1].legend()

plt.show()