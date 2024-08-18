import config
import os
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    m1 = 'EncoderPINN-F4'
    # m1 = 'BaselinePINN2'


    m2 = 'EncoderPINN-F3'
    # m2 = 'BaselineMLP3'


    comp = 'BaselineVsPinn'



    h1 = os.path.join(config.database_dir, m1 + '.json')
    h2 = os.path.join(config.database_dir, m2 + '.json')

    j1 = json.load(open(h1))
    j2 = json.load(open(h2))


    plt.plot(j1['loss'])
    plt.plot(j1['val_loss'])

    plt.plot(j2['loss'])
    plt.plot(j2['val_loss'])


    plt.title('model loss comparison: ' + comp)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # Set y axis range
    plt.ylim([0, 0.3])
    # plt.legend(['train', 'validation'], loc='upper right')
    plt.legend([m1 + ' train', m1 + ' validation', m2 + ' train', m2 + ' validation'], loc='upper right')

    plot_save_path = os.path.join(config.plots_dir, 'pinn', comp + '.png')
    plt.savefig(plot_save_path)















