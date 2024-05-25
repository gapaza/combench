import os
import json
import matplotlib.pyplot as plt
import config

def plot_hypervolume_progress(directory):
    # Create a dictionary to hold data from all json files
    data = {}

    # Iterate over files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                # Load the json data
                file_data = json.load(f)
                # Store the data using the filename (without extension) as the key
                data[os.path.splitext(filename)[0]] = file_data

    # Plotting the data
    plt.figure(figsize=(10, 6))

    for key, values in data.items():
        nfe = [pair[0] for pair in values]
        hv = [pair[1] for pair in values]
        plt.plot(nfe, hv, label=key)

    plt.xlabel('Number of Function Evaluations (NFE)')
    plt.ylabel('Hypervolume (HV)')
    plt.title('Hypervolume Progress Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    directory = os.path.join(config.root_dir, 'combench', 'ga', 'validation')
    plot_hypervolume_progress(directory)