import config
import requests
import json
import numpy as np

""" 
     - The purpose of this is to use PyMesh to get a very accurate estimate of a truss volume fraction.
     - The values calculated are used to validate other methods for calculating volume fraction.
"""

def call_pymesh():

    sidenum = 3
    num_vars = config.sidenum_nvar_map[sidenum]
    radius = 1.0
    sidelen = 0.5
    NC = []
    CA = np.array([
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
        [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
        [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9],
        [4, 5], [4, 6], [4, 7], [4, 8], [4, 9],
        [5, 6], [5, 7], [5, 8], [5, 9],
        [6, 7], [6, 8], [6, 9],
        [7, 8], [7, 9],
        [8, 9]
    ])  # Connectivity array

    print(NC)
    print(CA)

    # Define the URL of the Flask server
    url = 'http://localhost:8483/compute_geometry'

    # Define the data you want to send in the POST request
    data = {
        "CA": CA.tolist(),
        "NC": NC.tolist(),
        "radius": radius,
    }

    # Convert the data to JSON format
    json_data = json.dumps(data)

    # Send the POST request
    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json_data)

    # Print the response from the server
    print(response.status_code)
    print(response.json())

















if __name__ == "__main__":
    call_pymesh()



