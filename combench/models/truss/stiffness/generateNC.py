import numpy as np


def generateNC(sel, sidenum):
    # Generate a vector of equally spaced notch values
    notchvec = np.linspace(0, 1, sidenum)

    # Initialize an empty list to store nodal coordinates
    NC = []

    # Nested loops to generate all combinations of notch values
    for i in range(sidenum):
        for j in range(sidenum):
            # Append the current combination to the NC list
            NC.append([notchvec[i], notchvec[j]])

    # Convert NC list to a NumPy array for easier manipulation
    NC = np.array(NC)

    # Scale the nodal coordinates by the unit cell size
    NC = sel * NC

    return NC



if __name__ == '__main__':
    sel = 250e-6  # Unit cell size
    sidenum = 3  # Number of nodes along one side
    NC = generateNC(sel, sidenum)
    print(NC)




