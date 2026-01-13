"""
----- Design Representations -----

 - Bit List: [0, 1, 1, 1, 0, 0, ..., 1]
 - Bit Str: '011100...1'
 - List of node index pairs: [
          [0, 1],
          [0, 2]
    ]
 - List of node coordinates: [
        [ [0, 1, 1], [2, 1, 1] ] ,
        [ [0, 1, 1], [3, 1. 1] ]
    ]


----- Problem Data Structure -----

 - Nodes must always be ordered by x-coordinate, then y-coordinate
    problem = {
        'nodes': [  # Node coordinate system is in meters
            (0, 0, 0), (0, 1, 1),
            (1, 0, 0), (1, 1, 0),
            (2, 0, 0), (2, 1, 0),
            (3, 0, 0), (3, 1, 0),
        ],
        'nodes_dof': [  # Encodes which degrees of freedom are fixed for each node
            (0, 0, 0), (1, 1, 1),
            (0, 0, 0), (1, 1, 1),
            (0, 0, 0), (1, 1, 1),
            (0, 0, 0), (1, 1, 1),
        ],
        'load_conds': [ # Encodes the loads applied to each node in each direction (newtons)
            [  # Multiple load conditions can be specified
                (0, 0, 0), (0, 0, 0),
                (0, 0, 0), (0, 0, 0),
                (0, 0, 0), (0, 0, 0),
                (0, 0, 0), (1, 1, 1),
            ]
        ],
        'member_radii': 0.1,         # Radii is in meters
        'youngs_modulus': 1.8162e6,  # Material youngs modulus is in pascales (N/m^2)
}
"""