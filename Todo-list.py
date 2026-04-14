

# Find Ground truth
    # Total point score for pic's
    # Amount of crowns on board
    # Features for tile types
    # 


# Split data into train / test (If enough data Validation)

# Test Split
"6,9,12,19,22,24,28,29,34,39,42,52,56,68,72"

# Extract / make features
    # Crown data for HOG
        # implement Crown detection

    # HSV median / mean for Tiles
        # Make a CSV file of Features for tiles


# Find Tiles using KNN or SVM
    # Using Median / Mean for each tile
        # Add to a 5x5 grid list


# Check every tiles edge for crowns
    # Split the edges of tiles into smaller cut outs and find crowns using HOG (folding)

# two 5x5 grid lists?
    # tile type list
    # Crown amount

# Add tile type and crown amount to a grid list  
    # loop though list for connecting tiles and * the total connected tile piece with crowns
        # Plus every score together for final game score


