import numpy as np

# Load your 256x256x256 data (You can replace this with your data)
# data = np.load("0060.dat.npy")

# Check if the shape is compatible for splitting
# if data.shape != (256, 256, 96):
#     raise ValueError("Data shape must be 256x256x256")

# Initialize a list to store the four 128x128x128 pieces
pieces = []
recombined_data = np.zeros((256, 256, 96))

# Split the data into four consecutive 128x128x128 chunks and save them
# for x in range(2):
#     for y in range(2):
#             piece = data[x*128:(x+1)*128, y*128:(y+1)*128, :]
#             pieces.append(piece)
#             piece_filename = f"piece_{x}{y}.npy"
#             np.save(piece_filename, piece)
#             print(f"Saved {piece_filename}")
for x in range(2):
    for y in range(2):
        # for z in range(2):
        piece_filename = f"C:/Users/20649/Seismic/seis_diffusion_SR/seis_diffusion_SR/dataset/big_pieces_recombined/test4_combin/piece_{x}{y}.npy"
        piece = np.load(piece_filename)
        recombined_data[x*128:(x+1)*128, y*128:(y+1)*128, :] = piece

# Save the recombined data as a .npy file
np.save("recombined_data.npy", recombined_data)

