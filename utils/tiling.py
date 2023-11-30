import torch
from torchvision import transforms

def tile_image(image, tile_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(tile_size),
        transforms.ToTensor()
    ])

    height, width = image.shape[1], image.shape[2]
    print(height, width)
    num_tiles_h = height // tile_size
    num_tiles_w = width // tile_size

    tiles = []
    count = 0
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            tile = image[:, i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size]
            tile = transform(tile)
            tiles.append(tile)
            count += 1
    print(count)
    return torch.stack(tiles)

# Example usage:
# Assuming 'your_large_image' is a PyTorch tensor of shape (channels, height, width)
large_image = torch.rand((3, 1024, 1024))  # Replace this with your actual large image tensor
tile_size = 256  # Replace this with the desired tile size

tiles = tile_image(large_image, tile_size)
print("Shape of tiles:", tiles.shape, len(tiles))
print("Shape of one tile : ", tiles[0].shape)
