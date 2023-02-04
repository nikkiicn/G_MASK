import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torch


def random_noise(nc, width, height):
    '''Generator a random noise image from tensor.

    If nc is 1, the Grayscale image will be created.
    If nc is 3, the RGB image will be generated.

    Args:
        nc (int): (1 or 3) number of channels.
        width (int): width of output image.
        height (int): height of output image.
    Returns:
        PIL Image.
    '''
    img = torch.rand(nc, width, height)
    img = ToPILImage()(img)
    return img


if __name__ == '__main__':
    random_noise(3, 64, 64).save('random-noise.jpg')
    plt.show()
