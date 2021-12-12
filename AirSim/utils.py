
import airsim
import numpy as np


# name : str
def capture_as_np(client, name: str = None):

    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
        airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float = False, compress = False)], vehicle_name=name)  #scene vision image in uncompressed RGB array
    
    img1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
    
    
    img_rgb = img1d.reshape(responses[1].height, responses[1].width, 3) # reshape array to 4 channel image array H X W X 3
    
    
    return img_rgb


