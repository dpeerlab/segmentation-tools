from typing import Any
import numpy as np

class Mesh:
    
    def __init__(self, x, y, pad=24):
        """
        This class represents a mesh. Class helps reusing the same mesh for multiple images.
        """
        self.coordinate = [x, y]
        self.pad = pad
        
    def __repr__(self) -> str:
        return f"Mesh(x={self.coordinate[0]}, y={self.coordinate[1]}, pad={self.pad})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def _clip_arange(self, img, axis):
        """
        Select indices only >= 0 and <= image-dimension.
        """
        coordinate_max = img.shape[axis]
        indices = np.arange(
            max(self.coordinate[axis] - self.pad, 0),
            min(self.coordinate[axis] + self.pad + 1, coordinate_max)
        )
        return indices
    
    
    def get_mesh(self, img):
        """
        Calculate mesh and return part of image.
        """

        # get indices (notice that x and y are switched!)
        indices_x = self._clip_arange(img, axis=1)
        indices_y = self._clip_arange(img, axis=0)
        
        # get coordinates
        mesh_x, mesh_y = np.meshgrid(indices_x, indices_y)
                
        # get image mesh
        img_mesh = img[mesh_x.flatten(), mesh_y.flatten()]
        
        # reshape
        rows = max(indices_x) - min(indices_x) + 1
        img_mesh = img_mesh.reshape([-1, rows]).T
        
        return img_mesh
