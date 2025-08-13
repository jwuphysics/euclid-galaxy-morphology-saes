import os
import pandas as pd
import torch
import timm
import glob
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm

class ImageActivationExtractor:
    def __init__(
        self, 
        model_name: str,
        layer_name: str,
        batch_size: int = 32,
        image_size: int = 224,
        device: Optional[str] = None
    ):
        """
        Initialize the activation extractor with a pretrained model.
        
        Args:
            model_name: Name of the pretrained timm model to use
            layer_name: Name of the layer to extract activations from
            batch_size: Batch size for processing images
            image_size: Size to resize images to
            device: Device to run the model on (cuda or cpu)
        """
        self.model_name = model_name
        self.layer_name = layer_name
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the pretrained model
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Set up image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        # Register a hook to extract activations
        self.activations = {}
        self.hook_handles = []
        self._register_hook(layer_name)
        
    def _register_hook(self, layer_name: str):
        """Register a forward hook on the specified layer."""
        def hook_fn(module, input, output):
            self.activations[layer_name] = output.detach().cpu()
            
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                self.hook_handles.append(handle)
                print(f"Registered hook for layer: {layer_name}")
                return
                
        raise ValueError(f"Layer {layer_name} not found in model. Available layers: {[name for name, _ in self.model.named_modules()]}")
    
    def __del__(self):
        """Remove hooks when the object is deleted."""
        for handle in self.hook_handles:
            handle.remove()
    
    def extract_from_dataframe(
        self, 
        df: pd.DataFrame, 
        image_path_column: str,
        id_column: Optional[str] = None
    ) -> Tuple[np.ndarray, List]:
        """
        Extract activations for images in a DataFrame.
        
        Args:
            df: DataFrame containing image paths
            image_path_column: Name of column containing image paths
            id_column: Optional column to use as identifiers for the images
            
        Returns:
            Tuple of (activations array, list of image IDs)
        """
        class ImageDataset(Dataset):
            def __init__(self, df, image_path_column, id_column, transform):
                self.df = df
                self.image_paths = df[image_path_column].tolist()
                self.ids = df[id_column].tolist() if id_column else list(range(len(df)))
                self.transform = transform
                
            def __len__(self):
                return len(self.df)


            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                img_id = self.ids[idx]
                
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        img_tensor = self.transform(img)
                    return img_tensor, img_id
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    # Return a zero tensor with the correct shape if image loading fails
                    return torch.zeros(3, 224, 224), img_id
            
        dataset = ImageDataset(df, image_path_column, id_column, self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
        
        all_activations = []
        all_ids = []
        
        with torch.no_grad():
            for batch_imgs, batch_ids in tqdm(dataloader, desc="Extracting activations"):
                batch_imgs = batch_imgs.to(self.device)
                
                # Forward pass through the model to trigger hook
                self.model(batch_imgs)
                
                # Store activations and IDs
                all_activations.append(self.activations[self.layer_name].numpy())
                all_ids.extend(batch_ids)
                
        return np.vstack(all_activations), all_ids
        
    def extract_from_directory(
        self, 
        root_dir: str, 
        pattern: str = "*.jpg"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Recursively extract activations from all images in a directory.
        
        Args:
            root_dir: Root directory to scan for images
            pattern: File pattern to match (default: "*.jpg")
            
        Returns:
            Tuple of (activations array, list of image paths)
        """
        import glob
        
        # Find all image files recursively
        image_paths = []
        for dirpath, _, _ in os.walk(root_dir):
            paths = glob.glob(os.path.join(dirpath, pattern))
            image_paths.extend(paths)
            
        print(f"Found {len(image_paths)} images matching pattern {pattern}")
        
        # Create a DataFrame
        df = pd.DataFrame({"image_path": image_paths})
        
        # Extract features
        return self.extract_from_dataframe(df, "image_path")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    extractor = ImageActivationExtractor(
        model_name="hf_hub:mwalmsley/zoobot-encoder-euclid", 
        layer_name="head",   
        batch_size=64,
        image_size=224
    )


    image_paths = []
    for dirpath, _, _ in os.walk("./data/q1/cutouts_jpg/"):
        paths = glob.glob(os.path.join(dirpath, "*.jpg"))
        image_paths.extend(paths)
    np.save("../results/euclid_q1_image_paths.npy", image_paths)
    
    activations, _ = extractor.extract_from_directory(
        root_dir="../data/q1/cutouts_jpg/",  
        pattern="*.jpg"       
    )

    
    # Save the activations to a file
    np.save("../results/euclid_q1_embeddings.npy", activations)
    
    print(f"Extracted activations with shape: {activations.shape}") # (380111, 640)