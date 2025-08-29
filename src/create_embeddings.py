import torch
import timm
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from datasets import load_dataset

# Import shared utilities
from .utils import set_seed, configure_torch_reproducibility

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
    
    def extract_from_hf_dataset(
        self, 
        dataset_name: str = "mwalmsley/gz_euclid",
        split: str = "train"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract activations for images in a Huggingface dataset.
        
        Args:
            dataset_name: Name of the Huggingface dataset
            split: Dataset split to use ("train" or "test")
            
        Returns:
            Tuple of (activations array, list of image IDs)
        """
        print(f"Loading Huggingface dataset: {dataset_name}, split: {split}")
        dataset = load_dataset(dataset_name, split=split)
        
        class HFImageDataset(Dataset):
            def __init__(self, hf_dataset, transform):
                self.hf_dataset = hf_dataset
                self.transform = transform
                
            def __len__(self):
                return len(self.hf_dataset)

            def __getitem__(self, idx):
                item = self.hf_dataset[idx]
                img = item['image']
                img_id = item['id_str']
                
                try:
                    # Convert PIL image to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_tensor = self.transform(img)
                    return img_tensor, img_id
                except Exception as e:
                    print(f"Error processing image {img_id}: {e}")
                    # Return a zero tensor with the correct shape if image processing fails
                    return torch.zeros(3, 224, 224), img_id
            
        dataset_wrapper = HFImageDataset(dataset, self.transform)
        dataloader = DataLoader(dataset_wrapper, batch_size=self.batch_size, num_workers=4)
        
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

if __name__ == "__main__":
    # Initialize reproducible random state
    set_seed(42)
    configure_torch_reproducibility()
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    extractor = ImageActivationExtractor(
        model_name="hf_hub:mwalmsley/zoobot-encoder-euclid", 
        layer_name="head",   
        batch_size=64,
        image_size=224
    )

    # Extract embeddings from train split
    print("Extracting embeddings from train split...")
    train_activations, train_ids = extractor.extract_from_hf_dataset(
        dataset_name="mwalmsley/gz_euclid",
        split="train"
    )
    
    # Extract embeddings from test split  
    print("Extracting embeddings from test split...")
    test_activations, test_ids = extractor.extract_from_hf_dataset(
        dataset_name="mwalmsley/gz_euclid", 
        split="test"
    )
    
    # Save the embeddings and IDs
    np.save("./results/euclid_train_embeddings.npy", train_activations)
    np.save("./results/euclid_train_ids.npy", train_ids)
    np.save("./results/euclid_test_embeddings.npy", test_activations)
    np.save("./results/euclid_test_ids.npy", test_ids)
    
    print(f"Train embeddings shape: {train_activations.shape}")
    print(f"Test embeddings shape: {test_activations.shape}")
    print(f"Train IDs: {len(train_ids)} samples")
    print(f"Test IDs: {len(test_ids)} samples")