"""
Portable feature extraction backend that works without PyTorch.
Uses simple image features (color histograms, edge features) and text features (TF-IDF).
Falls back to this when PyTorch/CLIP is unavailable.
"""
from typing import List, Optional, Tuple
import hashlib

import cv2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import torch/open_clip, but gracefully fall back
TORCH_AVAILABLE = False
try:
    import torch
    import open_clip
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    pass


def get_device(device_str: str = "auto") -> str:
    """Return device string. If torch unavailable, always return 'cpu'."""
    if not TORCH_AVAILABLE:
        return "cpu"
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


class SimpleImageEncoder:
    """Simple image encoder using color histograms and edge features."""
    
    def __init__(self, embed_dim: int = 512):
        self.embed_dim = embed_dim
    
    def encode_image(self, img: np.ndarray) -> np.ndarray:
        """Encode single image to feature vector."""
        features = []
        
        # Color histogram (RGB channels)
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [64], [0, 256])
            features.append(hist.flatten())
        
        # Grayscale for edge features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Edge histogram using Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_hist = cv2.calcHist([edge_mag.astype(np.float32)], [0], None, [64], [0, 256])
        features.append(edge_hist.flatten())
        
        # Resize features to target dimension
        full_feat = np.concatenate(features)
        
        # Project to embed_dim using deterministic hash-based projection
        if len(full_feat) != self.embed_dim:
            # Use random projection (seeded by feature hash for consistency)
            seed = int(hashlib.md5(b"projection_matrix").hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            proj_matrix = rng.standard_normal((len(full_feat), self.embed_dim)).astype(np.float32)
            proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
            full_feat = full_feat @ proj_matrix
        
        # Normalize
        norm = np.linalg.norm(full_feat)
        if norm > 1e-6:
            full_feat = full_feat / norm
        
        return full_feat.astype(np.float32)


class SimpleTextEncoder:
    """Simple text encoder using TF-IDF and word embeddings."""
    
    def __init__(self, embed_dim: int = 512):
        self.embed_dim = embed_dim
        self.vectorizer = TfidfVectorizer(max_features=embed_dim, ngram_range=(1, 2))
        self.fitted = False
        self._projection = None
    
    def fit(self, corpus: List[str]) -> None:
        """Fit the vectorizer on a corpus."""
        if corpus:
            self.vectorizer.fit(corpus)
            self.fitted = True
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode single text to feature vector."""
        if not self.fitted:
            # Create a simple bag-of-words style encoding
            words = text.lower().split()
            feat = np.zeros(self.embed_dim, dtype=np.float32)
            for i, word in enumerate(words):
                # Hash word to get index
                idx = int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % self.embed_dim
                feat[idx] += 1.0
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm
            return feat
        
        # Use TF-IDF
        tfidf = self.vectorizer.transform([text]).toarray()[0]
        
        # Pad or truncate to embed_dim
        if len(tfidf) < self.embed_dim:
            feat = np.zeros(self.embed_dim, dtype=np.float32)
            feat[:len(tfidf)] = tfidf
        else:
            feat = tfidf[:self.embed_dim].astype(np.float32)
        
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        
        return feat
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts."""
        return np.stack([self.encode_text(t) for t in texts])


class ClipBackbone:
    """
    Feature extraction backbone.
    Uses OpenCLIP when available, falls back to simple features otherwise.
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu") -> None:
        self.device = device
        self.use_clip = False
        self.embed_dim = 512
        
        if TORCH_AVAILABLE:
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device=device
                )
                self.tokenizer = open_clip.get_tokenizer(model_name)
                self.model.eval()
                self.use_clip = True
                self.embed_dim = self.model.visual.output_dim
                print(f"[INFO] Using OpenCLIP model: {model_name}")
            except Exception as e:
                print(f"[WARN] Failed to load CLIP: {e}. Using simple features.")
                self._init_simple()
        else:
            print("[INFO] PyTorch unavailable. Using simple feature extractors.")
            self._init_simple()
    
    def _init_simple(self):
        """Initialize simple feature extractors."""
        self.image_encoder = SimpleImageEncoder(self.embed_dim)
        self.text_encoder = SimpleTextEncoder(self.embed_dim)
        self.use_clip = False
    
    def fit_text_encoder(self, corpus: List[str]) -> None:
        """Fit the text encoder on a corpus (only for simple mode)."""
        if not self.use_clip:
            self.text_encoder.fit(corpus)
    
    def encode_images(self, images) -> np.ndarray:
        """
        Encode images to feature vectors.
        images: either torch.Tensor (for CLIP) or list of np.ndarray (for simple)
        """
        if self.use_clip:
            with torch.no_grad():
                if not isinstance(images, torch.Tensor):
                    # Convert list of numpy arrays
                    images = torch.stack([self.preprocess(Image.fromarray(img)) for img in images])
                images = images.to(self.device, dtype=torch.float32)
                feats = self.model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return feats.cpu().numpy()
        else:
            # Simple encoding
            if isinstance(images, list):
                return np.stack([self.image_encoder.encode_image(img) for img in images])
            else:
                return self.image_encoder.encode_image(images)
    
    def encode_image_from_file(self, path: str, target_size: int = 224) -> np.ndarray:
        """Load and encode image from file path."""
        img = cv2.imread(path)
        if img is None:
            return np.zeros(self.embed_dim, dtype=np.float32)
        img = cv2.resize(img, (target_size, target_size))
        
        if self.use_clip:
            from PIL import Image
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
                feats = self.model.encode_image(img_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return feats.cpu().numpy()[0]
        else:
            return self.image_encoder.encode_image(img)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to feature vectors."""
        if self.use_clip:
            with torch.no_grad():
                tokens = self.tokenizer(texts).to(self.device)
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return feats.cpu().numpy()
        else:
            return self.text_encoder.encode_texts(texts)
    
    def get_preprocess(self):
        if self.use_clip:
            return self.preprocess
        return None
    
    def get_embed_dim(self) -> int:
        return self.embed_dim
    
    def get_text_embed_dim(self) -> int:
        return self.embed_dim
    
    def get_image_embed_dim(self) -> int:
        return self.embed_dim
