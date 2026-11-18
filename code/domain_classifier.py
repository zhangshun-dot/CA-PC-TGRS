# -*- coding: utf-8 -*-
"""CLIP Zero-Shot Domain Classifier for CD-TDM.

Classifies images into domains (remote sensing, natural scene, etc.)
using CLIP zero-shot text-image similarity. Zero training required.

Domains supported:
    - remote_sensing: satellite/aerial/overhead imagery
    - natural_scene: ground-level everyday photos
    - infrared: thermal infrared imagery
    - sar: synthetic aperture radar
    - underwater: underwater imagery
    - medical: medical imaging (X-ray, MRI, etc.)

Usage:
    from domain_classifier import DomainClassifier
    dc = DomainClassifier()
    domain = dc.classify(image)  # -> 'remote_sensing'
"""

import torch
import numpy as np
from PIL import Image

# Domain definitions: label -> CLIP text prompts (multiple prompts per domain for robustness)
DOMAIN_PROMPTS = {
    'remote_sensing': [
        'a remote sensing image',
        'a satellite image of terrain',
        'an aerial view photograph',
        'an overhead image from above',
    ],
    'natural_scene': [
        'a natural scene photograph',
        'a ground-level photo',
        'an everyday scene image',
        'a picture taken from eye level',
    ],
    'infrared': [
        'a thermal infrared image',
        'a heat signature photograph',
        'an infrared thermal camera view',
    ],
    'sar': [
        'a synthetic aperture radar image',
        'a SAR radar image with speckle noise',
        'a grainy radar scan of terrain',
    ],
    'underwater': [
        'an underwater photograph',
        'a deep sea image',
        'a submarine scene with blue tint',
    ],
    'medical': [
        'a medical scan image',
        'an X-ray or MRI medical image',
        'a clinical diagnostic image',
    ],
}

# Domain labels ordered for classifier output
DOMAIN_LABELS = list(DOMAIN_PROMPTS.keys())


class DomainClassifier:
    """Zero-shot domain classifier using CLIP ViT-B/32."""

    def __init__(self, clip_model_path=None, device=None):
        """
        Args:
            clip_model_path: path to CLIP ViT-B/32 weights.
                             Defaults to local weights directory.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load CLIP
        import clip
        self._clip = clip
        if clip_model_path is None:
            clip_model_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'

        self.clip_model, self.preprocess = clip.load(
            clip_model_path, device=self.device
        )
        self.clip_model.eval()

        # Pre-compute text embeddings for all domain prompts
        self._build_text_embeddings()

    def _build_text_embeddings(self):
        """Pre-compute CLIP text embeddings for all domain prompts."""
        self.domain_labels = []
        self.domain_indices = []  # maps text_embedding[i] -> domain_label index
        all_texts = []

        for domain_idx, label in enumerate(DOMAIN_LABELS):
            prompts = DOMAIN_PROMPTS[label]
            for prompt in prompts:
                all_texts.append(prompt)
                self.domain_indices.append(domain_idx)
            self.domain_labels.append(label)

        # Tokenize and encode all texts at once
        text_tokens = self._clip.tokenize(all_texts, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
        self.n_domains = len(DOMAIN_LABELS)

    def classify(self, image, return_scores=False):
        """Classify a single PIL Image.

        Args:
            image: PIL.Image in RGB mode.
            return_scores: if True, also return per-domain confidence scores.

        Returns:
            domain_label (str) or (domain_label, scores_dict) if return_scores.
        """
        # Preprocess and encode image
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_features = self.clip_model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # Cosine similarity with all text embeddings
        similarities = (img_features @ self.text_features.T).squeeze(0)

        # Aggregate per domain: max similarity across prompts
        domain_scores = torch.zeros(self.n_domains, device=self.device)
        for i, domain_idx in enumerate(self.domain_indices):
            domain_scores[domain_idx] = max(
                domain_scores[domain_idx].item(),
                similarities[i].item()
            )

        # Softmax for interpretability
        domain_probs = torch.softmax(domain_scores * 10, dim=0)  # temperature=0.1

        best_idx = domain_scores.argmax().item()
        best_label = self.domain_labels[best_idx]

        if return_scores:
            scores = {
                label: {
                    'score': float(domain_scores[i]),
                    'prob': float(domain_probs[i]),
                }
                for i, label in enumerate(self.domain_labels)
            }
            return best_label, scores
        return best_label

    def classify_batch(self, images, return_scores=False):
        """Classify a list of PIL Images.

        Returns:
            list of domain_label (str) or list of (domain_label, scores_dict).
        """
        results = [self.classify(img, return_scores=return_scores) for img in images]
        if return_scores:
            labels = [r[0] for r in results]
            scores = [r[1] for r in results]
            return labels, scores
        return results

    def get_domain_info(self, domain_label):
        """Get metadata for a domain."""
        info = {
            'remote_sensing': {
                'description': 'Satellite/aerial/overhead remote sensing imagery',
                'default_prefix': 'an overhead image of',
                'alt_prefixes': ['a satellite image of', 'a remote sensing image of',
                                'an aerial view of'],
                'typical_resolution': '0.5-30m GSD',
                'viewing_angle': 'nadir/oblique',
            },
            'natural_scene': {
                'description': 'Ground-level everyday photographs',
                'default_prefix': 'a photo of',
                'alt_prefixes': ['an image of', 'a picture of'],
                'typical_resolution': 'varies',
                'viewing_angle': 'eye-level',
            },
            'infrared': {
                'description': 'Thermal infrared imagery',
                'default_prefix': 'a thermal image of',
                'alt_prefixes': ['an infrared image of'],
                'typical_resolution': 'varies',
                'viewing_angle': 'varies',
            },
            'sar': {
                'description': 'Synthetic Aperture Radar imagery',
                'default_prefix': 'a SAR image of',
                'alt_prefixes': ['a radar image of'],
                'typical_resolution': '1-30m GSD',
                'viewing_angle': 'side-looking',
            },
            'underwater': {
                'description': 'Underwater/subsea imagery',
                'default_prefix': 'an underwater image of',
                'alt_prefixes': ['a submarine photo of'],
                'typical_resolution': 'varies',
                'viewing_angle': 'varies',
            },
            'medical': {
                'description': 'Medical diagnostic imaging',
                'default_prefix': 'a medical image of',
                'alt_prefixes': ['a clinical scan of', 'a diagnostic image of'],
                'typical_resolution': 'varies',
                'viewing_angle': 'varies',
            },
        }
        return info.get(domain_label, {})


# ── Quick test ──
if __name__ == '__main__':
    import sys
    print("Loading CLIP Domain Classifier...")
    dc = DomainClassifier()

    # Test with sample images if provided
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            img = Image.open(path).convert('RGB')
            label, scores = dc.classify(img, return_scores=True)
            print(f"\n{path}")
            print(f"  -> {label}")
            for domain, s in sorted(scores.items(), key=lambda x: -x[1]['score']):
                print(f"     {domain:20s} score={s['score']:.4f} prob={s['prob']:.4f}")
    else:
        # Quick sanity check: classify some DIOR and COCO images
        import os, glob
        # DIOR sample
        dior_dir = r'C:\Users\张三\modeling\dior_data\full20_images'
        dior_imgs = []
        for cls_dir in sorted(os.listdir(dior_dir))[:2]:
            imgs = sorted(glob.glob(os.path.join(dior_dir, cls_dir, '*.jpg')))[:2]
            dior_imgs.extend(imgs)

        # COCO sample
        coco_dir = r'C:\Users\张三\modeling\coco_data\val2017'
        coco_imgs = sorted(glob.glob(os.path.join(coco_dir, '*.jpg')))[:2]

        print("\n" + "="*60)
        print("DIOR (remote sensing) images:")
        for p in dior_imgs:
            img = Image.open(p).convert('RGB')
            label, scores = dc.classify(img, return_scores=True)
            print(f"  {os.path.basename(p):40s} -> {label} "
                  f"(rs={scores['remote_sensing']['prob']:.3f}, "
                  f"natural={scores['natural_scene']['prob']:.3f})")

        print("\nCOCO (natural scene) images:")
        for p in coco_imgs:
            img = Image.open(p).convert('RGB')
            label, scores = dc.classify(img, return_scores=True)
            print(f"  {os.path.basename(p):40s} -> {label} "
                  f"(rs={scores['remote_sensing']['prob']:.3f}, "
                  f"natural={scores['natural_scene']['prob']:.3f})")

        print("\nDone.")
