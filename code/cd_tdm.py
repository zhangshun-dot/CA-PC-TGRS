# -*- coding: utf-8 -*-
"""CD-TDM: Cross-Domain Token Distribution Modulation.

Full pipeline:
    Image → DomainClassifier → DomainRegistry → TDM prompt → Detection

Key components:
    1. DomainClassifier: CLIP zero-shot domain recognition
    2. DomainRegistry: per-domain OFAT measurement lookup
    3. TDM prompt generator: constructs domain-optimal prompts
    4. CD-TDM detector: end-to-end detection with cross-domain prompts

Usage:
    from cd_tdm import CDTDM
    cd_tdm = CDTDM()
    prompt = cd_tdm.build_prompt(image, 'airplane')
    # Or full detection:
    results = cd_tdm.detect(image, class_names=['airplane', 'airport', ...])
"""

import json
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from domain_classifier import DomainClassifier
from domain_registry import DomainRegistry, AXIS_MODIFIERS, AXIS_ORDER


@dataclass
class TDMPrompt:
    """TDM-generated prompt with metadata."""
    prompt: str
    domain: str
    prefix: str
    class_name: str
    modifiers: List[str]
    n_modifiers: int
    deltas: Dict[str, float]


@dataclass
class DetectionResult:
    """Single-image detection result."""
    image_path: str
    domain: str
    class_name: str
    prompt: str
    boxes: List[List[float]]  # [x1, y1, x2, y2]
    scores: List[float]
    labels: List[str]


class CDTDM:
    """Cross-Domain Token Distribution Modulation pipeline."""

    def __init__(self, device=None):
        """
        Args:
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"CD-TDM initializing on {self.device}...")

        # Load components
        print("  Loading DomainClassifier...")
        self.classifier = DomainClassifier(device=self.device)

        print("  Loading DomainRegistry...")
        self.registry = DomainRegistry()

        # Auto-import existing OFAT data if available
        self._auto_register_remote_sensing()

        print(f"  Ready. Registered domains: {self.registry.registered_domains}")

    def _auto_register_remote_sensing(self):
        """Import DIOR OFAT data if available."""
        ofat_path = (r'C:\Users\张三\Desktop\论文版本历史\初稿4_论文文件'
                     r'\实验数据\ofat_overhead_results.json')
        import os
        if os.path.exists(ofat_path) and not self.registry.has('remote_sensing'):
            self.registry.import_from_ofat_json(
                'remote_sensing', ofat_path,
                prefix='an overhead image of', threshold=0.03,
                metadata={
                    'description': 'DIOR 20-class remote sensing',
                    'dataset': 'DIOR',
                    'encoder': 'BERT (GDINO-tiny)',
                }
            )

    # ── Domain Recognition ──

    def recognize_domain(self, image):
        """Recognize domain of an image.

        Args:
            image: PIL.Image

        Returns:
            str: domain label (e.g., 'remote_sensing', 'natural_scene')
        """
        return self.classifier.classify(image)

    def recognize_domain_with_scores(self, image):
        """Recognize domain with confidence scores."""
        return self.classifier.classify(image, return_scores=True)

    # ── Prompt Generation ──

    def build_prompt(self, image_or_domain, class_name):
        """Build CD-TDM prompt for an image and class.

        Args:
            image_or_domain: PIL.Image (auto-detect domain) or str (known domain)
            class_name: object class name

        Returns:
            TDMPrompt object
        """
        # Step 1: Determine domain
        if isinstance(image_or_domain, str):
            domain = image_or_domain
        else:
            domain = self.classifier.classify(image_or_domain)

        # Step 2: Get TDM config from registry
        cfg = self.registry.get_tdm_config(domain, class_name)

        if cfg is None:
            # Domain or class not registered — fallback to simple prompt
            fallback_prefix = self._get_fallback_prefix(domain)
            return TDMPrompt(
                prompt=f'{fallback_prefix} {class_name}',
                domain=domain,
                prefix=fallback_prefix,
                class_name=class_name,
                modifiers=[],
                n_modifiers=0,
                deltas={},
            )

        return TDMPrompt(
            prompt=cfg['prompt'],
            domain=cfg['domain'],
            prefix=cfg['prefix'],
            class_name=cfg['class'],
            modifiers=cfg['modifiers'],
            n_modifiers=cfg['n_modifiers'],
            deltas=cfg['deltas'],
        )

    def build_prompts_batch(self, image_or_domain, class_names):
        """Build CD-TDM prompts for multiple classes.

        Args:
            image_or_domain: PIL.Image or str (domain label)
            class_names: list of class name strings

        Returns:
            list of TDMPrompt objects
        """
        # Determine domain once
        if isinstance(image_or_domain, str):
            domain = image_or_domain
        else:
            domain = self.classifier.classify(image_or_domain)

        return [self.build_prompt(domain, cls) for cls in class_names]

    def _get_fallback_prefix(self, domain):
        """Get a reasonable fallback prefix for unregistered domains."""
        fallbacks = {
            'remote_sensing': 'an overhead image of',
            'natural_scene': 'a photo of',
            'infrared': 'a thermal image of',
            'sar': 'a SAR image of',
            'underwater': 'an underwater image of',
            'medical': 'a medical image of',
        }
        return fallbacks.get(domain, 'an image of')

    # ── Detection ──

    def detect(self, image, class_names, model_name='gdino-tiny',
               conf_threshold=0.25, iou_threshold=0.25):
        """Run CD-TDM detection on an image.

        Args:
            image: PIL.Image
            class_names: list of class name strings
            model_name: 'gdino-tiny' | 'gdino-base' | 'yolo-world' | 'owlv2'
            conf_threshold: confidence threshold for detections
            iou_threshold: IoU threshold for NMS (if applicable)

        Returns:
            list of DetectionResult
        """
        # Step 1: Domain recognition
        domain, domain_scores = self.classifier.classify(image, return_scores=True)

        # Step 2: Build TDM prompts for all classes
        tdm_prompts = self.build_prompts_batch(domain, class_names)

        # Step 3: Run detection
        if 'gdino' in model_name:
            results = self._detect_gdino(
                image, tdm_prompts, model_name, conf_threshold
            )
        elif model_name == 'yolo-world':
            results = self._detect_yolo(
                image, tdm_prompts, conf_threshold, iou_threshold
            )
        elif model_name == 'owlv2':
            results = self._detect_owlv2(
                image, tdm_prompts, conf_threshold
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Attach domain info
        for r in results:
            r.domain = domain

        return results

    def _detect_gdino(self, image, tdm_prompts, model_name, conf_threshold):
        """Run GDINO detection with CD-TDM prompts."""
        if not hasattr(self, '_gdino_model'):
            self._load_gdino(model_name)

        results = []
        for tdm in tdm_prompts:
            inputs = self._gdino_processor(
                images=image, text=tdm.prompt, return_tensors='pt'
            )
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor)
                      else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._gdino_model(**inputs)

            target = torch.tensor([image.size[::-1]]).to(self.device)
            r = self._gdino_processor.post_process_grounded_object_detection(
                outputs, threshold=conf_threshold, target_sizes=target
            )[0]

            boxes = r['boxes'].cpu().tolist()
            scores = r['scores'].cpu().tolist()
            labels = [tdm.class_name] * len(boxes)

            results.append(DetectionResult(
                image_path='',
                domain='',
                class_name=tdm.class_name,
                prompt=tdm.prompt,
                boxes=boxes,
                scores=scores,
                labels=labels,
            ))

        return results

    def _load_gdino(self, model_name):
        """Lazy-load GDINO model."""
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import os
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

        if model_name == 'gdino-tiny':
            cache = (r'C:\Users\张三\.cache\huggingface\hub'
                     r'\models--IDEA-Research--grounding-dino-tiny'
                     r'\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71')
        else:  # gdino-base
            cache = (r'C:\Users\张三\.cache\huggingface\hub'
                     r'\models--IDEA-Research--grounding-dino-base\snapshots')
            snap = os.listdir(cache)[0]
            cache = os.path.join(cache, snap)

        self._gdino_processor = AutoProcessor.from_pretrained(cache)
        self._gdino_model = (AutoModelForZeroShotObjectDetection
                            .from_pretrained(cache, dtype=torch.float32)
                            .eval().to(self.device))

    def _detect_yolo(self, image, tdm_prompts, conf_threshold, iou_threshold):
        """Run YOLO-World detection with CD-TDM prompts."""
        if not hasattr(self, '_yolo_model'):
            from ultralytics import YOLOWorld
            self._yolo_model = YOLOWorld('yolov8s-worldv2.pt')
            self._yolo_model.to(self.device)

        results = []
        for tdm in tdm_prompts:
            self._yolo_model.set_classes([tdm.prompt])
            r = self._yolo_model.predict(
                image, verbose=False, device=self.device,
                conf=conf_threshold, iou=iou_threshold
            )
            boxes = r[0].boxes.xyxy.cpu().tolist() if r[0].boxes is not None else []
            scores = (r[0].boxes.conf.cpu().tolist()
                      if r[0].boxes is not None and r[0].boxes.conf is not None
                      else [])
            labels = [tdm.class_name] * len(boxes)

            results.append(DetectionResult(
                image_path='',
                domain='',
                class_name=tdm.class_name,
                prompt=tdm.prompt,
                boxes=boxes,
                scores=scores,
                labels=labels,
            ))

        return results

    def _detect_owlv2(self, image, tdm_prompts, conf_threshold):
        """Run OWLv2 detection with CD-TDM prompts."""
        if not hasattr(self, '_owlv2_model'):
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            import os
            cache = (r'C:\Users\张三\.cache\huggingface\hub'
                     r'\models--google--owlv2-base-patch16-ensemble\snapshots')
            snap = os.listdir(cache)[0]
            cache = os.path.join(cache, snap)
            self._owlv2_processor = Owlv2Processor.from_pretrained(cache)
            self._owlv2_model = (Owlv2ForObjectDetection
                                .from_pretrained(cache, dtype=torch.float32)
                                .eval().to(self.device))

        results = []
        for tdm in tdm_prompts:
            # Truncate long prompts for OWLv2 (CLIP token limit)
            prompt = tdm.prompt
            if len(prompt) > 60:
                prompt = f'{tdm.prefix} {tdm.class_name}'

            inputs = self._owlv2_processor(
                text=prompt, images=image, return_tensors='pt'
            )
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor)
                      else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._owlv2_model(**inputs)

            target = torch.tensor([image.size[::-1]]).to(self.device)
            r = self._owlv2_processor.post_process_object_detection(
                outputs, threshold=conf_threshold, target_sizes=target
            )[0]

            boxes = r['boxes'].cpu().tolist()
            scores = r['scores'].cpu().tolist()
            labels = [tdm.class_name] * len(boxes)

            results.append(DetectionResult(
                image_path='',
                domain='',
                class_name=tdm.class_name,
                prompt=tdm.prompt,
                boxes=boxes,
                scores=scores,
                labels=labels,
            ))

        return results

    # ── Cross-Domain Evaluation ──

    def evaluate_domain_accuracy(self, images, true_domains):
        """Evaluate domain classifier accuracy.

        Args:
            images: list of PIL.Image
            true_domains: list of str (ground-truth domain labels)

        Returns:
            dict with accuracy, per-domain accuracy, confusion matrix
        """
        pred_domains = [self.classifier.classify(img) for img in images]

        correct = sum(1 for p, t in zip(pred_domains, true_domains) if p == t)
        accuracy = correct / len(images)

        # Per-domain accuracy
        from collections import defaultdict
        per_domain = defaultdict(lambda: {'correct': 0, 'total': 0})
        for p, t in zip(pred_domains, true_domains):
            per_domain[t]['total'] += 1
            if p == t:
                per_domain[t]['correct'] += 1

        domain_acc = {
            d: v['correct'] / v['total'] if v['total'] > 0 else 0
            for d, v in per_domain.items()
        }

        # Confusion matrix
        all_domains = sorted(set(true_domains + pred_domains))
        confusion = {}
        for t in all_domains:
            confusion[t] = {}
            for p in all_domains:
                confusion[t][p] = sum(
                    1 for pt, tt in zip(pred_domains, true_domains)
                    if pt == p and tt == t
                )

        return {
            'accuracy': accuracy,
            'n_images': len(images),
            'per_domain_accuracy': domain_acc,
            'confusion_matrix': confusion,
        }

    def register_new_domain(self, domain_label, detect_fn, class_names,
                             sample_images, prefix_candidates=None,
                             threshold=0.03):
        """Register a new domain by running mini-OFAT scan.

        This is the core "domain registration" operation: given a few
        sample images and class names, run a quick OFAT measurement
        (5 prefixes × 6 axes ≈ 30 forward passes per class) and store
        the results in the registry.

        Args:
            domain_label: str, domain identifier
            detect_fn: callable(image, prompt) -> detection_count (int or float)
            class_names: list of class names in this domain
            sample_images: dict {class_name: [PIL.Image, ...]}
            prefix_candidates: list of prefix strings to test.
                Default: auto-selected from DOMAIN_PREFIX_CANDIDATES.
            threshold: delta screening threshold

        Returns:
            Path to saved domain config file.
        """
        from domain_registry import DOMAIN_PREFIX_CANDIDATES, AXIS_MODIFIERS, AXIS_ORDER

        if prefix_candidates is None:
            prefix_candidates = DOMAIN_PREFIX_CANDIDATES.get(
                domain_label,
                ['a photo of', 'an image of', 'a picture of']
            )

        ofat_classes = {}

        for cls_name in class_names:
            imgs = sample_images.get(cls_name, [])
            if not imgs:
                print(f"  Warning: no sample images for '{cls_name}', skipping")
                continue

            # Step 1: Find best prefix
            best_prefix = None
            best_score = -1
            prefix_scores = {}

            for prefix in prefix_candidates:
                prompt = f'{prefix} {cls_name}'
                scores = [detect_fn(img, prompt) for img in imgs]
                mean_score = np.mean(scores) if scores else 0
                prefix_scores[prefix] = mean_score
                if mean_score > best_score:
                    best_score = mean_score
                    best_prefix = prefix

            base_dr = best_score

            # Step 2: Measure per-axis deltas
            deltas = {'grammar': 0.0}  # grammar is the prefix itself
            for axis in AXIS_ORDER[1:]:  # skip grammar
                modifier = AXIS_MODIFIERS[axis]
                prompt = f'{best_prefix} {cls_name}, {modifier}'
                scores = [detect_fn(img, prompt) for img in imgs]
                mean_score = np.mean(scores) if scores else 0
                deltas[axis] = mean_score - base_dr

            ofat_classes[cls_name] = {
                'base': float(base_dr),
                'deltas': deltas,
                'prefix_scores': prefix_scores,
            }

            print(f"  {cls_name}: prefix={best_prefix}, base={base_dr:.3f}, "
                  f"pos_axes={sum(1 for ax,v in deltas.items() if v>threshold)}")

        # Compute mean deltas
        mean_deltas = {}
        for ax in AXIS_ORDER[1:]:
            vals = [c['deltas'].get(ax, 0) for c in ofat_classes.values()]
            mean_deltas[ax] = float(np.mean(vals)) if vals else 0.0

        # Register
        ofat_data = {
            'prefix': best_prefix or prefix_candidates[0],
            'threshold': threshold,
            'classes': ofat_classes,
            'mean_deltas': mean_deltas,
        }

        path = self.registry.register(
            domain_label, ofat_data,
            metadata={
                'description': f'Auto-registered: {domain_label}',
                'n_classes': len(ofat_classes),
                'prefix_candidates': prefix_candidates,
                'best_prefix': best_prefix,
            }
        )

        print(f"  Registered {len(ofat_classes)} classes to '{domain_label}'")
        return path


# ── Quick demo ──
if __name__ == '__main__':
    print("=" * 60)
    print("CD-TDM Pipeline Demo")
    print("=" * 60)

    cd_tdm = CDTDM()

    # Demo 1: Build prompts for different domains
    print("\n--- Demo 1: Prompt generation ---")
    domains = ['remote_sensing', 'natural_scene']
    classes = ['airplane', 'car']

    for domain in domains:
        for cls in classes:
            p = cd_tdm.build_prompt(domain, cls)
            print(f"  [{domain}] {cls}: \"{p.prompt}\" "
                  f"(modifiers: {p.modifiers})")

    # Demo 2: Test with actual images if available
    import os, glob
    dior_dir = r'C:\Users\张三\modeling\dior_data\full20_images'
    coco_dir = r'C:\Users\张三\modeling\coco_data\val2017'

    if os.path.exists(dior_dir) and os.path.exists(coco_dir):
        print("\n--- Demo 2: Domain recognition on real images ---")

        # DIOR image
        dior_imgs = []
        for cls_dir in sorted(os.listdir(dior_dir))[:1]:
            imgs = sorted(glob.glob(os.path.join(dior_dir, cls_dir, '*.jpg')))[:1]
            dior_imgs.extend(imgs)

        # COCO image
        coco_imgs = sorted(glob.glob(os.path.join(coco_dir, '*.jpg')))[:1]

        for path in dior_imgs + coco_imgs:
            img = Image.open(path).convert('RGB')
            domain, scores = cd_tdm.recognize_domain_with_scores(img)
            name = os.path.basename(path)
            print(f"  {name:40s} -> {domain}")
            for d, s in sorted(scores.items(), key=lambda x: -x[1]['prob']):
                bar = '█' * int(s['prob'] * 20)
                print(f"    {d:20s} {bar} {s['prob']:.3f}")

    print("\nDone.")
