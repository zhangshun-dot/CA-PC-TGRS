# -*- coding: utf-8 -*-
"""Domain Registry for CD-TDM.

Stores per-domain OFAT measurement data and provides:
    - Domain lookup: given a domain label, return TDM prompt config
    - Domain registration: quick mini-OFAT scan for new domains
    - Persistence: JSON-based, one file per domain

Directory structure:
    domain_registry/
        registry.json          # domain label -> config file mapping
        remote_sensing.json    # DIOR OFAT data
        natural_scene.json     # COCO OFAT data
        ...

Usage:
    from domain_registry import DomainRegistry
    dr = DomainRegistry()
    config = dr.get('remote_sensing')  # -> {prefix, deltas, threshold}
    dr.register('natural_scene', ofat_data)  # register new domain
"""

import json
import os
from pathlib import Path

# ── Default registry directory ──
REGISTRY_DIR = Path(r'C:\Users\张三\Desktop\domain_registry')

# Axis modifier mapping (consistent with existing TDM code)
AXIS_MODIFIERS = {
    'grammar': None,  # special: domain prefix, not a modifier
    'granularity': 'with fine details',
    'spatial': 'in its natural setting',
    'geometric': 'with distinct structural features',
    'scale': 'large and prominent',
    'spectral': 'with natural coloring',
    'texture': 'with clearly visible surface features',
}

AXIS_ORDER = ['grammar', 'granularity', 'spatial', 'geometric',
              'scale', 'spectral', 'texture']

# ── Domain prefix candidates for registration scan ──
DOMAIN_PREFIX_CANDIDATES = {
    'remote_sensing': [
        'a satellite image of',
        'an overhead image of',
        'an aerial view of',
        'a remote sensing image of',
    ],
    'natural_scene': [
        'a photo of',
        'an image of',
        'a picture of',
        'a natural scene of',
    ],
    'infrared': [
        'a thermal image of',
        'an infrared image of',
        'a heat camera image of',
    ],
    'sar': [
        'a SAR image of',
        'a radar image of',
        'a synthetic aperture radar image of',
    ],
    'underwater': [
        'an underwater image of',
        'a submarine photo of',
        'a deep sea image of',
    ],
    'medical': [
        'a medical image of',
        'a clinical scan of',
        'a diagnostic image of',
        'an X-ray image of',
    ],
}


class DomainRegistry:
    """Manages per-domain OFAT configurations for CD-TDM."""

    def __init__(self, registry_dir=None):
        self.registry_dir = Path(registry_dir or REGISTRY_DIR)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / 'registry.json'
        self._load()

    def _load(self):
        """Load registry index from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = {'domains': {}, 'version': '1.0'}

    def _save(self):
        """Save registry index to disk."""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    # ── Public API ──

    @property
    def registered_domains(self):
        """List of registered domain labels."""
        return list(self.index['domains'].keys())

    def has(self, domain_label):
        """Check if a domain is registered."""
        return domain_label in self.index['domains']

    def get(self, domain_label):
        """Get OFAT config for a domain.

        Returns:
            dict with keys: domain, prefix, deltas, threshold, classes, metadata
            or None if domain not registered.
        """
        if not self.has(domain_label):
            return None

        config_file = self.registry_dir / self.index['domains'][domain_label]['file']
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def register(self, domain_label, ofat_data, metadata=None):
        """Register (or update) a domain with OFAT measurement data.

        Args:
            domain_label: str, domain identifier (e.g., 'remote_sensing')
            ofat_data: dict with OFAT measurements.
                Format: {
                    'prefix': 'an overhead image of',  # optimal prefix
                    'threshold': 0.03,                  # delta screening threshold
                    'classes': {                        # per-class OFAT
                        'airplane': {
                            'base': 0.8,
                            'deltas': {'grammar': 0.0, 'granularity': 0.0, ...}
                        },
                        ...
                    },
                    'mean_deltas': {                    # across-class mean deltas
                        'granularity': 0.01, ...
                    },
                }
            metadata: optional dict with domain info.
        """
        config_file = f'{domain_label}.json'
        config = {
            'domain': domain_label,
            'prefix': ofat_data.get('prefix', ''),
            'threshold': ofat_data.get('threshold', 0.03),
            'classes': ofat_data.get('classes', {}),
            'mean_deltas': ofat_data.get('mean_deltas', {}),
            'axis_modifiers': AXIS_MODIFIERS,
            'metadata': metadata or {},
        }

        # Add default prefix candidates if not present
        if domain_label in DOMAIN_PREFIX_CANDIDATES:
            config['metadata']['prefix_candidates'] = \
                DOMAIN_PREFIX_CANDIDATES[domain_label]

        # Write domain config file
        filepath = self.registry_dir / config_file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Update registry index
        self.index['domains'][domain_label] = {
            'file': config_file,
            'prefix': ofat_data.get('prefix', ''),
            'n_classes': len(ofat_data.get('classes', {})),
            'metadata': metadata or {},
        }
        self._save()

        return filepath

    def import_from_ofat_json(self, domain_label, ofat_json_path,
                               prefix='an overhead image of', threshold=0.03,
                               metadata=None):
        """Import OFAT data from existing JSON file.

        The JSON file should have format like ofat_overhead_results.json:
            {class_name: {base: X, deltas: {axis: delta, ...}}, ...}

        Args:
            domain_label: domain identifier
            ofat_json_path: path to OFAT JSON file
            prefix: optimal domain prefix
            threshold: delta screening threshold
            metadata: optional domain metadata
        """
        with open(ofat_json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # Handle different JSON structures
        if 'ofat_overhead' in raw:
            classes = raw['ofat_overhead']
        elif 'classes' in raw:
            classes = raw['classes']
        else:
            classes = raw

        # Compute mean deltas across classes
        all_deltas = {ax: [] for ax in AXIS_ORDER[1:]}  # exclude grammar
        for cls_name, cls_data in classes.items():
            if not isinstance(cls_data, dict):
                continue
            deltas = cls_data.get('deltas', {})
            for ax in AXIS_ORDER[1:]:
                if ax in deltas:
                    all_deltas[ax].append(deltas[ax])

        mean_deltas = {
            ax: sum(vals) / len(vals) if vals else 0.0
            for ax, vals in all_deltas.items()
        }

        ofat_data = {
            'prefix': prefix,
            'threshold': threshold,
            'classes': classes,
            'mean_deltas': mean_deltas,
        }

        return self.register(domain_label, ofat_data, metadata=metadata)

    def get_tdm_config(self, domain_label, class_name):
        """Get TDM prompt configuration for a specific class in a domain.

        Args:
            domain_label: domain identifier
            class_name: object class name

        Returns:
            dict: {prefix, modifiers, deltas, n_modifiers} or None
        """
        config = self.get(domain_label)
        if config is None:
            return None

        cls_data = config['classes'].get(class_name)
        if cls_data is None:
            # Use mean deltas as fallback
            deltas = config.get('mean_deltas', {})
        else:
            deltas = cls_data.get('deltas', {})

        # Select positive deltas above threshold, excluding grammar
        thresh = config.get('threshold', 0.03)
        positive = sorted(
            [(ax, v) for ax, v in deltas.items()
             if ax != 'grammar' and v > thresh],
            key=lambda x: -x[1]
        )

        # At most 2 modifiers (S-N constraint)
        modifiers = [
            AXIS_MODIFIERS[ax]
            for ax, _ in positive[:2]
        ]

        return {
            'domain': domain_label,
            'prefix': config['prefix'],
            'class': class_name,
            'deltas': deltas,
            'modifiers': modifiers,
            'n_modifiers': len(modifiers),
            'prompt': self._build_prompt(config['prefix'], class_name, modifiers),
        }

    def _build_prompt(self, prefix, class_name, modifiers):
        """Build TDM prompt string."""
        prompt = f'{prefix} {class_name}'
        if modifiers:
            prompt += ', ' + ', '.join(modifiers)
        return prompt

    def get_domain_summary(self, domain_label):
        """Get human-readable domain summary."""
        config = self.get(domain_label)
        if config is None:
            return f"Domain '{domain_label}' not registered."

        lines = [
            f"Domain: {domain_label}",
            f"  Prefix: {config['prefix']}",
            f"  Threshold: {config.get('threshold', 0.03)}",
            f"  Classes: {len(config.get('classes', {}))}",
            f"  Mean axis deltas:",
        ]
        for ax in AXIS_ORDER[1:]:
            v = config.get('mean_deltas', {}).get(ax, 0.0)
            sign = '+' if v > 0 else ' '
            lines.append(f"    {ax:15s} {sign}{v:.4f}")
        return '\n'.join(lines)


# ── Quick init ──
if __name__ == '__main__':
    dr = DomainRegistry()

    # Import existing DIOR OFAT data as 'remote_sensing' domain
    ofat_path = (r'C:\Users\张三\Desktop\论文版本历史\初稿4_论文文件'
                 r'\实验数据\ofat_overhead_results.json')
    if os.path.exists(ofat_path):
        print("Registering remote_sensing domain from existing OFAT data...")
        dr.import_from_ofat_json(
            'remote_sensing',
            ofat_path,
            prefix='an overhead image of',
            threshold=0.03,
            metadata={
                'description': 'DIOR 20-class remote sensing',
                'dataset': 'DIOR',
                'encoder': 'BERT (GDINO-tiny)',
                'n_classes': 20,
                'default_prefix_candidates': [
                    'a satellite image of',
                    'an overhead image of',
                    'an aerial view of',
                    'a remote sensing image of',
                ],
            }
        )
        print(f"  Registered: {dr.registered_domains}")
        print(f"  Summary:\n{dr.get_domain_summary('remote_sensing')}")

        # Test TDM config for specific class
        cfg = dr.get_tdm_config('remote_sensing', 'airplane')
        print(f"\n  TDM config for 'airplane':")
        print(f"    Prefix: {cfg['prefix']}")
        print(f"    Modifiers: {cfg['modifiers']}")
        print(f"    Prompt: {cfg['prompt']}")
    else:
        print(f"OFAT file not found: {ofat_path}")
        print(f"Creating empty registry at: {dr.registry_dir}")

    print("\nDone.")
