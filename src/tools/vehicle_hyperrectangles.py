"""
Vehicle-lang Integration for Human-in-the-Loop Hyperrectangle Generation
NIDS Attack Boundary Definition Tool

This module integrates the research-based hyperrectangle definitions from the original
NIDS paper with Vehicle-lang formal verification and human-in-the-loop refinement.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import subprocess
import tempfile
from src.config import config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Import research hyperrectangles from attacks module
try:
    from src.attacks.pgd import get_research_hyperrectangles, generate_hyperrectangle_batch
except ImportError:
    logger.warning("Could not import research hyperrectangles from attacks module")
    
    def get_research_hyperrectangles():
        return {}
    
    def generate_hyperrectangle_batch(batch_size=2048, pkts_length=10):
        return np.array([]), np.array([])


@dataclass
class FeatureBounds:
    """Represents bounds for a single feature"""
    name: str
    min_value: float
    max_value: float
    description: str
    unit: str = ""
    
    def to_vehicle_constraint(self, var_name: str = "x", index: int = 0) -> str:
        """Convert to Vehicle-lang constraint syntax"""
        return f"{self.min_value} <= {var_name} ! {index} <= {self.max_value}"


@dataclass
class Hyperrectangle:
    """Represents a hyperrectangle attack boundary"""
    name: str
    description: str
    feature_bounds: List[FeatureBounds]
    attack_type: str  # "DoS", "DDoS", "Intrusion", etc.
    confidence: float = 1.0  # Human confidence in this boundary
    
    def to_vehicle_spec(self) -> str:
        """Convert hyperrectangle to Vehicle specification"""
        bounds_constraints = []
        for i, bound in enumerate(self.feature_bounds):
            constraint = bound.to_vehicle_constraint(var_name="x", index=i)
            bounds_constraints.append(f"  {constraint}")
        
        constraints_str = " and\n".join(bounds_constraints)
        
        return f"""
-- Attack boundary: {self.name}
-- Description: {self.description}
-- Attack type: {self.attack_type}
-- Confidence: {self.confidence}

{self.name.lower().replace(' ', '_')} : Vector Rat {len(self.feature_bounds)} -> Bool
{self.name.lower().replace(' ', '_')} x =
{constraints_str}
"""

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is within this hyperrectangle"""
        if len(point) != len(self.feature_bounds):
            return False
        
        for i, bound in enumerate(self.feature_bounds):
            if not (bound.min_value <= point[i] <= bound.max_value):
                return False
        return True
    
    def volume(self) -> float:
        """Calculate the volume of the hyperrectangle"""
        volume = 1.0
        for bound in self.feature_bounds:
            volume *= (bound.max_value - bound.min_value)
        return volume


class VehicleHyperrectangleGenerator:
    """
    Human-in-the-loop hyperrectangle generator using Vehicle-lang
    for formal verification and constraint specification
    """
    
    def __init__(self, data_path: str = None, output_dir: str = "vehicle"):
        self.data_path = data_path or os.path.join(config.data.data_dir, "preprocessed-dos-train.csv")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vehicle-lang files
        self.vehicle_dir = self.output_dir / "specifications"
        self.vehicle_dir.mkdir(exist_ok=True)
        
        self.hyperrectangles: List[Hyperrectangle] = []
        self.feature_names = []
        self.data = None
        
        logger.info(f"Initialized Vehicle hyperrectangle generator")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and analyze the training data"""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            self.data = pd.read_csv(self.data_path)
            self.feature_names = [col for col in self.data.columns if col != 'label']
            
            logger.info(f"Loaded {len(self.data)} samples with {len(self.feature_names)} features")
            logger.info(f"Features: {self.feature_names[:5]}..." if len(self.feature_names) > 5 else f"Features: {self.feature_names}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def analyze_feature_distributions(self) -> Dict[str, Dict[str, float]]:
        """Analyze feature distributions for attack vs benign traffic"""
        if self.data is None:
            self.load_data()
        
        analysis = {}
        
        # Separate attack and benign traffic
        attack_data = self.data[self.data['label'] == 1]  # Assuming 1 = attack
        benign_data = self.data[self.data['label'] == 0]  # Assuming 0 = benign
        
        logger.info(f"Attack samples: {len(attack_data)}, Benign samples: {len(benign_data)}")
        
        for feature in self.feature_names:
            attack_stats = attack_data[feature].describe()
            benign_stats = benign_data[feature].describe()
            
            analysis[feature] = {
                'attack_mean': float(attack_stats['mean']),
                'attack_std': float(attack_stats['std']),
                'attack_min': float(attack_stats['min']),
                'attack_max': float(attack_stats['max']),
                'attack_q25': float(attack_stats['25%']),
                'attack_q75': float(attack_stats['75%']),
                'benign_mean': float(benign_stats['mean']),
                'benign_std': float(benign_stats['std']),
                'benign_min': float(benign_stats['min']),
                'benign_max': float(benign_stats['max']),
                'benign_q25': float(benign_stats['25%']),
                'benign_q75': float(benign_stats['75%']),
                'separation_score': abs(attack_stats['mean'] - benign_stats['mean']) / (attack_stats['std'] + benign_stats['std'] + 1e-8)
            }
        
        # Save analysis
        analysis_file = self.output_dir / "feature_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Feature analysis saved to {analysis_file}")
        return analysis
    
    def load_research_hyperrectangles(self) -> Dict[str, Hyperrectangle]:
        """
        Load the research-based hyperrectangle definitions from the original NIDS paper.
        These represent the core contribution of carefully crafted attack patterns.
        
        Returns:
            Dictionary of research hyperrectangles by attack pattern name
        """
        logger.info("Loading research-based hyperrectangle definitions...")
        
        research_hyperrects = get_research_hyperrectangles()
        loaded_hyperrects = {}
        
        # Feature names for the 42-dimensional feature space (2 + 10*4)
        feature_names = [
            "time_elapsed", "protocol",
            "pkt_dir_0", "pkt_dir_1", "pkt_dir_2", "pkt_dir_3", "pkt_dir_4", 
            "pkt_dir_5", "pkt_dir_6", "pkt_dir_7", "pkt_dir_8", "pkt_dir_9",
            "pkt_flag_0", "pkt_flag_1", "pkt_flag_2", "pkt_flag_3", "pkt_flag_4",
            "pkt_flag_5", "pkt_flag_6", "pkt_flag_7", "pkt_flag_8", "pkt_flag_9",
            "pkt_iat_0", "pkt_iat_1", "pkt_iat_2", "pkt_iat_3", "pkt_iat_4",
            "pkt_iat_5", "pkt_iat_6", "pkt_iat_7", "pkt_iat_8", "pkt_iat_9",
            "pkt_size_0", "pkt_size_1", "pkt_size_2", "pkt_size_3", "pkt_size_4",
            "pkt_size_5", "pkt_size_6", "pkt_size_7", "pkt_size_8", "pkt_size_9"
        ]
        
        for pattern_name, bounds_list in research_hyperrects.items():
            feature_bounds = []
            
            for i, bounds in enumerate(bounds_list):
                if i >= len(feature_names):
                    break
                    
                feature_name = feature_names[i]
                min_val = float(bounds[0]) if isinstance(bounds, (list, tuple)) and len(bounds) > 0 else 0.0
                max_val = float(bounds[1]) if isinstance(bounds, (list, tuple)) and len(bounds) > 1 else 1.0
                
                # Create feature bound with descriptive information
                if "time" in feature_name:
                    description = "Time-based feature (normalized)"
                    unit = "seconds"
                elif "protocol" in feature_name:
                    description = "Protocol type indicator"
                    unit = "categorical"
                elif "pkt_dir" in feature_name:
                    description = "Packet direction indicator"
                    unit = "binary"
                elif "pkt_flag" in feature_name:
                    description = "Packet flag value (normalized)"
                    unit = "normalized"
                elif "pkt_iat" in feature_name:
                    description = "Inter-arrival time (normalized)"
                    unit = "seconds"
                elif "pkt_size" in feature_name:
                    description = "Packet size (normalized)"
                    unit = "bytes"
                else:
                    description = "Network feature"
                    unit = "normalized"
                
                feature_bounds.append(FeatureBounds(
                    name=feature_name,
                    min_value=min_val,
                    max_value=max_val,
                    description=description,
                    unit=unit
                ))
            
            # Determine attack type and description
            if pattern_name in ["goodHTTP1", "goodHTTP2"]:
                attack_type = "Benign"
                description = f"Benign HTTP traffic pattern: {pattern_name}"
            elif pattern_name == "hulk":
                attack_type = "DoS"
                description = "HULK DoS attack pattern with fast request generation"
            elif pattern_name == "slowIATsAttacks":
                attack_type = "DoS"
                description = "Slow rate DoS attack with variable inter-arrival times"
            elif pattern_name == "invalid":
                attack_type = "Generic"
                description = "Generic invalid/attack pattern with wide bounds"
            else:
                attack_type = "Unknown"
                description = f"Research pattern: {pattern_name}"
            
            # Create hyperrectangle object
            hyperrect = Hyperrectangle(
                name=pattern_name,
                description=description,
                feature_bounds=feature_bounds,
                attack_type=attack_type,
                confidence=1.0  # Research-based patterns have high confidence
            )
            
            loaded_hyperrects[pattern_name] = hyperrect
            self.hyperrectangles.append(hyperrect)
        
        logger.info(f"Loaded {len(loaded_hyperrects)} research-based hyperrectangles:")
        for name, hyperrect in loaded_hyperrects.items():
            logger.info(f"  - {name}: {hyperrect.attack_type} ({len(hyperrect.feature_bounds)} features)")
        
        return loaded_hyperrects

    def visualize_feature_distributions(self, top_features: int = 10):
        """Create visualizations to help human understanding"""
        if self.data is None:
            self.load_data()
        
        analysis = self.analyze_feature_distributions()
        
        # Sort features by separation score
        sorted_features = sorted(analysis.keys(), 
                               key=lambda x: analysis[x]['separation_score'], 
                               reverse=True)[:top_features]
        
        # Create subplots
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        axes = axes.flatten()
        
        attack_data = self.data[self.data['label'] == 1]
        benign_data = self.data[self.data['label'] == 0]
        
        for i, feature in enumerate(sorted_features):
            ax = axes[i]
            
            # Plot distributions
            ax.hist(benign_data[feature], bins=50, alpha=0.7, label='Benign', color='blue', density=True)
            ax.hist(attack_data[feature], bins=50, alpha=0.7, label='Attack', color='red', density=True)
            
            ax.set_title(f'{feature}\n(Sep Score: {analysis[feature]["separation_score"]:.3f})')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature distribution plots saved to {self.output_dir}/feature_distributions.png")
        
        return sorted_features
    
    def suggest_hyperrectangle_bounds(self, attack_type: str = "DoS", 
                                    method: str = "percentile", 
                                    percentile: float = 95.0) -> List[FeatureBounds]:
        """Suggest initial hyperrectangle bounds based on data analysis"""
        if self.data is None:
            self.load_data()
        
        analysis = self.analyze_feature_distributions()
        attack_data = self.data[self.data['label'] == 1]
        
        suggested_bounds = []
        
        for feature in self.feature_names:
            if method == "percentile":
                # Use percentile-based bounds
                lower = np.percentile(attack_data[feature], (100 - percentile) / 2)
                upper = np.percentile(attack_data[feature], percentile + (100 - percentile) / 2)
            elif method == "iqr":
                # Use IQR-based bounds
                q25 = analysis[feature]['attack_q25']
                q75 = analysis[feature]['attack_q75']
                iqr = q75 - q25
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
            elif method == "mean_std":
                # Use mean ± 2*std
                mean = analysis[feature]['attack_mean']
                std = analysis[feature]['attack_std']
                lower = mean - 2 * std
                upper = mean + 2 * std
            else:
                # Use min/max
                lower = analysis[feature]['attack_min']
                upper = analysis[feature]['attack_max']
            
            # Ensure reasonable bounds
            lower = max(lower, analysis[feature]['attack_min'])
            upper = min(upper, analysis[feature]['attack_max'])
            
            bound = FeatureBounds(
                name=feature,
                min_value=float(lower),
                max_value=float(upper),
                description=f"Suggested bounds for {feature} in {attack_type} attacks",
                unit=""  # Could be enhanced with actual units
            )
            suggested_bounds.append(bound)
        
        logger.info(f"Generated {len(suggested_bounds)} suggested bounds using {method} method")
        return suggested_bounds
    
    def interactive_hyperrectangle_creation(self, initial_suggestion: str = "percentile") -> Hyperrectangle:
        """
        Interactive creation of hyperrectangles with human input
        This would typically be used in a Jupyter notebook or interactive environment
        """
        print("🎯 VEHICLE-LANG HYPERRECTANGLE GENERATOR")
        print("=" * 50)
        
        # Get basic info
        name = input("Enter hyperrectangle name: ").strip()
        description = input("Enter description: ").strip()
        attack_type = input("Enter attack type (e.g., DoS, DDoS, Intrusion): ").strip()
        
        # Get suggested bounds
        suggested_bounds = self.suggest_hyperrectangle_bounds(
            attack_type=attack_type, 
            method=initial_suggestion
        )
        
        print(f"\n📊 SUGGESTED BOUNDS ({initial_suggestion} method):")
        print("-" * 50)
        
        final_bounds = []
        
        for i, bound in enumerate(suggested_bounds):
            print(f"\n{i+1}. {bound.name}")
            print(f"   Suggested range: [{bound.min_value:.6f}, {bound.max_value:.6f}]")
            
            # Allow user to modify
            choice = input("   Accept (y), modify (m), or skip (s)? [y/m/s]: ").strip().lower()
            
            if choice == 's':
                continue
            elif choice == 'm':
                try:
                    new_min = float(input(f"   Enter new minimum [{bound.min_value:.6f}]: ") or bound.min_value)
                    new_max = float(input(f"   Enter new maximum [{bound.max_value:.6f}]: ") or bound.max_value)
                    new_desc = input(f"   Enter description [{bound.description}]: ").strip() or bound.description
                    
                    bound.min_value = new_min
                    bound.max_value = new_max
                    bound.description = new_desc
                except ValueError:
                    print("   Invalid input, using suggested values")
            
            final_bounds.append(bound)
        
        # Get confidence level
        try:
            confidence = float(input(f"\nEnter confidence level (0.0-1.0) [1.0]: ") or "1.0")
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 1.0
        
        hyperrectangle = Hyperrectangle(
            name=name,
            description=description,
            feature_bounds=final_bounds,
            attack_type=attack_type,
            confidence=confidence
        )
        
        print(f"\n✅ Created hyperrectangle '{name}' with {len(final_bounds)} feature bounds")
        return hyperrectangle
    
    def add_hyperrectangle(self, hyperrectangle: Hyperrectangle):
        """Add a hyperrectangle to the collection"""
        self.hyperrectangles.append(hyperrectangle)
        logger.info(f"Added hyperrectangle: {hyperrectangle.name}")
    
    def generate_vehicle_specification(self, spec_name: str = "nids_attack_boundaries") -> str:
        """Generate complete Vehicle-lang specification file"""
        if not self.hyperrectangles:
            raise ValueError("No hyperrectangles defined. Create some first.")
        
        # Header
        spec = f"""-- NIDS Attack Boundary Specification
-- Generated by Vehicle-lang Hyperrectangle Generator
-- Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

-- Type definitions
type FeatureVector = Vector Rat {len(self.feature_names)}
type Label = Rat

-- Feature names (for documentation)
"""
        
        # Add feature name comments
        for i, name in enumerate(self.feature_names):
            spec += f"-- Feature {i}: {name}\n"
        
        spec += "\n"
        
        # Add hyperrectangle definitions
        for rect in self.hyperrectangles:
            spec += rect.to_vehicle_spec()
            spec += "\n"
        
        # Add combined attack region
        attack_predicates = [f"{rect.name.lower().replace(' ', '_')} x" 
                           for rect in self.hyperrectangles]
        
        spec += f"""
-- Combined attack region (union of all hyperrectangles)
isInAttackRegion : FeatureVector -> Bool
isInAttackRegion x = {' or '.join(attack_predicates)}

-- Property: If a sample is in any attack region, it should be classified as attack
@property
attackDetectionProperty : Bool
attackDetectionProperty = forall x .
  isInAttackRegion x => networkClassifiesAsAttack x

-- Network declaration (to be provided at verification time)
@network  
networkClassifiesAsAttack : FeatureVector -> Bool
"""
        
        # Save specification
        spec_file = self.vehicle_dir / f"{spec_name}.vcl"
        with open(spec_file, 'w') as f:
            f.write(spec)
        
        logger.info(f"Vehicle specification saved to {spec_file}")
        return str(spec_file)
    
    def validate_hyperrectangles(self) -> Dict[str, Any]:
        """Validate hyperrectangles against training data"""
        if self.data is None:
            self.load_data()
        
        validation_results = {}
        
        for rect in self.hyperrectangles:
            # Convert data to numpy for efficient processing
            features = self.data[self.feature_names].values
            labels = self.data['label'].values
            
            # Check coverage
            in_rect = np.array([rect.contains_point(point) for point in features])
            
            total_in_rect = np.sum(in_rect)
            attack_in_rect = np.sum((in_rect) & (labels == 1))
            benign_in_rect = np.sum((in_rect) & (labels == 0))
            
            precision = attack_in_rect / total_in_rect if total_in_rect > 0 else 0
            recall = attack_in_rect / np.sum(labels == 1) if np.sum(labels == 1) > 0 else 0
            
            validation_results[rect.name] = {
                'total_samples_in_rect': int(total_in_rect),
                'attack_samples_in_rect': int(attack_in_rect),
                'benign_samples_in_rect': int(benign_in_rect),
                'precision': float(precision),
                'recall': float(recall),
                'volume': float(rect.volume()),
                'confidence': float(rect.confidence)
            }
        
        # Save validation results
        validation_file = self.output_dir / "validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation results saved to {validation_file}")
        return validation_results
    
    def export_for_adversarial_training(self) -> List[np.ndarray]:
        """
        Export hyperrectangles as perturbation bounds for adversarial training
        
        IMPORTANT: This converts attack region boundaries to perturbation bounds.
        Your PGD implementation expects perturbation bounds (relative offsets),
        not absolute attack region boundaries.
        
        Returns:
            List of perturbation bounds [min_delta, max_delta] for each feature
        """
        if not self.hyperrectangles:
            logger.warning("No hyperrectangles to export")
            return []
        
        # Get all features that appear in any hyperrectangle
        all_features = set()
        for rect in self.hyperrectangles:
            for bound in rect.feature_bounds:
                all_features.add(bound.name)
        
        # Sort features for consistent ordering
        sorted_features = sorted(all_features)
        perturbation_bounds = []
        
        logger.info(f"Converting {len(self.hyperrectangles)} attack region hyperrectangles to perturbation bounds...")
        
        for feature_name in sorted_features:
            # Collect all bounds for this feature across hyperrectangles
            feature_ranges = []
            for rect in self.hyperrectangles:
                for bound in rect.feature_bounds:
                    if bound.name == feature_name:
                        feature_ranges.append((bound.min_value, bound.max_value))
            
            if feature_ranges:
                # Find the union of all attack regions for this feature
                min_attack_val = min([r[0] for r in feature_ranges])
                max_attack_val = max([r[1] for r in feature_ranges])
                
                # Convert to perturbation bounds
                # Assume clean data is typically around the middle of [0,1] range
                clean_baseline = 0.5
                
                # Calculate max perturbations needed to reach attack regions
                max_negative_delta = max(0, clean_baseline - min_attack_val)
                max_positive_delta = max(0, max_attack_val - clean_baseline)
                
                # Use symmetric bounds for robustness
                max_delta = max(max_negative_delta, max_positive_delta)
                
                # Clamp to reasonable limits
                max_delta = min(max_delta, 0.3)  # Max 30% perturbation
                max_delta = max(max_delta, 0.01)  # Min 1% perturbation
                
                # Create perturbation bound: [negative_delta, positive_delta]
                perturbation_bound = np.array([-max_delta, max_delta], dtype=np.float32)
                
                logger.debug(f"Feature {feature_name}: attack region [{min_attack_val:.3f}, {max_attack_val:.3f}] "
                           f"-> perturbation [{-max_delta:.3f}, {max_delta:.3f}]")
            else:
                # Default small perturbation if feature not found
                perturbation_bound = np.array([-0.05, 0.05], dtype=np.float32)
                logger.debug(f"Feature {feature_name}: using default perturbation [-0.05, 0.05]")
            
            perturbation_bounds.append(perturbation_bound)
        
        # Save perturbation bounds (what PGD expects)
        perturbation_file = self.output_dir / "perturbation_bounds.npy"
        np.save(perturbation_file, perturbation_bounds)
        
        # Also save original attack regions for reference/verification
        attack_regions = []
        for rect in self.hyperrectangles:
            bounds = np.array([[bound.min_value, bound.max_value] 
                             for bound in rect.feature_bounds])
            attack_regions.append(bounds)
        
        regions_file = self.output_dir / "attack_regions.npy"
        np.save(regions_file, attack_regions)
        
        logger.info(f"✅ Exported {len(perturbation_bounds)} perturbation bounds for PGD adversarial training")
        logger.info(f"📁 Perturbation bounds: {perturbation_file}")
        logger.info(f"📁 Original attack regions: {regions_file}")
        logger.warning("⚠️  Make sure to use perturbation_bounds.npy with your PGD implementation!")
        
        return perturbation_bounds
    
    def save_session(self, filename: str = "hyperrectangle_session.json"):
        """Save current session for later loading"""
        session_data = {
            'hyperrectangles': [asdict(rect) for rect in self.hyperrectangles],
            'feature_names': self.feature_names,
            'data_path': str(self.data_path),
            'output_dir': str(self.output_dir)
        }
        
        session_file = self.output_dir / filename
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session saved to {session_file}")
    
    def load_session(self, filename: str = "hyperrectangle_session.json"):
        """Load a previously saved session"""
        session_file = self.output_dir / filename
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Reconstruct hyperrectangles
        self.hyperrectangles = []
        for rect_data in session_data['hyperrectangles']:
            # Reconstruct FeatureBounds objects
            bounds = [FeatureBounds(**bound_data) for bound_data in rect_data['feature_bounds']]
            rect_data['feature_bounds'] = bounds
            self.hyperrectangles.append(Hyperrectangle(**rect_data))
        
        self.feature_names = session_data['feature_names']
        self.data_path = session_data['data_path']
        
        logger.info(f"Session loaded from {session_file}")
        logger.info(f"Loaded {len(self.hyperrectangles)} hyperrectangles")


def create_example_hyperrectangles(generator: VehicleHyperrectangleGenerator) -> List[Hyperrectangle]:
    """Create example hyperrectangles for demonstration"""
    
    # Example 1: High traffic volume DoS attack
    dos_bounds = generator.suggest_hyperrectangle_bounds("DoS", "percentile", 90)
    
    # Modify specific bounds for DoS characteristics
    for bound in dos_bounds:
        if 'packet' in bound.name.lower() or 'flow' in bound.name.lower():
            # DoS typically has high packet rates
            bound.min_value = max(bound.min_value, bound.max_value * 0.7)
        elif 'size' in bound.name.lower():
            # DoS might have smaller packet sizes
            bound.max_value = min(bound.max_value, bound.min_value + (bound.max_value - bound.min_value) * 0.3)
    
    dos_rect = Hyperrectangle(
        name="High Volume DoS",
        description="Denial of Service attack characterized by high packet volume and rate",
        feature_bounds=dos_bounds[:10],  # Use first 10 features for example
        attack_type="DoS",
        confidence=0.85
    )
    
    # Example 2: DDoS attack pattern
    ddos_bounds = generator.suggest_hyperrectangle_bounds("DDoS", "iqr")
    
    ddos_rect = Hyperrectangle(
        name="Distributed DoS",
        description="Distributed Denial of Service with multiple source characteristics",
        feature_bounds=ddos_bounds[:10],
        attack_type="DDoS", 
        confidence=0.80
    )
    
    return [dos_rect, ddos_rect]


# CLI interface
def main():
    """Main CLI interface for the Vehicle hyperrectangle generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vehicle-lang Hyperrectangle Generator for NIDS")
    parser.add_argument("--data", help="Path to training data CSV")
    parser.add_argument("--output", default="vehicle", help="Output directory")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--examples", action="store_true", help="Generate example hyperrectangles")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Create generator
    generator = VehicleHyperrectangleGenerator(args.data, args.output)
    
    # Load data
    generator.load_data()
    
    if args.visualize:
        generator.visualize_feature_distributions()
    
    if args.examples:
        print("Creating example hyperrectangles...")
        examples = create_example_hyperrectangles(generator)
        for example in examples:
            generator.add_hyperrectangle(example)
    
    if args.interactive:
        print("Starting interactive hyperrectangle creation...")
        while True:
            rect = generator.interactive_hyperrectangle_creation()
            generator.add_hyperrectangle(rect)
            
            if input("\nCreate another hyperrectangle? [y/N]: ").strip().lower() != 'y':
                break
    
    if generator.hyperrectangles:
        # Generate specifications
        spec_file = generator.generate_vehicle_specification()
        print(f"Generated Vehicle specification: {spec_file}")
        
        # Validate
        validation = generator.validate_hyperrectangles()
        print(f"Validation completed. Results in {generator.output_dir}/validation_results.json")
        
        # Export for adversarial training
        attack_rects = generator.export_for_adversarial_training()
        print(f"Exported {len(attack_rects)} hyperrectangles for adversarial training")
        
        # Save session
        generator.save_session()
        print(f"Session saved for future use")


if __name__ == "__main__":
    main()
