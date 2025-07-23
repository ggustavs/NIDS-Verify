#!/usr/bin/env python3
"""
NIDS Model Management CLI
"""
import argparse
import sys
from pathlib import Path
from src.models.registry import NIDSModelRegistry, create_model_comparison_report
from src.models.evaluation import ModelEvaluator
from src.utils.logging import get_logger

logger = get_logger(__name__)


def register_models_from_directory(models_dir: str, pattern: str = "*.keras") -> int:
    """Register all models from a directory"""
    import tensorflow as tf
    
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"❌ Directory not found: {models_dir}")
        return 0
    
    model_files = list(models_path.glob(pattern))
    print(f"📁 Found {len(model_files)} model files")
    
    registry = NIDSModelRegistry()
    registered_count = 0
    
    for model_file in model_files:
        try:
            # Parse filename for parameters (assuming DoS2 naming convention)
            filename = model_file.name.replace('.keras', '')
            parts = filename.split('_')
            
            if len(parts) >= 8 and filename.startswith('adv_'):
                # Parse adv_architecture_bs{batch}_epsilon_train_acc_adv_acc_test_acc
                architecture = parts[1]
                batch_size = parts[2].replace('bs', '')
                epsilon = parts[3]
                train_acc = parts[5]
                adv_acc = parts[7]
                test_acc = parts[9] if len(parts) > 9 else parts[8]
                
                model_name = f"nids-dos2-{architecture}-eps{epsilon}-bs{batch_size}"
                
                params = {
                    'model_type': architecture,
                    'training_type': 'adversarial',
                    'batch_size': int(batch_size),
                    'pgd_epsilon': float(epsilon),
                    'architecture': architecture
                }
                
                metrics = {
                    'final_train_accuracy': float(train_acc),
                    'final_val_accuracy': float(test_acc),
                    'adversarial_accuracy': float(adv_acc),
                    'pgd_epsilon': float(epsilon)
                }
                
                description = (
                    f"Adversarially trained NIDS model for DoS detection. "
                    f"Architecture: {architecture}, Epsilon: {epsilon}, "
                    f"Test Acc: {test_acc}, Adv Acc: {adv_acc}"
                )
                
                print(f"📝 Registering: {model_name}")
                registry.register_model_from_path(str(model_file), model_name, params, metrics, description)
                print(f"✅ Registered {model_name}")
                registered_count += 1
            else:
                # Generic registration for other model types
                model_name = filename.replace('_', '-').lower()
                print(f"📝 Registering: {model_name}")
                
                # Basic parameters
                params = {'model_type': 'unknown', 'training_type': 'unknown'}
                metrics = {}
                description = f"Model registered from {model_file.name}"
                
                registry.register_model_from_path(str(model_file), model_name, params, metrics, description)
                print(f"✅ Registered {model_name}")
                registered_count += 1
                
        except Exception as e:
            print(f"❌ Failed to register {model_file.name}: {e}")
            continue
    
    return registered_count


def evaluate_model_command(model_name: str, version: str = None, stage: str = "None",
                          report_dir: str = None, show_plots: bool = False):
    """Evaluate a model and optionally generate reports"""
    evaluator = ModelEvaluator()
    
    try:
        print(f"🔬 Evaluating model: {model_name}")
        eval_result = evaluator.evaluate_model(model_name, stage, version)
        
        print(f"✅ Model: {eval_result['model_name']}")
        print(f"📊 Test Accuracy: {eval_result['test_accuracy']:.4f}")
        print(f"📈 AUC Score: {eval_result['auc_score']:.4f if eval_result['auc_score'] else 'N/A'}")
        print(f"📋 Test Set Size: {eval_result['test_size']} samples")
        print()
        
        # Show classification report
        report = eval_result['classification_report']
        print("📋 Classification Report:")
        print("-" * 40)
        print(f"Benign Traffic:")
        print(f"  Precision: {report['0']['precision']:.4f}")
        print(f"  Recall: {report['0']['recall']:.4f}")
        print(f"  F1-Score: {report['0']['f1-score']:.4f}")
        print()
        print(f"Attack Traffic:")
        print(f"  Precision: {report['1']['precision']:.4f}")
        print(f"  Recall: {report['1']['recall']:.4f}")
        print(f"  F1-Score: {report['1']['f1-score']:.4f}")
        print()
        
        # Show confusion matrix
        conf_matrix = eval_result['confusion_matrix']
        print("🔍 Confusion Matrix:")
        print("-" * 25)
        print("       Predicted")
        print("       Benign  Attack")
        print(f"Benign   {conf_matrix[0][0]:4d}    {conf_matrix[0][1]:4d}")
        print(f"Attack   {conf_matrix[1][0]:4d}    {conf_matrix[1][1]:4d}")
        print()
        
        # Generate full report if requested
        if report_dir:
            report_path = evaluator.generate_evaluation_report(
                model_name, stage, version, report_dir
            )
            print(f"📄 Full report generated: {report_path}")
        
        # Show plots if requested
        if show_plots:
            evaluator.plot_confusion_matrix(eval_result)
            if eval_result['roc_curve']:
                evaluator.plot_roc_curve(eval_result)
        
    except Exception as e:
        print(f"❌ Failed to evaluate model: {e}")


def main():
    parser = argparse.ArgumentParser(description="NIDS Model Management CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List models
    list_parser = subparsers.add_parser('list', help='List all registered models')
    
    # Model info
    info_parser = subparsers.add_parser('info', help='Get detailed model information')
    info_parser.add_argument('model_name', help='Name of the model')
    
    # Compare models
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('--models', nargs='+', help='Model names to compare (default: all)')
    compare_parser.add_argument('--output', help='Output file for comparison report')
    
    # Promote model
    promote_parser = subparsers.add_parser('promote', help='Promote model to production')
    promote_parser.add_argument('model_name', help='Name of the model')
    promote_parser.add_argument('version', help='Version to promote')
    promote_parser.add_argument('--stage', default='Production', 
                               choices=['Staging', 'Production', 'Archived'],
                               help='Stage to promote to')
    
    # Get best model
    best_parser = subparsers.add_parser('best', help='Find best performing model')
    best_parser.add_argument('--metric', default='final_val_accuracy',
                            help='Metric to optimize for')
    best_parser.add_argument('--stage', help='Filter by stage')
    
    # Load model for testing
    load_parser = subparsers.add_parser('load', help='Load model for testing')
    load_parser.add_argument('model_name', help='Name of the model')
    load_parser.add_argument('--version', help='Specific version (default: latest production)')
    load_parser.add_argument('--stage', default='Production', help='Stage to load from')
    
    # Delete model version
    delete_parser = subparsers.add_parser('delete', help='Delete model version')
    delete_parser.add_argument('model_name', help='Name of the model')
    delete_parser.add_argument('version', help='Version to delete')
    delete_parser.add_argument('--confirm', action='store_true', 
                              help='Confirm deletion without prompt')
    
    # Register models from directory
    register_parser = subparsers.add_parser('register', help='Register models from directory')
    register_parser.add_argument('--dir', default='models/tf/DoS2', 
                                help='Directory containing .keras model files')
    register_parser.add_argument('--pattern', default='*.keras',
                                help='File pattern to match (default: *.keras)')
    
    # Evaluate model
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a registered model')
    evaluate_parser.add_argument('model_name', help='Name of the model to evaluate')
    evaluate_parser.add_argument('--version', help='Specific version to evaluate')
    evaluate_parser.add_argument('--stage', default='None', help='Stage to evaluate (default: None)')
    evaluate_parser.add_argument('--report', help='Generate full report and save to directory')
    evaluate_parser.add_argument('--plots', action='store_true', help='Show plots during evaluation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    registry = NIDSModelRegistry()
    
    try:
        if args.command == 'list':
            models = registry.list_models()
            if models:
                print("Registered Models:")
                for model in models:
                    print(f"  - {model}")
            else:
                print("No models found in registry.")
        
        elif args.command == 'info':
            info = registry.get_model_info(args.model_name)
            if info:
                print(f"Model: {info['name']}")
                print(f"Description: {info.get('description', 'N/A')}")
                print(f"Created: {info['creation_timestamp']}")
                print(f"Last Updated: {info['last_updated_timestamp']}")
                print(f"Tags: {info.get('tags', {})}")
                print("\nVersions:")
                for version in info['versions']:
                    print(f"  Version {version['version']} ({version['stage']})")
                    print(f"    Created: {version['creation_timestamp']}")
                    print(f"    Run ID: {version['run_id']}")
                    if version['description']:
                        print(f"    Description: {version['description']}")
            else:
                print(f"Model '{args.model_name}' not found.")
        
        elif args.command == 'compare':
            report = create_model_comparison_report(args.models)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"Comparison report saved to {args.output}")
            else:
                print(report)
        
        elif args.command == 'promote':
            registry.promote_model(args.model_name, args.version, args.stage)
        
        elif args.command == 'best':
            best = registry.get_best_model(args.metric, args.stage)
            if best:
                print(f"Best model by {args.metric}:")
                print(f"  Name: {best['name']}")
                print(f"  Version: {best['version']}")
                print(f"  Stage: {best['stage']}")
                print(f"  {args.metric}: {best['metric_value']:.4f}")
            else:
                print("No models found.")
        
        elif args.command == 'load':
            model = registry.load_model(args.model_name, args.stage, args.version)
            if model:
                print(f"Model loaded successfully!")
                print(f"Input shape: {model.input.shape}")
                print(f"Output shape: {model.output.shape}")
            else:
                print("Failed to load model.")
        
        elif args.command == 'delete':
            if not args.confirm:
                confirm = input(f"Are you sure you want to delete {args.model_name} v{args.version}? (y/N): ")
                if confirm.lower() != 'y':
                    print("Deletion cancelled.")
                    return
            registry.delete_model_version(args.model_name, args.version)
            
        elif args.command == 'register':
            success_count = register_models_from_directory(args.dir, args.pattern)
            print(f"Successfully registered {success_count} models from {args.dir}")
            
        elif args.command == 'evaluate':
            evaluate_model_command(args.model_name, args.version, args.stage, 
                                 args.report, args.plots)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
