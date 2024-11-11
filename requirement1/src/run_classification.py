def run_cnn_classification(args):
    """Run CNN-based classification."""
    print(f"\nRunning CNN classification with {args.model_name}...")
    
    # Initialize classifier
    classifier = CNNClassifier(model_name=args.model_name,
                             results_dir=args.results_dir)
    
    try:
        # Train and evaluate
        classifier.train(args.train_paths, args.val_paths)
        
        print("\nClassification experiments completed!")
        print(f"Results saved in: {args.results_dir}")
        
    except Exception as e:
        print(f"\nError in CNN processing: {str(e)}")
        print("Continuing with available results...") 