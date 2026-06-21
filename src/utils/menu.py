def select_model():
    models = {
        '1': 'unet_mobilenetv2',
        '2': 'unet_mobilenetv2_reg'
    }
    
    print("\nAvailable models:")
    print("1. unet_mobilenetv2 (Baseline UNet)")
    print("2. unet_mobilenetv2_reg (UNet with L2 Regularization)")
    
    while True:
        choice = input("Select a model [1-2]: ").strip()
        if choice in models:
            selected = models[choice]
            print(f"Selected: {selected}\n")
            return selected
        else:
            print("Invalid choice. Please select 1 or 2.")
