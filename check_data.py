import os

# Automatically detect local dataset path
possible_paths = [
    r'C:\Users\amitr\Downloads\archive (4)\plantvillage dataset\color',
    r'C:\Users\amitr\Downloads\plantvillage dataset\color',
]
dataset_path = next((p for p in possible_paths if os.path.exists(p)), None)

if not dataset_path:
    print("Error: Could not locate 'plantvillage dataset\\color' in Downloads.")
else:
    print(f"Dataset located at: {dataset_path}\n")
    classes = sorted(os.listdir(dataset_path))
    total_images = 0
    for class_name in classes:
        class_folder = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder):
            count = len(os.listdir(class_folder))
            total_images += count
            print(f"{class_name}: {count} images")
    print(f"\nSummary:")
    print(f"Total Classes : {len([c for c in classes if os.path.isdir(os.path.join(dataset_path, c))])}")
    print(f"Total Images  : {total_images}")
