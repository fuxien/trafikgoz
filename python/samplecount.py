import os

# Train klasörünün yolu
train_dir = 'dataset/train'
test_dir='dataset/validation'
# Her sınıfın bulunduğu klasörleri al
class_dirs = os.listdir(train_dir)
test_dirs = os.listdir(test_dir)
toplam=0
test={}
train={}
# Her sınıf için örnek sayısını say
for class_dir in class_dirs:
    class_path = os.path.join(train_dir, class_dir)

    if os.path.isdir(class_path):  # Eğer bu bir klasörse
        num_images = len(os.listdir(class_path))  # Klasördeki dosya sayısını al
        toplam+=num_images
        train[class_dir]=num_images
        #print(f"Sınıf: {class_dir}, Örnek Sayısı: {num_images}")
for t_dir in test_dirs:
    class_path = os.path.join(test_dir, t_dir)

    if os.path.isdir(class_path):  # Eğer bu bir klasörse
        num_images = len(os.listdir(class_path))  # Klasördeki dosya sayısını al
        toplam+=num_images
        test[t_dir]=num_images

for k in train.keys():
    print(k+" "+str(train[k])+" "+str(test[k]))
print("Toplam:",toplam)
