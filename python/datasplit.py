import os
import shutil
import random
# Klasörlerin olup olmadığını kontrol edip, eksik olanları oluşturur.
def create_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Veri kümesini eğitim ve doğrulama setlerine ayırma fonksiyonu
def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    # Kaynak dizindeki sınıf isimlerini al
    class_names = os.listdir(source_dir)

    for class_name in class_names:
        class_path = os.path.join(source_dir, class_name)

        # Eğer class_name klasörü değilse geç
        if not os.path.isdir(class_path):
            continue

        # Eğitim ve doğrulama dizinlerinde sınıf klasörlerinin oluşturulması
        create_folders(os.path.join(train_dir, class_name))
        create_folders(os.path.join(val_dir, class_name))

        # Klasördeki resim dosyalarını listele
        images = os.listdir(class_path)
        random.shuffle(images)  # Dosyaları karıştır

        # Eğitim ve doğrulama setine ayır
        num_train = int(len(images) * split_ratio)
        train_images = images[:num_train]
        val_images = images[num_train:]

        # Eğitim setine dosyaları kopyala
        for image in train_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(train_dir, class_name, image))

        # Doğrulama setine dosyaları kopyala
        for image in val_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(val_dir, class_name, image))


# Kaynak veriyi ve hedef dizinleri belirtin
source_directory = 'dataset/data'  # Orijinal veri klasörü
train_directory = 'dataset/train'  # Eğitim veri klasörü
validation_directory = 'dataset/validation'  # Doğrulama veri klasörü

# Eğitim ve doğrulama verilerini ayırma
split_data(source_directory, train_directory, validation_directory, split_ratio=0.8)

print("Veriler başarıyla ayarlandı!")
