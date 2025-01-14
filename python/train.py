# Gerekli Kütüphanelerin İçe Aktarılması
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch
import pandas as pd
# 1. Veri Hazırlığı ve Ön İşleme
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


test_datagen = ImageDataGenerator(rescale=1./255)
batch_size=64
train_generator = train_datagen.flow_from_directory(
    'dataset/train/',
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'dataset/validation/',
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Tahminlerle doğru eşleşme için shuffle kapatılır
)
print(len(train_generator),len(validation_generator))
# 2. Modelin Oluşturulması

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(91, activation='softmax')  # 91 sınıf için
])

# 3. Modelin Derlenmesi
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Modelin Eğitilmesi
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=150,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# 5. Tahminlerin Elde Edilmesi
true_labels = validation_generator.classes
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)

# 6. Precision, Recall ve F1-Score Hesaplama
report = classification_report(true_labels, predicted_labels, target_names=validation_generator.class_indices.keys())
print("Classification Report:")
print(report)
# 9. Test Sonuçları Yazdırma
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# 9. Modeli Kaydetme
model.save('trafik_isaretleri_modeli128.h5')
