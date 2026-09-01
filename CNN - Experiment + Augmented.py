#Import relevant Python libraries
import tensorflow
from tensorflow.keras import layers, models
import matplotlib.pyplot

#Load Train/Test images
train_full_folder = r"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\Input - Experiment + Augmented - CNN\Train"
test_full_folder  = r"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\Input - Experiment + Augmented - CNN\Test"

class_names = ["Cyclist", "Pedestrian"]

height = 480
width = 640

train_images = []
train_labels = []

for i in range(len(class_names)):
    folder = train_full_folder + "\\" + class_names[i]
    files = [file for file in tensorflow.io.gfile.listdir(folder)]

    for file in files:
        image_file = folder + "\\" + file
        image = tensorflow.io.read_file(image_file)
        image = tensorflow.image.decode_png(image, channels = 3, dtype = tensorflow.uint8, name = None)
        train_images.append(image)
        train_labels.append(i)

train_images = tensorflow.stack(train_images)
train_labels = tensorflow.convert_to_tensor(train_labels)

test_images = []
test_labels = []

for i in range(len(class_names)):
    folder = test_full_folder + "\\" + class_names[i]
    files = [file for file in tensorflow.io.gfile.listdir(folder)]

    for file in files:
        image_file = folder + "\\" + file
        image = tensorflow.io.read_file(image_file)
        image = tensorflow.image.decode_png(image, channels = 3, dtype = tensorflow.uint8, name = None)
        test_images.append(image)
        test_labels.append(i)

test_images = tensorflow.stack(test_images)
test_labels = tensorflow.convert_to_tensor(test_labels)

#Normalize pixel values to be between 0 and 1
train_images = tensorflow.cast(train_images, tensorflow.float32) / 255.0
test_images = tensorflow.cast(test_images, tensorflow.float32) / 255.0


#Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (height, width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = "relu")
])

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(len(class_names)))

#Compile CNN model
model.compile(
    optimizer = "adam",
    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ["accuracy"]
)

#Train CNN model
epochs = 15
history = model.fit(
    train_images, train_labels,
    epochs = epochs,
    validation_data = (test_images, test_labels)
)

#Save CNN model
model.save(rf"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\CNN_Experiment_Augmented - Epochs = {epochs}.keras")

#Evaluate CNN model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose = 2)
print(f"Test accuracy: {test_accuracy:.2f}")

predicted_classes = tensorflow.argmax(model.predict(test_images), axis = 1)
for i in range(len(test_labels)):
    print(f"Expected: {class_names[test_labels[i]]}, Predicted: {class_names[predicted_classes[i]]}")


