#Load relevant libraries
import tensorflow, matplotlib.pyplot
from tensorflow.keras import models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Load CNN model
model = models.load_model(r"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\20_Feature_Maps_CNN_Experiment_Augmented - Epochs = 15.keras")

#Load test data
class_names = ["Cyclist", "Pedestrian"]
test_full_folder  = r"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\Input - Experiment + Augmented - CNN\Test"

test_images = []
test_labels = []

for i in range(len(class_names)):
    folder = test_full_folder + "\\" + class_names[i]
    files = [file for file in tensorflow.io.gfile.listdir(folder)]

    for file in files:
        image_file = folder + "\\" + file
        image = tensorflow.io.read_file(image_file)
        image = tensorflow.image.decode_png(image, channels = 3, dtype = tensorflow.uint8, name = None)   #channel 3 = RGB image
        test_images.append(image)
        test_labels.append(i)

test_images = tensorflow.stack(test_images)
test_labels = tensorflow.convert_to_tensor(test_labels)

test_images = tensorflow.cast(test_images, tensorflow.float32) / 255.0

#Evaluate CNN model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose = 2)
print(f"Test accuracy: {test_accuracy:.2f}")

predicted_classes = tensorflow.argmax(model.predict(test_images), axis = 1)
for i in range(len(test_labels)):
    print(f"Expected: {class_names[test_labels[i]]}, Predicted: {class_names[predicted_classes[i]]}")

#Confusion matrix
confusion_matrix_cnn = confusion_matrix(test_labels, predicted_classes)
confusion_matrix_cnn_plot = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_cnn, display_labels = class_names)
confusion_matrix_cnn_plot.plot(cmap = "RdBu", text_kw = {"color": "white", "fontsize": 18})
confusion_matrix_cnn_plot.ax_.tick_params(axis = "both", labelsize = 15)
matplotlib.pyplot.title("CNN Confusion Matrix", fontsize = 20)
confusion_matrix_cnn_plot.ax_.xaxis.label.set_size(15)
confusion_matrix_cnn_plot.ax_.yaxis.label.set_size(15)
matplotlib.pyplot.savefig(r"C:\Users\Asnie\Downloads\MMW-HAT\MMW-HAT\MMW-HAT-Release\example_4_offline_proc\CNN Confusion Matrix - Experiment + Augmented.png", bbox_inches = "tight")
matplotlib.pyplot.show()
