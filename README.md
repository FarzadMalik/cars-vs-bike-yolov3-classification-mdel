![image](https://github.com/FarzadMalik/cars-vs-bike-yolov3-classification-mdel/assets/107833662/ac16d2cb-f0c5-4af2-b52b-9f2bc66439f2)

# Car vs Bike Classification

Classify images of cars and bikes using deep learning.

## Table of Contents

- [Description](#description)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [License](#license)

## Description

This project involves the classification of images into two classes: "car" and "bike." A deep learning model is trained on the provided dataset to perform the classification.

## Dataset

The dataset used in this project is the [Car vs Bike Classification Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset). It contains a collection of images of cars and bikes, which are divided into training and testing sets. The dataset is organized into separate folders for each class.

## Dependencies

List the external libraries and packages required to run your project. You can reference the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Installation

To use and run this project, you need to follow these installation steps:

1. **Clone the Repository:**

   Open a terminal and navigate to the directory where you want to store the project. Then, run the following command to clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```

   Replace `your-username` and `your-repo` with your actual GitHub username and repository name.

2. **Create a Virtual Environment (Optional but Recommended):**

   It's recommended to create a virtual environment to isolate the project's dependencies. Navigate into the project directory and create a virtual environment. You can use `venv` or `conda`, depending on your preference:

   Using `venv` (for Python 3):

   ```bash
   cd your-repo
   python3 -m venv venv
   ```

   Activate the virtual environment:

   On macOS and Linux:

   ```bash
   source venv/bin/activate
   ```

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Install the required packages and dependencies listed in the `requirements.txt` file using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset:**

   Download the Car vs Bike Classification Dataset from [here](https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset) and extract it.

5. **Organize Dataset:**

   Organize the dataset into separate folders for training and testing images. Make sure the folder structure matches the expected format for your code.

6. **Run the Project:**

   You can now run the project using the provided code or Jupyter Notebook. For example, if you have a Jupyter Notebook named `car_vs_bike_simplified_yolov3_classification_model.ipynb`, you can open it using Jupyter Notebook or JupyterLab and follow the instructions in the notebook to train and use the model.

Remember to deactivate the virtual environment when you're done working on the project:

```bash
deactivate
```


## Usage

This section provides detailed instructions on how to use the project to classify images of cars and bikes using the trained YOLOv3-based model.

1. **Training the Model:**

   To train the classification model, follow these steps:

   - Open the Jupyter Notebook named `car_vs_bike_simplified_yolov3_classification_model.ipynb`.
   - Run the cells in the notebook sequentially. The notebook guides you through the process of loading and preprocessing the dataset, creating the YOLOv3-based model architecture, and training the model.
   - You can adjust hyperparameters, training settings, and other parameters as needed.

   ```python
   # Example code snippet for training the model
   # Run this code in the provided Jupyter Notebook
   # Make sure you have followed the dataset organization steps
   # and have activated your virtual environment

   # Load and preprocess the dataset
   train_dataset = Dataset(train_files, train_dir, transform=transformations)
   val_dataset = Dataset(val_files, val_dir, transform=transformations)

   # Create DataLoader instances for training and validation
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

   # Create an instance of the YOLOv3-based model
   model = YOLOv3()

   # Define loss function and optimizer
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=learning_rate)

   # Train the model
   for epoch in range(epochs):
       # Training loop
       # ...

       # Validation loop
       # ...

       # Print and visualize training progress
       # ...

   # Save the trained model
   torch.save(model.state_dict(), 'car_vs_bike_yolov3.pth')
   ```

2. **Inference with the Trained Model:**

   Once the model is trained, you can use it to perform inference on new images:

   - Load the trained model using `torch.load()`.
   - Preprocess a new image, resize it to the required dimensions, and normalize the pixel values.
   - Pass the preprocessed image through the trained model to obtain predictions.
   - Interpret the model's predictions and classify the image as a car or a bike.

   ```python
   # Example code snippet for performing inference
   # Load the trained model
   loaded_model = YOLOv3()
   loaded_model.load_state_dict(torch.load('car_vs_bike_yolov3.pth'))
   loaded_model.eval()

   # Preprocess a new image
   new_image = Image.open('new_image.jpg')
   new_image = transformations(new_image).unsqueeze(0)

   # Perform inference
   with torch.no_grad():
       output = loaded_model(new_image)

   # Interpret predictions
   class_probs = F.softmax(output, dim=1)
   predicted_class = torch.argmax(class_probs).item()

   if predicted_class == 0:
       print("The image contains a car.")
   else:
       print("The image contains a bike.")
   ```

## Training

To train the classification model on the Car vs Bike dataset, follow these steps:

1. Download the dataset from [here](https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset).
2. Extract the dataset and organize it into separate folders for training and testing.
3. Run the training script provided in the Jupyter Notebook: `car_vs_bike_simplified_yolov3_classification_model.ipynb`.
4. Monitor the training progress and evaluate the model's performance.

You can also download the pre-trained model weights: `car_vs_bike_yolov3.pth`.

## Inference

After training, you can use the trained model for inference on new images. Provide code examples and instructions on how to load the model and perform inference.

## Final Results
![image](https://github.com/FarzadMalik/cars-vs-bike-yolov3-classification-mdel/assets/107833662/7e331290-2d1e-4fa9-9b7c-319a2f307425)

![image](https://github.com/FarzadMalik/cars-vs-bike-yolov3-classification-mdel/assets/107833662/8d9a0c77-1a10-4dbb-91a4-6598e2fb4a42)

## License

This project is licensed under the [MIT License](LICENSE).
