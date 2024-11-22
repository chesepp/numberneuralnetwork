import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import pandas as pd
import csv
class NeuralNetwork:
    def __init__(self, learning_rate, layer_sizes):
        self.weights = []
        self.biases = []
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        self.layer_outputs = []

        # Initialize weights and biases
        for i in range(1, len(self.layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]))  # Fix the shape for weights
            self.biases.append(np.random.randn(layer_sizes[i]))  # Biases initialized for each layer

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    def _relu(self, x):
        return np.maximum(0, x)
    def _relu_deriv(self, x):
        return np.where(x > 0, 1, 0)
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # For numerical stability
        return exp_x / np.sum(exp_x)
    def predict(self, input_vector):
        input_vector = np.array(input_vector).flatten()
        if input_vector.shape[0] != self.weights[0].shape[0]:
            raise ValueError(f"Input vector size {input_vector.shape[0]} does not match the network's input size {self.weights[0].shape[0]}")        
        activations = input_vector
        self.layer_inputs = []
        self.layer_outputs = [activations]  # Track activations for each layer

        # Forward pass through the layers
        for weight, bias in zip(self.weights, self.biases):
            layer_input = np.dot(activations, weight) + bias
            self.layer_inputs.append(layer_input)
            activations = self._sigmoid(layer_input)
            self.layer_outputs.append(activations)
        
        return self.layer_outputs[-1]
    
    def _compute_gradients(self, input_vector, target):
        input_vector = np.array(input_vector).flatten()
        self.predict(input_vector)  # Perform forward pass and store outputs
        derror_dweights = []
        derror_dbiases = []

        # Compute the error for the output layer
        prediction = self.layer_outputs[-1]
        derror_dprediction = 2 * (prediction - target)  # Squared error derivative

        # Backpropagate the error and calculate gradients
        delta = derror_dprediction * self._sigmoid_deriv(self.layer_outputs[-1])  # Output layer delta
        derror_dbiases.append(delta)
        derror_dweights.append(np.outer(self.layer_outputs[-2], delta[np.newaxis, :]))  # Gradient for output layer weights
        # Backpropagate through layers (starting from the output layer to the first hidden layer)
        for i in range(len(self.layer_sizes) - 2, -1, -1):
            if i == len(self.layer_sizes) - 2:
            # For the last hidden layer, use the output layer's weights to propagate the delta
                delta = (self.layer_outputs[-1] - target) * self._sigmoid_deriv(self.layer_outputs[i + 1])
            else:
            # For other hidden layers, propagate delta back through the weights of the next layer
                delta = np.dot(delta, self.weights[i + 1].T) * self._sigmoid_deriv(self.layer_outputs[i + 1])
        # Store the deltas for bias updates
            derror_dbiases.insert(0, delta)
        # For the first layer, use the input_vector to calculate weight gradients
            if i == 0:
                derror_dweights.insert(0, np.dot(input_vector[:, np.newaxis], delta[np.newaxis, :]))  # Use matrix multiplication
            else:
            # For hidden layers, use the activations of the previous layer to calculate weight gradients
                derror_dweights.insert(0, np.dot(self.layer_outputs[i][:, np.newaxis], delta[np.newaxis, :]))  # Use matrix multiplication

        return derror_dbiases, derror_dweights
    
    def _update_parameters(self, derror_dbiases, derror_dweights):
        # Update weights and biases using the computed gradients
        for i in range(len(self.weights)):
            if self.weights[i].shape == derror_dweights[i].shape:
                self.weights[i] -= self.learning_rate * derror_dweights[i]
                self.biases[i] -= self.learning_rate * derror_dbiases[i]
            else:
                raise ValueError(f"Shape mismatch: weights {self.weights[i].shape} vs gradients {derror_dweights[i].shape}")

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Select a random data point for training
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            input_vector = np.array(input_vector).flatten()
            target = targets[random_data_index]

            # Compute gradients and update weights and biases
            derror_dbiases, derror_dweights = self._compute_gradients(input_vector, target)
            self._update_parameters(derror_dbiases, derror_dweights)

            # Every 100 iterations, calculate the cumulative error
            if current_iteration % 100 == 0:
                cumulative_error = 0
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index].reshape(-1)
                    target = targets[data_instance_index]
                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)
                    cumulative_error += error
                cumulative_errors.append(cumulative_error)
        
        return cumulative_errors


output_dir = "training_data"
os.makedirs(output_dir, exist_ok=True)
data_file = os.path.join(output_dir, "training_data.csv")
if not os.path.exists(data_file):
    with open(data_file, "w") as f:
        f.write("pixel_data,label\n")


canvas_size = (8, 8)
# Initialize a blank canvas (RGB white)
canvas = np.zeros((canvas_size[0], canvas_size[1],3))
  # Blank canvas
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Draw a Digit")
ax.axis("on")
img = ax.imshow(canvas)
# Drawing parameters
drawing_color = [255, 255, 255]  
brush_size = 0 

def on_mouse_move(event):
    if event.inaxes == ax and event.button == 1:  # Left mouse button click
        if event.xdata is not None and event.ydata is not None:
            # Correct for 0.5 offset by subtracting 0.5 and then flooring to the nearest integer
            grid_x = int(event.xdata + 0.5)
            grid_y = int(event.ydata + 0.5)

            # Ensure the coordinates are within bounds (0 to 7 for an 8x8 grid)
            if 0 <= grid_x < canvas_size[1] and 0 <= grid_y < canvas_size[0]:
                canvas[grid_y, grid_x] = drawing_color  # Update the canvas

                # Refresh the image on the plot
                img.set_data(canvas)
                fig.canvas.draw_idle()
# Connect the mouse event to the function
fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
fig.canvas.mpl_connect("button_press_event", on_mouse_move)

# Function to reset the canvas
def reset_canvas(event):
    global canvas
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)  # Clear the canvas (set all pixels to black)
    img.set_data(canvas)  # Update the image
    if is_training == False: 
        ax.set_title("Draw A Digit")
    fig.canvas.draw_idle()
      # Redraw the canvas

reset_button_ax = plt.axes([0.75, 0.02, 0.2, 0.075])  # Position of the reset button
reset_button = Button(reset_button_ax, "Reset", color='red', hovercolor='lightcoral')
reset_button.on_clicked(reset_canvas)


#instantiate neural network object
learning_rate = 0.008
layer_sizes =[64,32,16,10]
#instantiate neural network object
nn = NeuralNetwork(learning_rate, layer_sizes)
#data for training neural network to become more accurate

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]
#TRAINING IMAGE RECOG

def load_training_data(file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    # Ensure pixel_data column is present
    print(data.head)
    if 'pixel_data' not in data.columns:
        raise KeyError("'pixel_data' column not found in the dataset.")
    if 'label' not in data.columns:
        raise KeyError("'label' column not found in the dataset.")
    # Convert pixel_data to strings to ensure compatibility with np.fromstring
    data['pixel_data'] = data['pixel_data'].astype(str)
    # Convert pixel_data from comma-separated strings to NumPy arrays
    def parse_pixel_data(row):
        try:
            # Remove extra spaces and split by comma
            #row = data['pixel_data']
            row = str(row).strip()  # Remove leading/trailing whitespace
            #print(f"Raw row: {row}")
            pixel_values = row.split(',')
            #print(f"pixel values after split: {pixel_values}")
            if len(pixel_values) > 64:
                label = pixel_values[-1]  # The last value should be the label
                pixel_values = pixel_values[:-1]  # Remove the label from the pixel list
            else:
                label = None  # Split by comma to get the pixel values
            pixel_array = []
            for value in pixel_values:
                try:
                    # Strip spaces and convert to float
                    pixel_array.append(float(value.strip()))
                except ValueError as e:
                    print(f"Error converting '{value}' to float: {e}")
                    pixel_array.append(0.0)
            #print(f"label: {label} \n pixel array: {pixel_array}")
            
              # Convert to float
            if len(pixel_array) != 64:  # Ensure correct size (64 pixels)
                raise ValueError(f"Invalid pixel data size: {len(pixel_array)}")      
            return pixel_array,label
        except Exception as e:
            print(f"Error parsing row: {row}. {e}")
            return None  # Return None for invalid rows
    #print(f"LABEL PREPARSE: {data['label']} \n PIXEL DATA BEFORE PARSE{data['pixel_data']}")
    
    data['parsed_pixel_data'] = data['pixel_data'].apply(lambda x: parse_pixel_data(x)[0])  # Extract only the pixel array
    # If you want to also store the label:
    # Drop rows with invalid or missing pixel data
    valid_data = data.dropna(subset=['parsed_pixel_data'])
    # Stack pixel data into a 2D array (num_samples, 64)
    if valid_data.empty:
        raise ValueError("No valid pixel data found after cleaning.")
    
    inputs = np.stack(valid_data['parsed_pixel_data'].values)
    labels = valid_data['label'].values
    # Ensure labels are present
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integers.")

    # Ensure labels are within the expected range
    if not np.all((0 <= labels) & (labels < 10)):
        raise ValueError("Labels must be between 0 and 9.")
    # Extract labels as a NumPy array
    print(f"loaded {len(inputs)} valid samples.")
    return inputs, labels



def print_csv_data(data_file):
    with open(data_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # Skip the header row
        header = next(csvreader)
        print("Header:", header)
        
        # Find the index of 'pixel_data' and 'label' columns
        pixel_data_index = header.index('pixel_data')
        label_index = header.index('label')

        print("\nContents of CSV file:")
        # Loop through the rows and print the pixel data and label
        for row in csvreader:
            pixel_data = row[pixel_data_index]
            label = row[label_index]
            print(f"Pixel Data: {pixel_data}, Label: {label}")

print_csv_data(data_file)
#directory to save training data
is_training = None
is_training = False

def on_done(event):
    global canvas
    global is_training
    # Convert to grayscale
    print(f"Canvas shape: {canvas.shape}")
    grayscale_canvas = np.mean(canvas, axis=2)  # Average RGB to grayscale
    print(f"grayscale_canvas shape: {grayscale_canvas.shape}")
    normalized_canvas = grayscale_canvas / 255.0  # Normalize to range [0, 1]
    
    # Example: Flatten canvas for prediction
    flattened_canvas = normalized_canvas.flatten()
    print(f"flattened canvas shape: {len(flattened_canvas)}")

    if is_training == False:
        print("processing the drawing...")
        print("grayscale canvas shape: ", grayscale_canvas.shape)
        output = nn.predict(flattened_canvas)
        predicted_digit = np.argmax(output)
        print(f"predicted digit: {predicted_digit}")
        ax.set_title(f"Predicted Digit: {predicted_digit}", fontsize=16)
        fig.canvas.draw_idle()

    elif is_training == True:
        global target_digit
        if np.sum(canvas) == 0:
            print("Canvas is empty. Draw something before saving.")
            return

        # Flatten and validate the canvas data
        if len(flattened_canvas) != 64:
            raise ValueError(f"Flattened canvas size is {len(flattened_canvas)}, expected 64.")

        # Validate the target digit
        if not (0 <= target_digit <= 9):
            raise ValueError(f"Invalid target digit: {target_digit}")
        
        row = ",".join(map(str, flattened_canvas))
        # Save the row into the data file
        file_exists = os.path.exists(data_file)
        with open(data_file, mode="a",newline='') as f:
            if not file_exists or os.stat(data_file).st_size == 0:  # Add header only if file is new/empty
                f.write("pixel_data,label\n")
        # Write the data row
            print(f"row: {row}")  # Convert pixels to a comma-separated string
            df = pd.DataFrame({'pixel_data': [row],'label': [target_digit]})
            df.to_csv(data_file,mode="a",header=not file_exists,index=False)
            print(f"saved image data of target digit: {target_digit} to training_data.csv")

    is_training = False
    #reset_canvas(event)
        
# Add a "Done" button
done_button_ax = plt.axes([0.02, 0.02, 0.2, 0.075])  # Position of the button
done_button = Button(done_button_ax, "Done", color='green', hovercolor='lightgreen')
done_button.on_clicked(on_done)

target_digit = None

def display_training_digit(event):
    global target_digit
    target_digit = np.random.randint(0, 10)  # Generate a random digit
    global is_training 
    is_training = True
    ax.set_title(f"Draw the digit: {target_digit}", fontsize=16)
    fig.canvas.draw_idle()
    print(f"Digit to draw: {target_digit}")

button_ax_train = plt.axes([0.4, 0.02, 0.2, 0.075])  # Axes for the "Training" button
button_train = Button(button_ax_train, "Training")
button_train.on_clicked(display_training_digit)
# Show the interactive canvas

inputs, labels = load_training_data(data_file)
targets = one_hot_encode(labels)

print(f"Inputs shape: {inputs.shape}")  # Should be (num_samples, 64)
print(f"Targets shape: {targets.shape}")

training_error = nn.train(inputs, targets, 100000)
#plot the training data

plt.figure(figsize=(10,6))
plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.show()



