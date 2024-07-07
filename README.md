# Bitcoin Price Prediction

**Description:**
This project aims to predict Bitcoin prices using machine learning techniques. The goal is to build a model that can forecast future Bitcoin prices based on historical data. We use a combination of preprocessing steps and neural networks to achieve this.

**Methods:**
- Data preprocessing with Pandas and MinMaxScaler
- Time series data generation using TimeseriesGenerator
- Neural network modeling with Sequential API from TensorFlow/Keras
- Model evaluation using r2_score

**Technologies:**
- Python
- Pandas
- Scikit-Learn
- TensorFlow/Keras
- Matplotlib

**Steps:**

1. **Data Loading and Preprocessing:**
   - Load the historical Bitcoin price data using Pandas.
   - Normalize the data using MinMaxScaler.
   - Split the data into training and testing sets.
   
2. **Time Series Data Generation:**
   - Use TimeseriesGenerator to create sequences of data for the neural network.

3. **Model Building:**
   - Define a Sequential model with appropriate layers for time series prediction.
   - Train the model on the training data.

4. **Model Evaluation:**
   - Evaluate the model's performance using r2_score.
   - Visualize the results with Matplotlib.

<<<<<<< HEAD
**License:**

This project is licensed under the MIT License.

=======
>>>>>>> d386e24 (Update README.md)
**Installation:**

Clone the repository and install the required packages:

```sh
git clone https://github.com/AI-models-and-cybersecurity-Python/Bitcoin-Price-Prediction.git
cd Bitcoin-Price-Prediction
pip install -r requirements.txt

