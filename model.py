import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load your EAR/MAR dataset
data = pd.read_csv("data/drowsiness_data.csv")
X = data[['EAR', 'MAR']]
y = data['Label']

# Build ANN
model = Sequential([
    Dense(64, input_dim=2, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=30, batch_size=8)

# Save model
model.save("drowsiness_ann_model.h5")
