import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# 1️⃣ Generate Nigeria-based flight dataset
# ----------------------------
n = 10000  # number of flights
np.random.seed(42)

# Nigeria-based airlines
airlines = [
    "Arik Air", "Air Peace", "Dana Air", "Green Africa", "Ibom Air",
    "United Nigeria", "Max Air", "Overland Airways", "Aero Contractors", "Azman Air"
]

# Nigerian airports
airports = [
    "LOS", "ABV", "PHC", "KAN", "BEN", "YOLA", "LOK", "JOS", "ILR", "BAK"
]

# Generate fake dataset
df = pd.DataFrame({
    "airline": np.random.choice(airlines, n),
    "origin": np.random.choice(airports, n),
    "destination": np.random.choice(airports, n),
    "departure_hour": np.random.randint(0, 24, n),
    "distance": np.random.randint(200, 1500, n),  # domestic distances in km
    "day": np.random.randint(1, 32, n)
})

# Remove rows where origin == destination
df = df[df["origin"] != df["destination"]].reset_index(drop=True)

# Create delay column: longer distances slightly more likely delayed
df["delay"] = df["distance"].apply(lambda x: np.random.choice([0,1], p=[0.7,0.3]) if x < 800 else np.random.choice([0,1], p=[0.5,0.5]))

# Save dataset
df.to_csv("nigeria_flight_delay.csv", index=False)
print("Nigeria-based flight dataset saved!")
print(df.head())

# ----------------------------
# 2️⃣ Train Random Forest Model
# ----------------------------

# Encode categorical features
from sklearn.preprocessing import LabelEncoder

le_airline = LabelEncoder()
le_origin = LabelEncoder()
le_dest = LabelEncoder()

df["airline"] = le_airline.fit_transform(df["airline"])
df["origin"] = le_origin.fit_transform(df["origin"])
df["destination"] = le_dest.fit_transform(df["destination"])

# Features and target
X = df[["airline","origin","destination","departure_hour","distance","day"]]
y = df["delay"]

# Train Random Forest
model = RandomForestClassifier(n_estimators=20, random_state=42)  # fewer trees for fast training
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_airline, open("airline_encoder.pkl", "wb"))
pickle.dump(le_origin, open("origin_encoder.pkl", "wb"))
pickle.dump(le_dest, open("dest_encoder.pkl", "wb"))

print("Model trained and saved successfully as 'model.pkl'!")