import pandas as pd
import numpy as np
import os

df = pd.read_csv("data/food_waste_raw.csv")

df = df[['week', 'num_orders']]

df.rename(columns={'num_orders': 'People_Served'}, inplace=True)

df['Food_Prepared'] = df['People_Served'] + np.random.randint(10, 40, size=len(df))

df['Food_Wasted'] = df['Food_Prepared'] - df['People_Served']

df['Day_Type'] = df['week'] % 2

df['Event_Day'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

final_df = df[['Food_Prepared', 'People_Served', 'Day_Type', 'Event_Day', 'Food_Wasted']]

final_df.to_csv("data/food_waste.csv", index=False)

print("✅ Food waste dataset created successfully!")
