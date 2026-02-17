import pandas as pd

input_file = ""
output_file = ""

df = pd.read_csv(input_file)

inconsistent_df = df[df["consistency label"] == 0]

inconsistent_df.to_csv(output_file, index=False)

