import pandas as pd

# Replace 'your_file.parquet' with the actual path to your Parquet file
file_path = '/home/adminad/wyd/Absolute-Zero-Reasoner-master/checkpoints/code_io/azr/azr_coder7b/test_answer/Qwen2.5-Coder-7B/answer_conditional/code/train_gen_code_i.parquet'

try:
    df = pd.read_parquet(file_path)

    # Select the desired columns
    selected_columns_df = df[['prompt', 'problem', 'reward_model']]

    # Define the output CSV file path
    output_csv_path = 'selected_data.csv'

    # Save the selected DataFrame to a CSV file
    # index=False prevents pandas from writing the DataFrame index as a column in the CSV
    selected_columns_df.to_csv(output_csv_path, index=False)

    print(f"Successfully saved 'prompt', 'problem', and 'reward_model' columns to '{output_csv_path}'")
    print("\nFirst 5 rows of the saved data (as it would appear in the CSV):")
    print(selected_columns_df.head())

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")