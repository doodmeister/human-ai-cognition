import os
import torch
from memory.short_term_memory import ShortTermMemory
from model.dpad_rnn import DPADRNN
from train_dpad import train_dpad

def lambda_handler(event, context):
    """
    AWS Lambda handler to trigger DPADRNN training.
    """
    try:
        print("[Lambda] Starting DPAD Training Job")

        stm = ShortTermMemory()
        all_data = stm.load_all()

        if not all_data:
            print("[Lambda] No STM data found for training.")
            return {"statusCode": 200, "body": "No data to train on."}

        training_data = [(item['embedding'], item['flags']) for item in all_data]

        train_dpad(
            training_data=training_data,
            input_size=int(os.getenv("VECTOR_DIM", 768)),
            output_size=1,
            epochs=10,
            batch_size=32,
            save_path="/tmp/dpad_trained.pt"
        )

        print("[Lambda] DPAD training completed successfully.")
        return {"statusCode": 200, "body": "DPAD training complete."}

    except Exception as e:
        print(f"[Lambda] Error during training: {str(e)}")
        return {"statusCode": 500, "body": str(e)}
