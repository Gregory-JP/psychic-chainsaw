from dotenv import load_dotenv
import os

load_dotenv()

def wandb_key():
    return os.getenv('WANDB_API_KEY')


if __name__ == '__main__':
    print(wandb_key())