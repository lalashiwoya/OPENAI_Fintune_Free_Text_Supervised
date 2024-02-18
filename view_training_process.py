from utils import print_training_process, load_toml
import openai
import argparse


if __name__ == "main":
    parser = argparse.ArgumentParser(description='View training process')
    parser.add_argument('response_id', type=str, help='ID of response, should be printed after running run.py')
    args = parser.parse_args()
    print("response id is ", args.response_id)
    config = load_toml("fine_tune_specification.toml")
    client = openai.OpenAI(api_key=config['model']['OPENAI_API_KEY'])

    print_training_process(args.response_id, client)



