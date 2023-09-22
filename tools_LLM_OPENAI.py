import yaml
import openai
#----------------------------------------------------------------------------------------------------------------------
def LLM(filename_config):
    with open(filename_config, 'r') as config_file:
        config = yaml.safe_load(config_file)
        openai.api_key = config['openai']['key']

    return
#----------------------------------------------------------------------------------------------------------------------
def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0,stop=["<<END>>"]):
    prompt = prompt.encode(encoding='UTF8', errors='ignore').decode()
    response = openai.Completion.create(engine=engine, prompt=prompt, temperature=temp, max_tokens=tokens, top_p=top_p,frequency_penalty=freq_pen, presence_penalty=pres_pen, stop=stop)
    text = response['choices'][0]['text'].strip()
    return text
# ----------------------------------------------------------------------------------------------------------------------