import os
import re
import time
import json
import random
import finnhub
import torch
import gradio as gr
import pandas as pd
import yfinance as yf
from pynvml import *
from peft import PeftModel
from collections import defaultdict
from datetime import date, datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import random
import matplotlib.pyplot as plt
import transformers

halal_list = ['AAPL','TSLA','JNJ','XOM','V','PG','NVDA','HD','CVX','LLY','MA','PFE','MRK']
access_token = os.getenv("hf_token")
finnhub_client = finnhub.Client(api_key = os.getenv("finnhub_key"))
# model2 = AutoModelForCausalLM.from_pretrained('./models/Llama-2-7b-chat-hf', torch_dtype=torch.float16, device_map="auto")
# model2.cuda()
# tokenizer = AutoTokenizer.from_pretrained('./models/Llama-2-7b-chat-hf')
# tokenizer.use_default_system_prompt = False
model_id = "./models/Meta-Llama-3-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )


    
base_model = AutoModelForCausalLM.from_pretrained(
    './models/Llama-2-7b-chat-hf',
    token=access_token,
    trust_remote_code=True, 
    device_map="auto",
    load_in_8bit=True,
    offload_folder="offload/"
)
model = PeftModel.from_pretrained(
    base_model,
    './models/fingpt-forecaster_dow30_llama2-7b_lora',
    offload_folder="offload/"
)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    './models/Llama-2-7b-chat-hf',
    token=access_token
)

streamer = TextStreamer(tokenizer)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
pre = """
You are a seasoned Islamic stock market analyst. Your task is to provide comprehensive guidance to a complete newbie who wants to get started with shariah law comlient investment.
 Explain what to expect, where to begin safely, and some of the best practices to follow.

[Specific Instructions]:
1. Start by explaining the importance of understanding risk and setting investment goals.
2. Provide an overview of different investment options, such as stocks, bonds, and mutual funds.
3. Explain the concept of diversification and its role in reducing risk.
4. Discuss the importance of conducting research before making investment decisions.
5. Offer tips on how to choose a reputable brokerage platform or financial advisor.
6. Emphasize the significance of long-term investing and staying patient during market fluctuations.

[Additional Context]:
- Consider using simple language and examples to make the information accessible to a newbie.
- Provide practical tips and resources that the newbie can use to start investing confidently.

Your response should be informative, easy to understand, and tailored to the needs of a beginner investor.

Whatever comes after this paragraph is The newbie's question, considering the question above, provide an answer that 
matches the below question, with the above information as context and only context. meaning that if the question is generic,
you may be guided by the above layout, but if the question isn't generic, you should answer it as is, with focusing on keeping things
complient with the Islamic Sharia-law. 
"""
# messages = [
#     {"role": "system", "content": "pre"},
# ]
def chat_with_llama(prompt):
#     messages.append({"role": "user", "content": "prompt"})
#     prompt = pipeline.tokenizer.apply_chat_template(
#             messages, 
#             tokenize=False, 
#             add_generation_prompt=True
#     )

#     terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = pipeline(
#         prompt,
#         max_new_tokens=256,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )
#     out = outputs[0]["generated_text"][len(prompt):]
#     messages.append({"role": "system", "content": out})
    out = 'fuck you'    
    return out   



SYSTEM_PROMPT = "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]\nPrediction: ...\nAnalysis: ..."


def print_gpu_utilization():
    
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def get_curday():
    
    return date.today().strftime("%Y-%m-%d")


def n_weeks_before(date_string, n):
    
    date = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)

    return date.strftime("%Y-%m-%d")


def get_stock_data(stock_symbol, steps):

    stock_data = yf.download(stock_symbol, steps[0], steps[-1])
    if len(stock_data) == 0:
        raise gr.Error(f"Failed to download stock price data for symbol {stock_symbol} from yfinance!")
    
#     print(stock_data)
    
    dates, prices = [], []
    available_dates = stock_data.index.format()
    
    for date in steps[:-1]:
        for i in range(len(stock_data)):
            if available_dates[i] >= date:
                prices.append(stock_data['Close'][i])
                dates.append(datetime.strptime(available_dates[i], "%Y-%m-%d"))
                break

    dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
    prices.append(stock_data['Close'][-1])
    
    return pd.DataFrame({
        "Start Date": dates[:-1], "End Date": dates[1:],
        "Start Price": prices[:-1], "End Price": prices[1:]
    })


def get_news(symbol, data):
    
    news_list = []
    
    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
#         print(symbol, ': ', start_date, ' - ', end_date)
        time.sleep(1) # control qpm
        weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        if len(weekly_news) == 0:
            raise gr.Error(f"No company news found for symbol {symbol} from finnhub!")
        weekly_news = [
            {
                "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                "headline": n['headline'],
                "summary": n['summary'],
            } for n in weekly_news
        ]
        weekly_news.sort(key=lambda x: x['date'])
        news_list.append(json.dumps(weekly_news))
    
    data['News'] = news_list
    
    return data


def get_company_prompt(symbol):

    profile = finnhub_client.company_profile2(symbol=symbol)
    if not profile:
        raise gr.Error(f"Failed to find company profile for symbol {symbol} from finnhub!")
        
    company_template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
        "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."

    formatted_str = company_template.format(**profile)
    
    return formatted_str


def get_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. Company news during this period are listed below:\n\n".format(
        start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    basics = json.loads(row['Basics'])
    if basics:
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."
    
    return head, news, basics


def sample_news(news, k=5):
    
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]


def get_current_basics(symbol, curday):

    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    if not basic_financials['series']:
        raise gr.Error(f"Failed to find basic financials for symbol {symbol} from finnhub!")
        
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
    
    for basic in basic_list[::-1]:
        if basic['period'] <= curday:
            break
            
    return basic
    

def get_all_prompts_online(symbol, data, curday, with_basics=True):

    company_prompt = get_company_prompt(symbol)

    prev_rows = []

    for row_idx, row in data.iterrows():
        head, news, _ = get_prompt_by_row(symbol, row)
        prev_rows.append((head, news, None))
        
    prompt = ""
    for i in range(-len(prev_rows), 0):
        prompt += "\n" + prev_rows[i][0]
        sampled_news = sample_news(
            prev_rows[i][1],
            min(5, len(prev_rows[i][1]))
        )
        if sampled_news:
            prompt += "\n".join(sampled_news)
        else:
            prompt += "No relative news reported."
        
    period = "{} to {}".format(curday, n_weeks_before(curday, -1))
    
    if with_basics:
        basics = get_current_basics(symbol, curday)
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."

    info = company_prompt + '\n' + prompt + '\n' + basics
    prompt = info + f"\n\nBased on all the information before {curday}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
        f"Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction."
        
    return info, prompt


def construct_prompt(ticker, curday, n_weeks, use_basics):

    try:
        steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]
    except Exception:
        raise gr.Error(f"Invalid date {curday}!")
        
    data = get_stock_data(ticker, steps)
    data = get_news(ticker, data)
    data['Basics'] = [json.dumps({})] * len(data)
    # print(data)
    
    info, prompt = get_all_prompts_online(ticker, data, curday, use_basics)
    
    prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST
    # print(prompt)
    
    return info, prompt


def predict(ticker, date, n_weeks, use_basics):

    print_gpu_utilization()

    info, prompt = construct_prompt(ticker, date, n_weeks, use_basics)
    with open ('prompt.txt', 'w') as f:
        f.write(prompt)  
    inputs = tokenizer(
        prompt, return_tensors='pt', padding=False
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    #print("Inputs loaded onto devices.")
        
    res = model.generate(
        **inputs, max_length=4096, do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True, streamer=streamer
    )
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

    torch.cuda.empty_cache()
    
    return info, answer

def slow_echo(message):
 
    response = chat_with_llama(message)
    for i in range(len(response)):
        time.sleep(0.01)
        yield response[: i + 1]
import random
headlines, summaries = [],[]
def extract_json(ticker, date, n_weeks, use_basics): 
    text_to_parse,answer = predict(ticker, date, n_weeks, use_basics)   
    pattern = r"From (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2}), .*? (\d+\.\d+) to (\d+\.\d+)"
    dates,prices = [],[]
    items = {'[Headlines]':[], '[Summaries]':[]}
    key,value = '',''
    t = True
    for char in text_to_parse:
        if (t and char != ':'):
            key += char
        if (char == ':'):
            t = False
            continue
        if (char == '[' and key != '['):
            t = True
            if (key == '[Headline]'):
                items['[Headlines]'].append(value.strip())
                key,value = '',''
                key += char
                continue
            if key == '[Company Introduction]':
                match = re.search(pattern, value)
                if match:
                    start_date = match.group(1)
                    end_date = match.group(2)
                    start_price = float(match.group(3))
                    end_price = float(match.group(4))
                    dates.append(start_date)
                    dates.append(end_date)
                    prices.append(start_price)
                    prices.append(end_price)
                    value = value[:value.find('From '+start_date)]
        
            if (key == '[Summary]'):
                match = re.search(pattern, value)
                if match:
                    start_date = match.group(1)
                    end_date = match.group(2)
                    start_price = float(match.group(3))
                    end_price = float(match.group(4))
                    dates.append(start_date)
                    dates.append(end_date)
                    prices.append(start_price)
                    prices.append(end_price)
                    value = value[:value.find('From '+start_date)]
                items['[Summaries]'].append(value.strip())
                key,value = '',''
                key += char
                continue
            items[key] = value.strip()
            key,value = '',''
            key += char
        if (not t):
            value += char
    nums = []
    while len(nums) < 3:
        n = random.randint(0, len(items['[Headlines]'])-1)
        if n not in nums:
            nums.append(n)
    for n in nums:
        headlines.append(items['[Headlines]'][n])
        summaries.append(items['[Summaries]'][n])
    listt = items['[Summaries]']
    items['[Summaries]'][-1]=listt[-1][:listt[-1].find('Some recent basic financials')].strip()
    listt = items['[Company Introduction]']
    items['[Company Introduction]'] = listt[:listt.find('From '+dates[0])].strip()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #line plot
    ax.set_title(f'Stock Price Movement for {ticker} from {dates[0]} to {dates[-1]}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    color = 'r' if prices[0] > prices[-1] else 'g' 
    ax.grid()
    ax.plot(dates, prices, color = color, marker = 'o')
    pred = answer[answer.find('Prediction: ')+len('Prediction: '):answer.find('Analysis: ')]
    if pred.startswith('Up'):
        pred = '# 📈 '+pred
    else:
        pred = '# 📉 '+pred
    return fig,headlines[0],summaries[0],headlines[1],summaries[1],headlines[2],summaries[2],pred,answer[answer.find('[Positive Developments]')+len('[Positive Developments]:'):answer.find('[Potential Concerns]')].strip(),answer[answer.find('[Potential Concerns]')+len('[Potential Concerns]:'):answer.find('[Prediction & Analysis]')].strip(),answer[answer.find('Analysis: ')+len('Analysis: '):]
def dummy(ticker, date, n_weeks, use_basics):
    return None,None
# demo = gr.Interface(
#     predict,
#     inputs=[
#         gr.Dropdown(
#             label="Ticker",
#             choices = halal_list,
#             info="This provides an initial list of verifiably Halal companies"
#         ),
#         gr.Textbox(
#             label="Date",
#             value=get_curday,
#             info="Date from which the prediction is made, use format yyyy-mm-dd"
#         ),
#         gr.Slider(
#             minimum=1,
#             maximum=4,
#             value=3,
#             step=1,
#             label="n_weeks",
#             info="Information of the past n weeks will be utilized, choose between 1 and 4"
#         ),
#         gr.Checkbox(
#             label="Use Latest Basic Financials",
#             value=False,
#             info="If checked, the latest quarterly reported basic financials of the company is taken into account."
#         )
#     ],
#     outputs=[
#         gr.Textbox(
#             label="Information"
#         ),
#         gr.Textbox(
#             label="Response"
#         )
#     ],
#     title="MAL Co-investor",
#     description="""This an mvp designed by MAL team to demonstrate what our product can bring to the table. For now, this retrieves
#     ready-for-analysis information about a given company, from a given date, considering a given number of weeks.

# **Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
# """
# )

demo = gr.Blocks()

with demo:
    gr.Image('MALlogo.jpeg',height=100,width=100)
    gr.Markdown(
        """
        # MAL Co-investor
        This an MVP designed by MAL team to demonstrate what our product can bring to the table. For now, this retrieves
        ready-for-analysis information about a given company, from a given date, considering a given number of weeks.

        **Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
        """
    )
    output=[]
    with gr.Tab('History and news'):
        with gr.Row():
            with gr.Column():
                input = [
                gr.Dropdown(
                    label="Ticker",
                    choices=halal_list,
                    info="This provides an initial list of verifiably Halal companies"
                ),
                gr.Textbox(
                    label="Date",
                    value=get_curday(),
                    info="Date from which the prediction is made, use format yyyy-mm-dd"
                ),
                gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=3,
                    step=1,
                    label="n_weeks",
                    info="Information of the past n weeks will be utilized, choose between 1 and 4"
                ),
                gr.Checkbox(
                    label="Use Latest Basic Financials",
                    value=False,
                    info="If checked, the latest quarterly reported basic financials of the company is taken into account."
                ),
                ]
            with gr.Column():
                with gr.Accordion(f"Stock Price Movement", open=False):
                    output.append(gr.Plot())
                with gr.Accordion(f"News Story 1", open=False):
                    output.append(gr.Markdown())
                    output.append(gr.Markdown())
                with gr.Accordion(f"News Story 2", open=False):
                    output.append(gr.Markdown())
                    output.append(gr.Markdown())
                with gr.Accordion(f"News Story 3", open=False):
                    output.append(gr.Markdown())
                    output.append(gr.Markdown())
        button = gr.Button("Submit")
    with gr.Tab('Analysis and forecast'):
        with gr.Row():
            with gr.Accordion("Prediction", open=False):
                output.append(gr.Markdown())
            with gr.Accordion(f"Positive Developments", open=False):
                output.append(gr.Markdown())
            with gr.Accordion(f"Potential Concerns", open=False):
                output.append(gr.Markdown())
            with gr.Accordion(f"Analysis", open=False):
                output.append(gr.Markdown())
        button.click(extract_json, inputs=input, outputs=output)

    with gr.Tab('Answer Prompt'):
        with gr.Column():
                inp = gr.Textbox(label="Prompt")
                out = gr.Textbox(label="Answer")
        button2 = gr.Button("Inquire!")
        button2.click(slow_echo, inputs=inp, outputs=out)
        

demo.launch()
# source venv/bin/activate (inside tmux session)
# tmux new -s session_name
# to detach ctrl=b then d
# tmux attach -t gradio