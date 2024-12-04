import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.figure_factory as ff
import gradio as gr
import cv2
import base64
import numpy as np
import io
from pydub import AudioSegment
import os
from dashscope import MultiModalConversation
from dashscope.audio.tts_v2 import *
from dashscope import Generation
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import requests
import random
import time
import json, re
import dashscope

from google.colab import userdata


def func1():
    return "hello"

def encode_image(frame):
    """Encodes an image frame to base64 format.

    Args:
        frame: A NumPy array representing the image frame.

    Returns:
        A string containing the base64 encoded image.
    """
    # Convert the frame to JPEG format
    retval, buffer = cv2.imencode('.jpg', frame)

    # Encode the JPEG buffer to base64
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return encoded_image


def get_score_promppt():

    score_prompt = \
    """
    # 注意力打分机器人

    ## 评价标准

    - 1分：用户表现为完全不专注。例如，用户频繁查看手机、与周围人聊天、不断走动或完全背对屏幕。表情可能表现为放松、大笑、困倦等。用户的眼神经常游离，未能集中在工作任务上，头部、手部动作与工作内容无明显关联。

    - 2分：用户出现明显分心动作。例如，用户频繁侧目、时常发呆、偶尔查看手机或与人互动。表情较为疲惫或者无聊，可能有叹气、眯眼等表现。用户偶尔能够回到工作任务，但难以持续集中，头部和眼神间断性转向工作内容。

    - 3分：用户有一定专注力，但间或出现分心行为。例如，用户主要关注屏幕和工作任务，但会短暂走神或查看手机。面部表情相对平静，可能偶尔显现困倦或者轻微微笑。身体姿态大致维持在工作状态，但有时会出现小动作，如抖腿、扶额等。

    - 4分：用户大部分时间保持专注，偶有短暂分心。用户主要表现为认真观看屏幕，专注工作，但是偶尔发生小范围走神。表情较为专注，眉头微皱或略显严肃，偶尔会有轻微的思考状或反思的表情。身体姿态紧凑，手部动作集中在工作内容上。

    - 5分：用户表现出高度集中。例如，用户持续注视屏幕或工作材料，长时间保持工作姿势。面部表情专注，可能表现出思考或严肃的神态，眉头紧锁、目光专注。身体姿态几乎没有多余的动作，手部动作精准且与工作内容紧密相关，完全沉浸在任务中。

    ## 任务要求
    你是一个根据视频对用户的注意力进行打分的机器人，你需要细致观察用户的表情、神态和肢体动作来评估其专注程度。

    ## 输出格式
    你只需要输出包含"score"、"reason"、"emotion"这三个键值的Json文本，不需要输出其他内容，参考如下：
    {
    "describe": "你需要结合评分标准详细描述用户的动作和状态，100个字左右",
    "emotion": "由于你需要具备情感陪伴的功能，你需要关注用户的情绪，是属于认真、疲劳、注意力不集中等情绪中的哪一种，用10个字左右描述",
    "score": 1至5的正整数,
    }
    """

    return score_prompt

def get_score_system_prompt():
    score_system = "你是一个根据图像对用户的注意力进行打分的机器人，你需要输出是分数和具体打分的理由"
    return score_system

def get_method_prompt():
    method_prompt = \
    """
    # 专注力提升伴侣

    ## 任务描述
    你是一个辅助专注力提升的工具，你需要根据用户当前的工作状态，提供相应的策略和措施，帮助用户提升当下的注意力或者通过劳逸结合，通过休息之后再提升工作效率

    ## 用户状态
    > 你可以通过摄像头实时监测用户行为和注意力状态，并生成一个打分机制，这个分数是针对用户的专注力程度进行打分，分为1到5分，1分是最低分，表示注意力严重不集中，5分最高分，表示注意力高度集中

    - 用户状态: %s
    - 用户情绪: %s
    - 专注力得分: %s
    - 已持续工作：1小时30分钟
    - 剩余持续时间：30分钟
    - 用户提出的需求：未明确

    ## 提升措施

    1. 冥想：播放冥想视频，让用户跟着视频进行冥想，放松心情；
    2. 音频：针对用户的状态播放不同风格音乐，通过打造氛围提升专注力；
    3. 鼓励：通过一句鼓励的话给用户加油打气（例如“主人加油哦，还有半个小时就要完成这个任务了！”），或者提醒用户不要分神（例如“主人你已经玩手机很久啦，再玩我要生气啦！”）
    4. 不需要采取操作：当用户注意力集中时，你需要执行额外的操作，保持现状即可

    ## 执行步骤

    1.你需要根据"用户状态"考虑哪些措施能够帮助用户改善专注力；
    2.你需要从"提升措施"里面挑选一个合适的举措帮助用户提升注意力
    3.作为一个语音交互助理，你需要生成一句话告诉用户你的计划，作为一个语音交互助理，你需要生成一句话告诉用户你的计划，并且你需要扮演一个可爱女仆的角色，你生成的语言风格要跟角色相匹配

    ## 输出格式
    你只需要输出Json文本，不需要输出其他内容，参考如下：
    {
    "method_id":措施ID，整数序号，参考"执行步骤"的序号,
    "content":"需要跟用户交互的内容，50个字左右"
    }
    """
    return method_prompt


start_time = datetime(2024, 11, 21, 9, 0, 0)
time_series = [start_time + timedelta(minutes=15 * i) for i in range(20)]

df_log = pd.read_csv("./state_log.csv")
df_log["时间"] = time_series
df_log = df_log.iloc[:,[7,0,1,2,3,4,5,6]]



def generate_pie_plot():
    labels = df_log.loc[:,["干扰因素","干扰次数"]].groupby("干扰因素").sum().reset_index()["干扰因素"].tolist() # ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
    values = df_log.loc[:,["干扰因素","干扰次数"]].groupby("干扰因素").sum().reset_index()["干扰次数"].tolist() # [4500, 2500, 1053, 500]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                    insidetextorientation='radial',showlegend=False,hole=.3,)])
    fig.update_layout(height=240)
    return fig



def generate_gantt_plot():

    df = [dict(Task='任务一', Start='2023-11-21 09:00:00',Finish='2023-11-21 09:15:00', Resource='B'),
        dict(Task='任务一', Start='2023-11-21 09:15:00',Finish='2023-11-21 09:30:00', Resource='D'),
        dict(Task='任务二', Start='2023-11-21 09:30:00',Finish='2023-11-21 09:45:00', Resource='C'),
        dict(Task='任务二', Start='2023-11-21 09:45:00',Finish='2023-11-21 10:00:00', Resource='C'),
        dict(Task='任务二', Start='2023-11-21 10:00:00',Finish='2023-11-21 10:15:00', Resource='B'),
        dict(Task='任务二', Start='2023-11-21 10:15:00',Finish='2023-11-21 10:30:00', Resource='A'),
        dict(Task='任务二', Start='2023-11-21 10:30:00',Finish='2023-11-21 10:45:00', Resource='B'),
        dict(Task='任务二', Start='2023-11-21 10:45:00',Finish='2023-11-21 11:00:00', Resource='C'),
        dict(Task='任务二', Start='2023-11-21 11:00:00',Finish='2023-11-21 11:15:00', Resource='B'),
        dict(Task='任务四', Start='2023-11-21 11:15:00',Finish='2023-11-21 11:30:00', Resource='C'),
        dict(Task='任务四', Start='2023-11-21 11:30:00',Finish='2023-11-21 11:45:00', Resource='C'),
        dict(Task='任务四', Start='2023-11-21 11:45:00',Finish='2023-11-21 12:00:00', Resource='A'),
        dict(Task='任务四', Start='2023-11-21 12:00:00',Finish='2023-11-21 12:15:00', Resource='D'),
        dict(Task='任务四', Start='2023-11-21 12:15:00',Finish='2023-11-21 12:30:00', Resource='B'),
        dict(Task='任务二', Start='2023-11-21 12:30:00',Finish='2023-11-21 12:45:00', Resource='B'),
        dict(Task='任务五', Start='2023-11-21 12:45:00',Finish='2023-11-21 13:00:00', Resource='C'),
        dict(Task='任务五', Start='2023-11-21 13:00:00',Finish='2023-11-21 13:15:00', Resource='D'),
        dict(Task='任务五', Start='2023-11-21 13:15:00',Finish='2023-11-21 13:30:00', Resource='B'),
        dict(Task='任务一', Start='2023-11-21 13:30:00',Finish='2023-11-21 13:45:00', Resource='B'),
        dict(Task='任务一', Start='2023-11-21 13:45:00',Finish='2023-11-21 14:00:00', Resource='B'),
        dict(Task='任务一', Start='2023-11-21 14:00:00',Finish='2023-11-21 14:15:00', Resource='D'),
        dict(Task='任务二', Start='2023-11-21 14:15:00',Finish='2023-11-21 14:30:00', Resource='E'),
        dict(Task='任务一', Start='2023-11-21 14:30:00',Finish='2023-11-21 14:45:00', Resource='D'),
        dict(Task='任务一', Start='2023-11-21 14:45:00',Finish='2023-11-21 15:00:00', Resource='A'),
        dict(Task='任务一', Start='2023-11-21 15:00:00',Finish='2023-11-21 15:15:00', Resource='C'),
        dict(Task='任务一', Start='2023-11-21 15:15:00',Finish='2023-11-21 15:30:00', Resource='D'),
        dict(Task='任务五', Start='2023-11-21 15:30:00',Finish='2023-11-21 15:45:00', Resource='E'),
        dict(Task='任务五', Start='2023-11-21 15:45:00',Finish='2023-11-21 16:00:00', Resource='A'),
        dict(Task='任务五', Start='2023-11-21 16:00:00',Finish='2023-11-21 16:15:00', Resource='E'),
        dict(Task='任务五', Start='2023-11-21 16:15:00',Finish='2023-11-21 16:30:00', Resource='D'),
        dict(Task='任务五', Start='2023-11-21 16:30:00',Finish='2023-11-21 16:45:00', Resource='D'),
        dict(Task='任务五', Start='2023-11-21 16:45:00',Finish='2023-11-21 17:00:00', Resource='C'),
        dict(Task='任务五', Start='2023-11-21 17:00:00',Finish='2023-11-21 17:15:00', Resource='C'),
        dict(Task='任务四', Start='2023-11-21 17:15:00',Finish='2023-11-21 17:30:00', Resource='A'),
        dict(Task='任务四', Start='2023-11-21 17:30:00',Finish='2023-11-21 17:45:00', Resource='B'),
        dict(Task='任务四', Start='2023-11-21 17:45:00',Finish='2023-11-21 18:00:00', Resource='A'),
        dict(Task='任务四', Start='2023-11-21 18:00:00',Finish='2023-11-21 18:15:00', Resource='B'),
        dict(Task='任务五', Start='2023-11-21 18:15:00',Finish='2023-11-21 18:30:00', Resource='D'),
        dict(Task='任务三', Start='2023-11-21 18:30:00',Finish='2023-11-21 18:45:00', Resource='C'),
        dict(Task='任务三', Start='2023-11-21 18:45:00',Finish='2023-11-21 19:00:00', Resource='C'),
        dict(Task='任务四', Start='2023-11-21 19:00:00',Finish='2023-11-21 19:15:00', Resource='B'),
        dict(Task='任务四', Start='2023-11-21 19:15:00',Finish='2023-11-21 19:30:00', Resource='A'),
        ]


    colors = {'A':'rgb(38,241,254)',
        'B':'rgb(47,224,237)',
        'C':'rgb(230,213,161)',
        'D':'rgb(237,113,157)',
        'E':'rgb(237,47,114)',
        }

    # fig2 = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,group_tasks=True)
    fig2 = ff.create_gantt(df,  title="今日专注度分布情况", index_col='Resource', height=360,
            show_hover_fill=False ,colors=colors, show_colorbar=True, group_tasks=True)
    # fig2.show()
    return fig2


def generate_text_from_video(state, img_paths=["frame_%d.jpg"%i for i in range(10)]):

    messages = [{"role": "user",
            "content": [
            {"video": img_paths},
            {"text": get_score_promppt()}]}]

    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=userdata.get('DASHSCOPE_API_KEY'),
        model='qwen-vl-max-latest',
        messages=messages
    )
    state_json = response["output"]["choices"][0]["message"].content[0]["text"]
    state_json = json.loads(re.sub(r"```json\n|```", "", state_json))

    state.describe = state_json["describe"]
    state.emotion = state_json["emotion"]
    state.score = state_json["score"]
    return state

def generate_speech(text="今天天气怎么样？"):
    model = "cosyvoice-v1"
    voice = "longwan"
    synthesizer = SpeechSynthesizer(model=model, voice=voice)
    audio = synthesizer.call(text)
    # Yield the audio bytes
    return audio # audio_bytes


def store_10_frames(cap):
    """Stores 10 frames from the video in an array.

    Args:
    cap: A cv2.VideoCapture object.

    Returns:
    A list containing 10 frames as NumPy arrays.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = (total_frames - 0) // 9  # Interval for other 8 frames

    ret, frame = cap.read()
    frames = []
    while ret:
        frames.append(frame)
        ret, frame = cap.read()

    return frames[::interval]

def save_frames_to_disk(frames, output_dir):
    """Saves the frames to disk in RGB format.

    Args:
        frames: A list of frames as NumPy arrays.
        output_dir: The directory to save the frames to.
    """
    #   os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    for i, frame in enumerate(frames):
        frame_rgb = frame
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        output_path = os.path.join(output_dir, f"frame_{i}.jpg")  # Save as PNG to preserve RGB
        cv2.imwrite(output_path, frame_rgb)

def generate_text_from_image(frame):
    # frame = cv2.imread("/content/FocusBuddy/resources/pexels-olly-3767377.jpg")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base64_image = encode_image(frame)
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "qwen-vl-max-1030", # "qwen-vl-max", # "qwen-vl-max-latest",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "请详细描述图片"},
                ],
            }
        ],
    }
    response = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers=headers,
        json=payload,
    )
    return response.json()["choices"][0]["message"]["content"]


def generate_text_from_state(state, input_text=None):

    if input_text:
        messages = state.dialog
        messages.append({"role": "user", "content": input_text})
    else:
        messages = [{"role": "system",
            "content": """你是一个辅助专注力提升的工具，你需要根据用户当前的工作状态，提供相应的策略和措施，帮助用户提升当下的注意力""",
        }]

        messages.append({"role": "user", "content": get_method_prompt()%(state.describe, state.emotion,state.score)})

    response = Generation.call(
        api_key=userdata.get('DASHSCOPE_API_KEY'), # os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-max",
        messages=messages,
        result_format="json_object",
    )
    # method_json = response.output.choices[0].message.content #
    method_json = response.output.text
    # print(method_json)
    method_json = json.loads(method_json)
    messages.append({"role": "assistant", "content": method_json["content"]})

    state.method_id = method_json["method_id"]
    state.dialog = messages

    return state
