import math

import pandas as pd

import gradio as gr
import datetime
import numpy as np
from models.lstm import model, get_preds_koef, train
import random


def get_time():
    return datetime.datetime.now()


# df = pd.DataFrame()
df=pd.read_csv('data/train_sfo_processed_2.csv')

def get_plot1(period=1):
    global df
    pred = get_preds_koef(df, period, model=model, target_column="revenue")
    print(pred)
    update = gr.LinePlot(
        value=pd.DataFrame({"weeks": [i for i in range(len(pred))], "revenue": pred}),
        # value=pd.DataFrame({"weeks": [i for i in range(len(pred))], "revenue":[random.randint(1, int(i)) for i in pred]}),
        x="weeks",
        y="revenue",
        title="disease rate over revenue",
        width=600,
        height=350,
    )
    return update
def get_plot2(period=1):
    global df
    pred = get_preds_koef(df, period, model=model, target_column="revenue")
    print(pred)
    update = gr.LinePlot(
        value=pd.DataFrame({"weeks": [i for i in range(len(pred))], "revenue": pred}),
        # value=pd.DataFrame({"weeks": [i for i in range(len(pred))], "revenue":[random.randint(1, int(i)) for i in pred]}),
        x="weeks",
        y="revenue",
        title="disease rate over revenue",
        width=600,
        height=350,
        tooltip="revenue"
    )
    return update
def corr_revenue(feature_name="wordstat"):
  X=df.loc[:, feature_name].tolist()
  y=df.loc[:, "revenue"].tolist()
  update = gr.ScatterPlot(
      value=pd.DataFrame({feature_name: X, "revenue": y}),
      # value=pd.DataFrame({"weeks": [i for i in range(len(pred))], "revenue":[random.randint(1, int(i)) for i in pred]}),
      x=feature_name,
      y="revenue",
      title="disease rate over revenue",
      width=600,
      height=350,
      tooltip="revenue",
  )
  return update

def file_load(input_file):
    # Process the uploaded file
    global df
    if (input_file.name.endswith('.xlsx')):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    # You can perform any specific operations on the file content here
    # print(df.columns)
    # return file_content


with gr.Blocks() as demo:
    # with gr.Column():
    file = gr.File(file_types=['.csv', '.xlsx'])

    # c_time2 = gr.Textbox(label="Current Time refreshed every second")
    gr.Textbox(
        "Change the value of the slider to automatically update the plot",
        label="",
    )
    period = gr.Slider(
        label="Period of plot", value=1, minimum=1 / 4, maximum=5, step=0.25
    )
    plot1 = gr.LinePlot(show_label=False)
    dropdown=gr.Dropdown(df.columns.tolist())

    plot2 = gr.ScatterPlot(show_label=False)
    # dep = demo.load(get_plot, None, plot)
    period.release(get_plot1, period, plot1)
    file.upload(file_load, inputs=file, outputs=None)
    dropdown.change(corr_revenue, inputs=dropdown, outputs=plot2)



if __name__ == "__main__":
    demo.launch()
