import gradio as gr
def file_load(input_file):
    # Process the uploaded file
    file_content = input_file.read().decode("utf-8")

    # You can perform any specific operations on the file content here

    return file_content


io = gr.Interface(fn=file_load, inputs="file", outputs="text")
io.launch()
