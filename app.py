import gradio as gr
from main import generate_dataset, save_jsonl

def run_pipeline(file, progress=gr.Progress()):
    try:  
        def update(p):  
            progress(p)  
        data = generate_dataset(file, update)  
        path = save_jsonl(data)  
        avg_score = round(  
            sum(d["score"] for d in data)/len(data), 3  
        ) if data else 0  
        return (  
            f"✅ Generated {len(data)} samples\n"  
            f"Average Quality Score: {avg_score}",  
            path  
        )  

    except Exception as e:  
        return f"❌ Error: {str(e)}", None

with gr.Blocks() as app:
    gr.Markdown("# Synthetic Dataset Generator")  

    file = gr.File(label="Upload PDF")  
    run_btn = gr.Button("Generate Dataset")  

    status = gr.Textbox(label="Status")  
    output = gr.File(label="Download Dataset")  

    run_btn.click(  
        run_pipeline,  
        inputs=file,  
        outputs=[status, output]  
     )
app.launch()
