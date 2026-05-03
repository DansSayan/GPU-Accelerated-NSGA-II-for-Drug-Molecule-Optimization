import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import cupy as cp
import cudf
from nsga2_gpu import run_nsga2_optimization, load_data
from visualize import create_plots

def run_optimization(pop_size, generations, mutation_std):
    """
    Run NSGA-II optimization with given parameters
    """
    try:
        pop_size = int(pop_size)
        generations = int(generations)
        mutation_std = float(mutation_std)
        
        # Run NSGA-II
        pareto_front_df, history = run_nsga2_optimization(
            pop_size=pop_size,
            generations=generations,
            mutation_std=mutation_std
        )
        
        # Save results
        pareto_front_df.to_csv("pareto_front.csv", index=False)
        
        # Generate plots using visualize.py
        plot_paths = create_plots(
            pareto_csv="pareto_front.csv",
            original_csv="processed_molecules.csv"
        )
        
        # Return results with plot images
        return (
            pareto_front_df,
            plot_paths,
            "pareto_front.csv",
            "✓ Optimization completed successfully!"
        )
    
    except Exception as e:
        return (
            pd.DataFrame({"Error": [str(e)]}),
            [],
            None,
            f"✗ Error: {str(e)}"
        )


with gr.Blocks(title="🧬 GPU-Accelerated Drug Molecule Optimizer") as interface:
    gr.Markdown("# 🧬 GPU-Accelerated Drug Molecule Optimizer")
    gr.Markdown("Optimize drug-like molecules using NSGA-II multi-objective optimization on GPU")

    with gr.Row():
        # Left column: controls
        with gr.Column(scale=1):
            pop_size = gr.Slider(100, 2000, value=500, step=100, label="Population Size")
            generations = gr.Slider(10, 200, value=50, step=10, label="Generations")
            mutation_std = gr.Slider(0.01, 1.0, value=0.1, step=0.05, label="Mutation Standard Deviation")
            run_btn = gr.Button("Run Optimization")
            status = gr.Textbox(label="Status", interactive=False)

        # Right column: plots
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Optimization Plots", columns=1, object_fit="contain", height=600)

    # Bottom: pareto results + download
    with gr.Row():
        with gr.Column():
            dataframe = gr.Dataframe(label="Pareto Front Results", interactive=False, wrap=False)
            download = gr.File(label="Download Pareto Results")

    run_btn.click(
        fn=run_optimization,
        inputs=[pop_size, generations, mutation_std],
        outputs=[dataframe, gallery, download, status]
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
