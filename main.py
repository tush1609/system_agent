import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch AI Chat UI")
    parser.add_argument(
        "--ui",
        choices=["chainlit", "streamlit"],
        required=True,
        help="Choose UI: chainlit or streamlit"
    )
    args = parser.parse_args()

    if args.ui == "chainlit":
        print("Launching Chainlit UI...")
        subprocess.run(["chainlit", "run", "ui/chainlit_adapter.py", "-w"])

    # print('HI')
    # Chainlit.register()
    # graph = Graph()
    # graph.get_result('create directory name poop in Documents directory')
