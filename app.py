"""Run the NBA Stats Agent in a local Gradio chat."""

import logging

import gradio as gr

from agent import stream_ask


def respond(question, history):
    sql = ""
    activity = ""
    try:
        for answer, queries, activity in stream_ask(question, history):
            sql = "\n\n".join(queries)
            yield answer, sql, activity
    except FileNotFoundError as error:
        answer = str(error)
    except Exception:
        logging.exception("NBA question failed")
        answer = "I couldn't answer that. Check `OPENAI_API_KEY` in `.env` and try again. See the terminal for details."
    else:
        return
    yield answer, sql, f"{activity}\n\nRun stopped."


with gr.Blocks(title="NBA Stats Agent", analytics_enabled=False) as demo:
    sql = gr.Code(language="sql", label="Queries from the latest answer", interactive=False, render=False)
    thinking = gr.Textbox(label="Agent activity", lines=14, interactive=False, show_label=False, render=False)
    with gr.Row():
        with gr.Column(scale=2):
            gr.ChatInterface(
                respond,
                title="NBA Stats Agent",
                examples=[
                    "Who scored the most points in a single game since 2010?",
                    "Who led the 2023-24 playoffs in total assists?",
                    "How many three-pointers did Stephen Curry make in the 2015-16 regular season?",
                ],
                additional_outputs=[sql, thinking],
                fill_height=False,
                api_name="ask",
            )
        with gr.Column(scale=1), gr.Tabs():
            with gr.Tab("Thinking"):
                thinking.render()
            with gr.Tab("SQL used"):
                sql.render()


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
