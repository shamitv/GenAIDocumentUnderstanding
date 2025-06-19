"""
agentic_pdf_assessor_team_vision.py
-----------------------------------
• Planner / Executor / Reporter / User agents in an AutoGen *Team*
• PDF pages rendered to PNG → data-URL → sent to GPT-4o as vision messages
"""

import asyncio, base64, datetime as dt, io, json, pathlib, sys
from typing import List, Dict

import fitz  # PyMuPDF

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.team import Team, Task
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ────────────────────────────────────────────────────────────────────
# 0 · Vision-capable model client
# ────────────────────────────────────────────────────────────────────
model_client = OpenAIChatCompletionClient(model="gpt-4o")  # vision-enabled


# ────────────────────────────────────────────────────────────────────
# 1 · Utilities: PDF ⇒ list[dict]  (OpenAI image objects)
# ────────────────────────────────────────────────────────────────────
def pdf_to_image_messages(pdf_path: str) -> List[Dict]:
    """Return OpenAI 'image_url' message objects for every page of the PDF."""
    imgs: List[Dict] = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=220)                       # decent resolution
        buf = io.BytesIO(pix.tobytes("png"))
        b64 = base64.b64encode(buf.getvalue()).decode()
        dataurl = f"data:image/png;base64,{b64}"
        imgs.append({"type": "image_url", "image_url": {"url": dataurl}})
    doc.close()
    return imgs


# ────────────────────────────────────────────────────────────────────
# 2 · Agents
# ────────────────────────────────────────────────────────────────────
planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message=(
        "You output STRICT JSON only.\n"
        "When clarification is needed, create a task for `user`.\n"
        "When ready to finish, create a task for `reporter` then return DONE."
    ),
)

executor = AssistantAgent(
    name="executor",
    model_client=model_client,
    system_message=(
        "Execute ONE task using ONLY the provided PDF page images.\n"
        "Return STRICT JSON {result:str, citations:[{page:int, quote:str}]}.\n"
        "Use short quotes from the visible text in images (≤200 chars)."
    ),
)

reporter = AssistantAgent(
    name="reporter",
    model_client=model_client,
    system_message=(
        "Create a Markdown report with numbered footnote citations (¹,²,³…).\n"
        "Input: objective, finished_tasks[{result, citations}].\n"
        "Return STRICT JSON {report_markdown:str}"
    ),
)

user = UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=0,
    system_message="Answer clarification questions briefly and factually.",
)


# ────────────────────────────────────────────────────────────────────
# 3 · Team builder
# ────────────────────────────────────────────────────────────────────
def build_team(img_msgs: List[Dict], objective: str) -> Team:
    root = Task(
        name="root",
        agent=planner,
        instructions=json.dumps(
            {
                "mode": "INIT",
                "today": dt.date.today().isoformat(),
                "objective": objective,
                "note": "The PDF pages are provided below as images.",
            }
        ),
        # first message(s) to the planner include the images:
        opener_messages=img_msgs,
        expected_output_key="final_report",
    )
    return Team(
        name="PDF-Vision-Assessment",
        agents=[planner, executor, reporter, user],
        tasks=[root],
        allow_parallel=False,
    )


# ────────────────────────────────────────────────────────────────────
# 4 · Orchestrator wrapper
# ────────────────────────────────────────────────────────────────────
async def assess_pdf(pdf_path: str, objective: str) -> str:
    img_messages = pdf_to_image_messages(pdf_path)
    tm = build_team(img_messages, objective)
    await tm.run()                                     # interactive — may ask user
    report = tm.tasks["root"].output.get("final_report")
    if not report:
        raise RuntimeError("No final report was produced!")
    return report


# ────────────────────────────────────────────────────────────────────
# 5 · CLI / demo
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_file= None

    if len(sys.argv) < 3:
        print("Usage: python agentic_pdf_assessor_team_vision.py <PDF> <objective>")
        sys.exit(1)
    else:
        print("Running agentic PDF assessor with default input…")
        pdf_file = pathlib.Path("data/test_pdfs/sample.pdf").expanduser()
        objective = ("WHO’s recommended daily limit for free "
                     "sugar intake and give at least two health "
                     "risks of excessive sugar consumption?")


    pdf_file = pathlib.Path(sys.argv[1]).expanduser()
    if not pdf_file.is_file():
        print(f"PDF not found: {pdf_file}")
        sys.exit(1)

    objective = " ".join(sys.argv[2:])

    print("\n=== Starting interactive assessment… ===\n")
    md_report = asyncio.run(assess_pdf(str(pdf_file), objective))
    print("\n=== FINAL MARKDOWN REPORT ===\n")
    print(md_report)
