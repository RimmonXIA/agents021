import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

# Load environment variables (e.g., DEEPSEEK_API_KEY)
load_dotenv()

def main():
    # 1. Define specialized agents
    research_agent = Agent(
        name="Researcher",
        role="Search the web for information",
        model=DeepSeek(id="deepseek-chat"),
        # You can add tools here, e.g., DuckDuckGo, Wikipedia, etc.
        # tools=[DuckDuckGoTools()],
        instructions="You are a senior researcher. Provide clear and concise facts.",
        show_tool_calls=True,
    )

    writer_agent = Agent(
        name="Writer",
        role="Write cohesive summaries based on research",
        model=DeepSeek(id="deepseek-chat"),
        instructions="You are an expert writer. Synthesize information into a well-written summary.",
    )

    # 2. Coordinate them as a team
    team_agent = Agent(
        name="Team Lead",
        team=[research_agent, writer_agent],
        instructions="You are a team that researches topics and writes articles together. Delegate research to the Researcher, and writing to the Writer.",
        show_tool_calls=True,
        markdown=True,
    )

    # 3. Run the multi-agent system
    print("Starting multi-agent task...")
    team_agent.print_response(
        "Research the top 3 capabilities of modern multi-agent systems and write a short summary about them.",
        stream=True
    )

if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Please set your DEEPSEEK_API_KEY in the .env file or environment variables before running.")
    else:
        main()
