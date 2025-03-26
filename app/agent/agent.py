# agent.py
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class Agent:
    def __init__(self, tools): #), assistant):
        """
        Initialize the agent with the given tools and assistant.
        This sets up the langgraph state graph.
        """
        self.builder = StateGraph(MessagesState)
        self.sys_msg = SystemMessage(content="You are a helpful assistant tasked with fetching news.")
        
        self.tools = tools
        # self.assistant = assistant
        self._initialize_graph()

    def _initialize_graph(self):
        """Initialize graph structure in langgraph"""
        # Define nodes: assistant node and tool node.
        self.builder.add_node("assistant", self.assistant)
        self.builder.add_node("tools", ToolNode(self.tools))
        # Define the initial edge from START to the assistant node.
        self.builder.add_edge(START, "assistant")
        # Define conditional edges from assistant based on message content.
        self.builder.add_conditional_edges("assistant", tools_condition)
        # Connect tools back to assistant (ReAct)
        self.builder.add_edge("tools", "assistant")
        # Compile the state graph.
        self.react_graph = self.builder.compile()

    def assistant(self, state: MessagesState) -> str:
        """
        Process the user's text by updating the graph.
        In a real implementation, you would update the MessagesState,
        trigger the graph execution, and capture the response.
        """
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

        llm_with_tools = llm.bind_tools(self.tools, parallel_tool_calls=False)
        return {"messages": [llm_with_tools.invoke([self.sys_msg] + state["messages"])]}


    def process_message(self, user_input: str) -> str:
        """Processes the user input"""
        messages = [HumanMessage(content=user_input)]

        messages = self.react_graph.invoke({"messages": messages})


        return messages['messages'][-1].content
