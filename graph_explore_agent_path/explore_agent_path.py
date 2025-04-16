#Understanding how langchain graph for prompt execution works
# This will be simple raw graph traversing code to list down all the paths that can be traversed to get ans

# Problem Statement: LLM Agent Routing with conditional Paths, Cycles and Fallbacks
# We are making a LangChain agent that navigates tools and fallback strategies in a dynamic reasoning graph.
# It could:
    #1: May loop if doest reach an answer
    #2: May hit a dead end an backbrack
    #3: May try multiple tool in sequence befor answering

## Copyright (c) 2025, Nitin Chaube

import json
from collections import deque
def load_graph(path: str)-> dict:
    with open(path,'r') as file:
        graph = json.load(file)
    return graph

def explore_agent_paths(graph: dict, start_node: str = "START" ) -> None:
    def is_tool(node):
        return node not in ["START"] and not node.startswith("Q") and not node.startswith("A")
    
    queue = deque()
    queue.append((start_node, [start_node]))

    while queue:
        current, path = deque.popleft(queue)
        if current.startswith('A'):
            iscycle = len(path) != len(set(path))
            tools = [node for node in path if is_tool(node)]
            print(f"Path: {'->'.join(path)} | Tools: {', '.join(tools)} | Hops: {len(path)} | Cycle: {iscycle}")
            continue

        for neighbor in graph.get(current, []):
            if neighbor not in path or neighbor.startswith("A"):
                queue.append((neighbor, path + [neighbor]))



    
if __name__ == "__main__":
    graph = load_graph("./agent_tool_graph.json")
    explore_agent_paths(graph)
