"""Pydantic models = the JSON schema we hand to OpenAI Structured Outputs.

Strict mode rules we follow: every field is required (optional values are `X | None`),
no free-form dicts. That is why Action is a flat model with one nullable field per argument.

KEY DIFFERENCE between the two agents lives here:
  ReactiveOutput -> the model may emit ONLY an action (no explicit reasoning).
  ReActOutput    -> the model MUST write a `thought` FIRST, then the action.
Field order matters: the model generates JSON left-to-right, so `thought` before `action`
means the reasoning is produced before the decision it justifies.
"""

from typing import Literal
from pydantic import BaseModel


class Action(BaseModel):
    tool: Literal["web_search", "read_file", "calculator", "finish"]
    query:  str| None
    path: str | None 
    expression: str | None
    answer: str | None 
    
class ReactiveOutput(BaseModel):
    action: Action


class ReActOutput(BaseModel):
    thought: str
    action: Action
    
    
TOOL_DOC = """Tools (set unused argument fields to null):
- web_search(query): look up product prices and facts. Returns text snippets.
- read_file(path): read a text file from the data directory, e.g. "order.txt".
- calculator(expression): evaluate arithmetic like "3*450 + 5*1200". Use it for ALL math.
- finish(answer): give the final answer to the user. Call this once you have everything."""