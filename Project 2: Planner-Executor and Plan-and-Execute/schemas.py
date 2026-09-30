"""Pydantic models. Two layers, on purpose:

LAYER 1 - the plan shape the LLM must produce (Plan / Step / StepArgs / ReplanDecision).
   Handed to OpenAI Structured Outputs (strict mode), so it must be simple:
   every field required, no free-form dicts, `tool` is an enum. `args` is one flat object with
   nullable fields (same pattern as Project 1's Action).

LAYER 2 - one STRICT args model per tool (WebSearchArgs, ...), used AFTER decoding to check that
   the step supplies exactly the arguments its tool needs. Strict decoding guarantees valid JSON;
   it does NOT guarantee the plan makes sense. Layer 2 is our first "is this plan executable?" check.
"""


from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


ToolName = Literal["web_search", "read_file", "write_file", "calculator",
                   "db_query", "send_email", "llm_reason"]

class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebSearchArgs(_Args):
    query: str

class ReadFileArgs(_Args):
    path: str

class WriteFileArgs(_Args):
    path: str
    content: str

class CalculatorArgs(_Args):
    expression: str

class DbQueryArgs(_Args):
    collection: Literal["products", "orders"]
    field: Literal["name", "sku", "order_id"]
    value: str

class SendEmailArgs(_Args):
    to: str
    subject: str
    body: str

class LlmReasonArgs(_Args):
    instruction: str


# StepArgs holds the arguments for every tool in one flat object. Strict Structured Outputs needs one fixed schema, so each step returns all 11 fields, and the ones its tool doesn't use are null.

# A calculator step from the LLM looks like this:

# StepArgs(query=None, path=None, content=None, expression="{{s5}}",
#          collection=None, field=None, value=None, to=None,
#          subject=None, body=None, instruction=None)
# non_null() drops the None fields and keeps only the arguments the step actually supplies:

# {"expression": "{{s5}}"}

class StepArgs(BaseModel):
    """Flat union of every tool's arguments. Unused fields MUST be null."""
    query: str | None
    path: str | None
    content: str | None
    expression: str | None
    collection: str | None
    field: str | None
    value: str | None
    to: str | None
    subject: str | None
    body: str | None
    instruction: str | None
    
    def non_null(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class Step(BaseModel):
    id: str = Field(description="Unique id such as s1, s2, s3")
    description: str = Field(description="One line: what this step achieves")
    depends_on: list[str] = Field(description="Ids of steps whose OUTPUT this step needs; [] if none")
    tool: ToolName
    args: StepArgs


class Plan(BaseModel):
    goal: str
    steps: list[Step]
    answer_step: str = Field(description="Id of the step whose output is the final answer to the user")


class ReplanDecision(BaseModel):
    """Returned by the replanner after a step. This is what makes Plan-and-Execute *dynamic*."""
    decision: Literal["continue", "revise", "finish"]
    reason: str
    new_steps: list[Step] = Field(description="If revise: the COMPLETE list of remaining steps. Else []")
    answer_step: str | None = Field(description="If revise: id of the answer step. Else null")
    final_answer: str | None = Field(description="If finish: the final answer for the user. Else null")
